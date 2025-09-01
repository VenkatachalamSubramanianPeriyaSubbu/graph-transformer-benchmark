import numpy as np
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
import torch
from torch_geometric.utils import subgraph
from torch_geometric.nn import TopKPooling
from graph_transformer_pytorch import GraphTransformer

def balanced_kmeans(X, K, iters=10, seed=0):
    """
    Balanced KMeans clustering.
    Args:
        X: [N, d] data points
        K: number of clusters
        iters: number of iterations
        seed: random seed
    Returns:
        clusters: list of K arrays, each containing the indices of points in that cluster
        assign: [N] array, assign[i] is the cluster index of point i
    """
    N, d = X.shape
    m = N // K
    r = N % K

    # capacity distribution: r clusters get (m+1), rest get m
    capacities = np.array([m+1 if i < r else m for i in range(K)])

    # init centroids with vanilla kmeans++
    km = KMeans(n_clusters=K, n_init=1, max_iter=10, random_state=seed).fit(X)
    C = km.cluster_centers_

    for _ in range(iters):
        # repeat centroids according to capacities
        centroids_repeated = np.vstack([
            np.repeat(C[i][None, :], capacities[i], axis=0)
            for i in range(K)
        ])  # [N, d]

        # compute distances
        D = ((X[:, None, :] - centroids_repeated[None, :, :])**2).sum(-1)  # [N, N]

        # assignment
        r_idx, c_idx = linear_sum_assignment(D)
        slot2cluster = np.concatenate([
            np.full(capacities[i], i) for i in range(K)
        ])
        assign = slot2cluster[c_idx]

        # recompute centroids
        C = np.vstack([X[assign == k].mean(0) for k in range(K)])

    clusters = [np.where(assign == k)[0] for k in range(K)]
    return clusters, assign


class SwinGraphBlock(torch.nn.Module):
    """
    Swin Graph Block with clustered attention which is performed inside the cluster.
    Args:
        in_dim: input feature dimension
        heads: number of attention heads
        depth: number of transformer layers
        attn_dropout: attention dropout rate
        ff_dropout: feedforward dropout rate
    Returns:
        x_out: [N, C] output node features
    """
    def __init__(self, in_dim, heads=8, depth=1, attn_dropout=0.1, ff_dropout=0.1):
        super().__init__()
        self.gt = GraphTransformer(
            dim=in_dim, depth=depth, heads=heads,
            edge_dim=None, with_feedforwards=True,
            gated_residual=True,
            rel_pos_emb=True,
            accept_adjacency_matrix=True
        )

    def forward(self, x, edge_index, clusters):
        """
        Forward pass with clustered attention.
        Args:
            x: [N, C] node features
            edge_index: [2, E] edge indices
            clusters: list of LongTensor node indices per cluster
        Returns:
            x_out: [N, C] output node features
        """
        x_out = x.clone()

        for idxs in clusters:
            # Ensure indices are long tensors on the correct device
            idxs = torch.as_tensor(idxs, dtype=torch.long, device=x.device)
            
            # Skip empty clusters
            if len(idxs) == 0:
                continue
            
            # Skip single-node clusters (no self-attention needed)
            if len(idxs) == 1:
                continue

            # restrict edges to subgraph
            mask = torch.zeros(x.size(0), dtype=torch.bool, device=x.device)
            mask[idxs] = True
            sub_ei, _ = subgraph(mask, edge_index, relabel_nodes=True)

            # local features
            x_sub = x[idxs]                          # [num_nodes_in_cluster, dim]
            num_nodes = idxs.numel()

            # build dense adjacency [seq, seq]
            adj = torch.zeros(num_nodes, num_nodes, device=x.device)
            if sub_ei.size(1) > 0:  # Check if there are edges
                adj[sub_ei[0], sub_ei[1]] = 1

            # add batch dimension for GraphTransformer
            x_sub_in = x_sub.unsqueeze(0)            # [1, seq, dim]
            adj_in = adj.unsqueeze(0)                # [1, seq, seq]

            # run transformer
            try:
                x_sub_out, _ = self.gt(x_sub_in, adj_mat=adj_in)
                # remove batch dimension
                x_sub_out = x_sub_out.squeeze(0)         # [seq, dim]
                # write back with residual
                x_out[idxs] = x_out[idxs] + x_sub_out
            except:
                # Skip this cluster if transformer fails
                continue

        return x_out # [N, C]
    
class SwinGraphStage(torch.nn.Module):
    """
    Swin Graph Stage: windowed attention + shifted window attention + pooling.
    Args:
        in_dim: input feature dimension
        out_dim: output feature dimension after pooling
        heads: number of attention heads
        depth: number of transformer layers per block
        pool_ratio: ratio for TopKPooling
    Returns:
        x: [N', out_dim] output node features after pooling
        edge_index: [2, E'] new edge indices after pooling
        batch: [N'] new batch assignment after pooling
        perm: [N'] indices of nodes kept after pooling
        score: [N'] scores of nodes kept after pooling
    """
    def __init__(self, in_dim, out_dim, heads=8, depth=1, pool_ratio=0.5):
        super().__init__()
        self.window_block = SwinGraphBlock(in_dim, heads=heads, depth=depth)
        self.shift_block  = SwinGraphBlock(in_dim, heads=heads, depth=depth)
        self.expand = torch.nn.Linear(in_dim, out_dim)  # widen channels
        self.pool = TopKPooling(out_dim, ratio=pool_ratio)  # downsample nodes

    def forward(self, x, edge_index, clusters_fn, shift_clusters_fn, batch=None):
        """
        Args:
            x: [N, in_dim] node features
            edge_index: [2, E] adjacency in COO format
            clusters_fn: callable, returns clustering assignment per node
            shift_clusters_fn: callable, returns shifted clustering
            batch: [N] batch assignment (for multiple graphs)
        Returns:
            x: [N', out_dim] node features after pooling
            edge_index: [2, E'] new edge indices after pooling
            batch: [N'] new batch assignment after pooling
            perm: [N'] indices of nodes kept after pooling
            score: [N'] scores of nodes kept after pooling
        """

        # 1) window attention (partition into clusters)
        clusters = clusters_fn(x)  
        x = self.window_block(x, edge_index, clusters) # updated node features with residual

        # 2) shifted windows (different partitioning)
        shift_clusters = shift_clusters_fn(x)
        x = self.shift_block(x, edge_index, shift_clusters)

        # 3) channel expansion
        x = self.expand(x)

        # 4) pooling (downsample nodes + edges + batch info)
        x, edge_index, _, batch, perm, score = self.pool(x, edge_index, None, batch)

        return x, edge_index, batch, perm, score


def make_balanced_clusterers(K, seed_base=0):
    """
    Create balanced clustering functions for SwinGraphStage.
    Args:
        K: number of clusters
        seed_base: random seed base for reproducibility
    Returns:
        clusters_fn: callable, returns clustering assignment per node
        shift_clusters_fn: callable, returns shifted clustering assignment
    """
    def clusters_fn(x):
        N = x.size(0)
        # Adjust K if we have fewer nodes than clusters
        effective_K = min(K, N)
        
        if effective_K <= 1:
            # If only 1 cluster or fewer nodes than clusters, return all nodes in one cluster
            return [torch.arange(N, dtype=torch.long)]
        
        X = x.detach().cpu().numpy()
        try:
            clusters, _ = balanced_kmeans(X, K=effective_K, iters=2, seed=seed_base)  # Reduced iterations
            # Filter out empty clusters and convert to long tensors
            non_empty_clusters = [torch.tensor(c, dtype=torch.long) for c in clusters if len(c) > 0]
            return non_empty_clusters if non_empty_clusters else [torch.arange(N, dtype=torch.long)]
        except:
            # Fallback: simple partitioning if clustering fails
            chunk_size = N // effective_K
            remainder = N % effective_K
            clusters = []
            start = 0
            for i in range(effective_K):
                end = start + chunk_size + (1 if i < remainder else 0)
                if start < N:
                    clusters.append(torch.arange(start, min(end, N), dtype=torch.long))
                start = end
            return clusters
    
    def shift_clusters_fn(x):
        N = x.size(0)
        effective_K = min(K, N)
        
        if effective_K <= 1:
            return [torch.arange(N, dtype=torch.long)]
        
        X = x.detach().cpu().numpy()
        # Add smaller jitter to avoid numerical issues
        X = X + 0.001 * np.random.RandomState(seed_base+1).randn(*X.shape)
        
        try:
            clusters, _ = balanced_kmeans(X, K=effective_K, iters=2, seed=seed_base+1)
            non_empty_clusters = [torch.tensor(c, dtype=torch.long) for c in clusters if len(c) > 0]
            return non_empty_clusters if non_empty_clusters else [torch.arange(N, dtype=torch.long)]
        except:
            # Fallback: shifted simple partitioning
            shift = N // (2 * effective_K) if N > effective_K else 0
            chunk_size = N // effective_K
            remainder = N % effective_K
            clusters = []
            start = shift
            for i in range(effective_K):
                end = start + chunk_size + (1 if i < remainder else 0)
                if start < N:
                    cluster_indices = torch.arange(start, min(end, N), dtype=torch.long) % N
                    clusters.append(cluster_indices)
                start = end
            return clusters
    
    return clusters_fn, shift_clusters_fn


class SwinGraphTransformer(torch.nn.Module):
    def __init__(self, in_dim, dims=(96, 192, 384), heads=(4, 8, 8), Ks=(8, 4, 2), pool_ratio=0.5):
        super().__init__()
        self.proj = torch.nn.Linear(in_dim, dims[0])

        # Keep track of clustering sizes K per stage
        self.Ks = Ks

        # Build stages
        self.stages = torch.nn.ModuleList()
        for i in range(len(dims) - 1):
            stage = SwinGraphStage(
                in_dim=dims[i],
                out_dim=dims[i + 1],
                heads=heads[i],
                depth=2,
                pool_ratio=pool_ratio
            )
            self.stages.append(stage)

        # Final readout
        self.readout = torch.nn.Sequential(
            torch.nn.LayerNorm(dims[-1]),
            torch.nn.Linear(dims[-1], dims[-1])
        )

    def forward(self, x, edge_index, batch=None):
        # Ensure edge_index is long tensor
        edge_index = edge_index.long()
        
        # Initial projection
        x = self.proj(x)

        # Track node permutations through pooling stages
        perm = torch.arange(x.size(0), device=x.device)
        
        # Pass through stages with clustering
        for s, stage in enumerate(self.stages):
            # Create cluster functions per stage (different K and seeds)
            clusters_fn, shift_clusters_fn = make_balanced_clusterers(
                K=self.Ks[s],
                seed_base=1337 + 10 * s
            )

            # Each stage uses clustering functions
            x, edge_index, batch, stage_perm, score = stage(
                x,
                edge_index,
                clusters_fn=clusters_fn,
                shift_clusters_fn=shift_clusters_fn,
                batch=batch
            )
            
            # Update global permutation to track which original nodes remain
            perm = perm[stage_perm]

        # Final representation
        return self.readout(x), perm