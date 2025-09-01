import numpy as np
import torch
from torch_geometric.utils import subgraph
from torch_geometric.nn import TopKPooling
from graph_transformer_pytorch import GraphTransformer
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment

def balanced_kmeans(X, cluster_size, iters=10, seed=0):
    """
    Balanced KMeans clustering with fixed cluster size.
    Args:
        X: [N, d] data points
        cluster_size: fixed number of nodes per cluster
        iters: number of iterations
        seed: random seed
    Returns:
        clusters: list of arrays, each containing the indices of points in that cluster
        assign: [N] array, assign[i] is the cluster index of point i
    """
    N, _ = X.shape
    K = N // cluster_size  # number of clusters
    r = N % cluster_size   # remaining nodes
    
    # All clusters have exactly cluster_size nodes, except we ignore remainder
    # (alternative: could distribute remainder across clusters)
    N_used = K * cluster_size
    X_used = X[:N_used]  # only use nodes that fit exactly into clusters

    # init centroids with vanilla kmeans++
    km = KMeans(n_clusters=K, n_init=1, max_iter=10, random_state=seed).fit(X_used)
    C = km.cluster_centers_

    for _ in range(iters):
        # each cluster gets exactly cluster_size slots
        centroids_repeated = np.vstack([
            np.repeat(C[i][None, :], cluster_size, axis=0)
            for i in range(K)
        ])  # [N_used, d]

        # compute distances
        D = ((X_used[:, None, :] - centroids_repeated[None, :, :])**2).sum(-1)  # [N_used, N_used]

        # assignment using Hungarian algorithm
        r_idx, c_idx = linear_sum_assignment(D)
        slot2cluster = np.concatenate([
            np.full(cluster_size, i) for i in range(K)
        ])
        assign_used = slot2cluster[c_idx]

        # recompute centroids
        C = np.vstack([X_used[assign_used == k].mean(0) for k in range(K)])

    # create final assignment for all nodes
    assign = np.full(N, -1)  # -1 for unused nodes
    assign[:N_used] = assign_used
    
    clusters = [np.where(assign == k)[0] for k in range(K)]
    return clusters, assign


class SwinGraphBlock(torch.nn.Module):
    """
    Swin Graph Block with clustered attention performed inside each cluster.
    Args:
        in_dim: input feature dimension
        heads: number of attention heads
        depth: number of transformer layers
        attn_dropout: attention dropout rate
        ff_dropout: feedforward dropout rate
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
            idxs = torch.as_tensor(idxs, dtype=torch.long, device=x.device)

            # restrict edges to subgraph
            mask = torch.zeros(x.size(0), dtype=torch.bool, device=x.device)
            mask[idxs] = True
            sub_ei, _ = subgraph(mask, edge_index, relabel_nodes=True)

            # local features
            x_sub = x[idxs]
            num_nodes = idxs.numel()

            # build dense adjacency [seq, seq]
            adj = torch.zeros(num_nodes, num_nodes, device=x.device)
            if sub_ei.size(1) > 0:  # check if there are edges
                adj[sub_ei[0], sub_ei[1]] = 1

            # add batch dimension for GraphTransformer
            x_sub_in = x_sub.unsqueeze(0)            # [1, N_sub, dim]
            adj_in = adj.unsqueeze(0)                # [1, N_sub, seq]

            x_sub_out, _ = self.gt(x_sub_in, adj_mat=adj_in)
            x_sub_out = x_sub_out.squeeze(0) # remove batch dimension
            x_out[idxs] = x_out[idxs] + x_sub_out # residual

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
    """
    def __init__(self, in_dim, out_dim, heads=8, depth=1, pool_ratio=0.5):
        super().__init__()
        self.window_block = SwinGraphBlock(in_dim, heads=heads, depth=depth)
        self.mlp_1 = torch.nn.Sequential(
            torch.nn.Linear(in_dim, in_dim),
            torch.nn.GELU(),
            torch.nn.Linear(in_dim, in_dim)
        )
        self.mlp_2 = torch.nn.Sequential(
            torch.nn.Linear(in_dim, in_dim),
            torch.nn.GELU(),
            torch.nn.Linear(in_dim, in_dim)
        )
        self.shift_block = SwinGraphBlock(in_dim, heads=heads, depth=depth)
        self.expand = torch.nn.Linear(in_dim, out_dim)  # widen channels
        self.pool = TopKPooling(out_dim, ratio=pool_ratio)  # downsample nodes

    def forward(self, x, edge_index, cluster_size, batch=None):
        """
        Args:
            x: [N, in_dim] node features
            edge_index: [2, E] adjacency in COO format
            cluster_size: fixed number of nodes per cluster
            batch: [N] batch assignment (for multiple graphs)
        Returns:
            x: [N', out_dim] node features after pooling
            edge_index: [2, E'] new edge indices after pooling
            batch: [N'] new batch assignment after pooling
            perm: [N'] indices of nodes kept after pooling
            score: [N'] scores of nodes kept after pooling
        """
        # Window attention (partition into clusters)
        X = x.detach().cpu().numpy()
        clusters, _ = balanced_kmeans(X, cluster_size=cluster_size, seed=42)
        x = self.window_block(x, edge_index, clusters)

        # MLP with residual
        x = x + self.mlp_1(x)

        # Shifted windows (different partitioning)
        X_shift = x.detach().cpu().numpy()
        shift_clusters, _ = balanced_kmeans(X_shift, cluster_size=cluster_size, seed=43)
        x = self.shift_block(x, edge_index, shift_clusters)

        # MLP with residual
        x = x + self.mlp_2(x)

        # Channel expansion like in SwinVisionTransformer
        x = self.expand(x)

        # Pooling (downsample nodes + edges + batch info)
        x, edge_index, _, batch, perm, score = self.pool(x, edge_index, None, batch)

        return x, edge_index, batch, perm, score


class SwinGraphTransformer(torch.nn.Module):
    """
    Swin Graph Transformer with fixed cluster size throughout all stages.
    Args:
        in_dim: input feature dimension
        dims: tuple of feature dimensions per stage
        heads: tuple of attention heads per stage
        cluster_size: fixed number of nodes per cluster (same for all stages)
        pool_ratio: pooling ratio for downsampling
    """
    def __init__(self, in_dim, dims=(96, 192, 384), heads=(4, 8, 8), cluster_size=8, pool_ratio=0.5):
        super().__init__()
        self.proj = torch.nn.Linear(in_dim, dims[0])
        self.cluster_size = cluster_size

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
        self.final_MLP = torch.nn.Sequential(
            torch.nn.LayerNorm(dims[-1]),
            torch.nn.Linear(dims[-1], dims[-1])
        )

    def forward(self, x, edge_index, batch=None):
        """
        Args:
            x: [N, in_dim] node features
            edge_index: [2, E] edge indices
            batch: [N] batch assignment (optional)
        Returns:
            x: [N_final, out_dim] final node features
            perm: [N_final] indices of nodes kept after all pooling
        """
        # Ensure edge_index is long tensor
        edge_index = edge_index.long()
        
        # Initial projection
        x = self.proj(x)

        # Track node permutations through pooling stages
        perm = torch.arange(x.size(0), device=x.device)
        
        # Pass through stages with same cluster size
        for stage in self.stages:
            x, edge_index, batch, stage_perm, score = stage(
                x, edge_index, self.cluster_size, batch
            )
            
            # Update global permutation to track which original nodes remain
            perm = perm[stage_perm]

        # Final representation
        return self.final_MLP(x), perm