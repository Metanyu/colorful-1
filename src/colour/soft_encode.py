from sklearn.neighbors import NearestNeighbors
# from src.colour.quantize import get_ab_centroids
import numpy as np

import torch

# Pre-compute nearest neighbors index for faster lookup
_cached_nn = None
_cached_centroids = None

def _get_cached_nn(centroids, neighbours=5):
    global _cached_nn, _cached_centroids
    if _cached_nn is None or _cached_centroids is None or not torch.equal(centroids.cpu(), _cached_centroids):
        _cached_centroids = centroids.cpu()
        _cached_nn = NearestNeighbors(n_neighbors=neighbours, algorithm='ball_tree').fit(centroids.cpu().numpy())
    return _cached_nn

def soft_encode(ab, centroids, neighbours=5):
    n, h, w, c = ab.shape
    original_shape = (n, h, w)
    ab_flat = ab.reshape(-1, 2)
    
    # Use cached nearest neighbors for faster lookup
    nn = _get_cached_nn(centroids, neighbours)
    distances_np, indices_np = nn.kneighbors(ab_flat.cpu().numpy())
    
    distances = torch.from_numpy(distances_np).to(ab.device)
    indices = torch.from_numpy(indices_np).to(ab.device)
    
    # Gaussian weighting
    weights = torch.exp(-distances**2 / (2 * 5**2))
    weights = weights / torch.sum(weights, dim=-1, keepdim=True)

    # Sparse assignment - more memory efficient
    encoding = torch.zeros((ab_flat.shape[0], centroids.shape[0]), dtype=torch.float32, device=ab.device)
    batch_indices = torch.arange(ab_flat.shape[0], device=ab.device)[:, None].expand_as(indices)
    encoding[batch_indices, indices] = weights

    encoding = encoding.reshape(n, h, w, -1)
    return encoding


def soft_encode_fast(ab, centroids, neighbours=5):
    """Faster GPU-optimized version using batched distance computation."""
    n, h, w, c = ab.shape
    ab_flat = ab.reshape(-1, 2)  # (N*H*W, 2)
    
    # Compute distances in chunks to avoid memory issues
    chunk_size = 16384  # Process 16k pixels at a time
    num_chunks = (ab_flat.shape[0] + chunk_size - 1) // chunk_size
    
    encoding = torch.zeros((ab_flat.shape[0], centroids.shape[0]), dtype=torch.float32, device=ab.device)
    
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, ab_flat.shape[0])
        ab_chunk = ab_flat[start_idx:end_idx]
        
        # Compute pairwise distances for this chunk
        distances = torch.cdist(ab_chunk, centroids)  # (chunk, 326)
        
        # Get top-k nearest
        distances, indices = torch.topk(distances, k=neighbours, dim=-1, largest=False)
        
        # Gaussian weighting
        weights = torch.exp(-distances**2 / 50.0)  # 2 * 5^2 = 50
        weights = weights / weights.sum(dim=-1, keepdim=True)
        
        # Assign weights
        batch_idx = torch.arange(ab_chunk.shape[0], device=ab.device)[:, None].expand_as(indices)
        encoding[start_idx:end_idx].scatter_(1, indices, weights)
    
    return encoding.reshape(n, h, w, -1)

if __name__ == "__main__":
    hull = torch.from_numpy(np.load("data/hull.npy"))
    ab = np.array([[[[50, -51], [-52,50]], [[50, -51], [-52,50]]], [[[50, -51], [-52,50]], [[50, -51], [-52,50]]], [[[50, -51], [-52,50]], [[50, -51], [-52,50]]]], dtype=np.float32)
    ab = torch.from_numpy(ab)
    encoding = soft_encode(ab, hull, neighbours=5)
    print(encoding.shape)
    print(encoding[0, 0, 0, :])