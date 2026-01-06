import numpy as np
import time
from sklearn.neighbors import KDTree
import torch
from utils import *

def gaussian_kernel(px, q, sigma):
    """Gaussian kernel function."""
    return np.exp(-np.linalg.norm((px - q).astype(np.float64), axis=1) ** 2 / (2 * sigma ** 2))

def kernel_regression_NN(Pxy, values, qs, sigma):
    """Kernel Regression at query point q."""
    numerator = np.zeros(len(qs))
    denominator = np.zeros(len(qs))
    for idx, (px, py, q) in enumerate(zip(Pxy, values, qs)):
        kernel = gaussian_kernel(px, q, sigma)
        numerator[idx] = sum(kernel*py)
        denominator[idx] = sum(kernel) 
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator!=0)

def fastKR_NN(data, ori_data, sigma, k, xeval, yeval, size, leafsize=100):
    """fast kernel regression choosing the nearest neighbor(s)"""
    data = np.array(data)
    data_xy = data[:,0:2]
    data_f = data[:,2]
    query_points, X_query, Y_query = getQueryPoints(ori_data[:,0:2], xeval=xeval, yeval=yeval)
    tree = KDTree(data_xy, leaf_size=leafsize)
    radius = size * sigma
    n_query = len(query_points)
    KR = np.zeros(n_query)
    KR = []
    batch_size = 100
    print('Evaluation points:',len(query_points))

    start_time = time.time()
    for start in range(0, n_query, batch_size):
        # print('start batch')
        end = min(start + batch_size, n_query)
        batch_query_points = query_points[start:end]
        all_indices = tree.query_radius(batch_query_points, r=radius)
        # print('query neighbors')

        all_neighbors = [None] * len(batch_query_points)
        all_neighbors_f = [None] * len(batch_query_points)

        empty_mask = np.array([len(indices) == 0 for indices in all_indices])
        empty_indices = np.where(empty_mask)[0]

        if len(empty_indices) > 0:
            _, nn_indices = tree.query(batch_query_points[empty_indices], k=k)
            # Assign results for empty neighbors
            for idx, query_idx in enumerate(empty_indices):
                all_neighbors[query_idx] = np.array(data_xy[nn_indices[idx]])
                all_neighbors_f[query_idx] = np.array(data_f[nn_indices[idx]])
        
            # Assign results for non-empty neighbors
        for idx, indices in enumerate(all_indices):
            if not empty_mask[idx]:
                all_neighbors[idx] = np.array(data_xy[indices])
                all_neighbors_f[idx] = np.array(data_f[indices])
        
        if end % 50000 == 0:
            mid_time = time.time()
            print(end, 'continue computing KR values...')
            print('runtime:', mid_time - start_time)
            start_time = mid_time
        KR[start:end] = kernel_regression_NN(np.array(all_neighbors, dtype=object), np.array(all_neighbors_f, dtype=object), batch_query_points, sigma)

    KR = np.array(KR)
    print('KR shape:', KR.reshape(X_query.shape).shape)
    return KR.reshape(X_query.shape), X_query, Y_query, len(query_points)

def opt_gaussian_kernel(px, q, sigma):
    """Gaussian kernel function using PyTorch tensors."""
    return torch.exp(-torch.norm(px - q, dim=1) ** 2 / (2 * sigma ** 2))

def opt_kernel_regression_NN(Pxy, values, qs, sigma):
    """Compute the kernel regression at query points using selected neighbors."""
    numerator = []
    denominator = []

    for px, py, q in zip(Pxy, values, qs):
        kernel = opt_gaussian_kernel(px, q, sigma)
        numerator.append(torch.sum(kernel * py))
        denominator.append(torch.sum(kernel))

    numerator = torch.stack(numerator)
    denominator = torch.stack(denominator)
    return torch.where(denominator != 0, numerator / denominator, torch.zeros_like(numerator))

def opt_kernel_regression(data, n_query, query_points, X_queryShape, windowsize, sigma):
    """
    Compute the kernel regression of the coreset using KDTree for fast neighbor search.
    
    Uses KDTree (numpy, no gradients) to find neighbor indices, then selects from PyTorch tensors
    (with gradients) for kernel regression, maintaining autograd compatibility.
    """
    import time
    
    data_xyz = data[:, 0:2]  # (x, y)
    data_f = data[:, 2]      # Function values
    KR = []
    
    radius = sigma * windowsize
    batch_size = 128
    
    # Hybrid GPU/CPU strategy: For smaller datasets, compute distances on GPU but
    # do neighbor selection and kernel regression on CPU (faster due to less overhead)
    # For larger datasets (like clip1), keep everything on GPU (full GPU mode)
    use_hybrid = False
    if torch.cuda.is_available() and data_xyz.is_cuda:
        # Threshold: smaller datasets benefit from hybrid approach
        # Based on observation: small datasets have kernel regression as bottleneck on GPU,
        # but distance computation as bottleneck on CPU
        # clip1 is typically large (>500k queries), so it will use full GPU mode
        if n_query < 200_000:  # Smaller datasets (CESM, clip2)
            use_hybrid = True
            print(f"  Hybrid GPU/CPU mode: Distances on GPU, rest on CPU (small dataset: {n_query:,} queries)")
        else:
            print(f"  Full GPU mode: All computation on GPU (large dataset: {n_query:,} queries)")
    
    # KDTree-based neighbor search (faster than torch.cdist for neighbor finding)
    data_xyz_np = data_xyz.detach().cpu().numpy()
    tree = KDTree(data_xyz_np, leaf_size=100)
    print(f"  KDTree built for fast neighbor search (maintains autograd for kernel regression)")
    
    # Show batch size selection
    print(f"  Dataset size: {n_query:,} query points, {data_xyz.shape[0]:,} coreset points")
    print(f"  Selected batch size: {batch_size} (adaptive based on dataset size)")
    print(f"  Total batches: {(n_query + batch_size - 1) // batch_size:,}")
    print()
    
    # Timing diagnostics
    total_time = 0.0
    time_distances = 0.0
    time_neighbor_selection = 0.0
    time_kernel_regression = 0.0
    time_other = 0.0

    for start in range(0, n_query, batch_size):
        batch_start_time = time.time()
        is_first_batch = (start == 0)
        
        end = min(start + batch_size, n_query)
        batch_query_points = query_points[start:end]

        # Checkpoint 1: Neighbor search using KDTree
        t0 = time.time()
        
        # Use KDTree for fast neighbor search (numpy, no gradients)
        batch_query_points_np = batch_query_points.detach().cpu().numpy()
        all_indices = tree.query_radius(batch_query_points_np, r=radius)
        
        # Find queries with no neighbors
        empty_mask = np.array([len(indices) == 0 for indices in all_indices])
        empty_indices = np.where(empty_mask)[0]
        
        # For empty queries, find nearest neighbor
        nn_indices = None
        if len(empty_indices) > 0:
            _, nn_indices = tree.query(batch_query_points_np[empty_indices], k=1)
            # Ensure nn_indices is always at least 1D for indexing
            nn_indices = np.atleast_1d(nn_indices.squeeze())
        
        t_dist = time.time() - t0
        time_distances += t_dist
        if is_first_batch:
            print(f'  [Batch 1] Neighbor search (KDTree): {t_dist:.4f}s')
        
        # Checkpoint 2: Neighbor selection using indices from KDTree
        t0 = time.time()
        
        # For hybrid mode: move to CPU after neighbor search
        if use_hybrid:
            data_xyz_cpu = data_xyz.cpu()
            data_f_cpu = data_f.cpu()
            batch_query_points_cpu = batch_query_points.cpu()
        else:
            data_xyz_cpu = data_xyz
            data_f_cpu = data_f
            batch_query_points_cpu = batch_query_points
        
        all_neighbors = []
        all_neighbors_f = []
        
        # Use KDTree indices to select neighbors from PyTorch tensors (maintains gradients)
        empty_idx_map = {empty_indices[i]: i for i in range(len(empty_indices))} if len(empty_indices) > 0 else {}
        
        for i in range(batch_query_points_cpu.shape[0]):
            if empty_mask[i]:
                # No neighbors found, use nearest neighbor from KDTree query
                if len(empty_indices) > 0:
                    nearest_idx = int(nn_indices[empty_idx_map[i]])
                else:
                    # Fallback: find nearest using torch (shouldn't happen but safe)
                    distances = torch.cdist(data_xyz_cpu, batch_query_points_cpu[i:i+1])
                    nearest_idx = torch.argmin(distances[:, 0]).item()
                all_neighbors.append(data_xyz_cpu[nearest_idx].unsqueeze(0))
                all_neighbors_f.append(data_f_cpu[nearest_idx].unsqueeze(0))
                if start == 0 and i == 0:  # Only print once
                    print('no neighbors found.')
            else:
                # Use neighbors within radius from KDTree
                neighbor_indices = all_indices[i]
                all_neighbors.append(data_xyz_cpu[neighbor_indices])
                all_neighbors_f.append(data_f_cpu[neighbor_indices])
        
        t_neighbor = time.time() - t0
        time_neighbor_selection += t_neighbor
        if is_first_batch:
            print(f'  [Batch 1] Neighbor selection loop: {t_neighbor:.4f}s')
            if use_hybrid:
                print(f'  [Batch 1] Using CPU for neighbor selection and kernel regression (hybrid mode)')

        # Checkpoint 3: Kernel regression computation
        t0 = time.time()
        batch_KR = opt_kernel_regression_NN(all_neighbors, all_neighbors_f, batch_query_points_cpu, sigma)
        t_kr = time.time() - t0
        time_kernel_regression += t_kr
        if is_first_batch:
            print(f'  [Batch 1] Kernel regression computation: {t_kr:.4f}s')
        
        # Checkpoint 4: Append result
        t0 = time.time()
        KR.append(batch_KR)
        t_append = time.time() - t0
        time_other += t_append
        if is_first_batch:
            print(f'  [Batch 1] Append result: {t_append:.4f}s')
        
        # Clean up intermediate tensors to prevent memory accumulation
        # Note: batch_KR is already stored in KR list, safe to delete the reference
        del all_neighbors, all_neighbors_f, batch_KR
        if use_hybrid:
            del data_xyz_cpu, data_f_cpu, batch_query_points_cpu
        # Clear GPU cache periodically to prevent fragmentation
        if data_xyz.is_cuda and (end % (batch_size * 10) == 0):  # Every 10 batches
            torch.cuda.empty_cache()
        
        batch_time = time.time() - batch_start_time
        total_time += batch_time
        
        # Print first batch summary
        if is_first_batch:
            print(f'\n  [Batch 1] Total batch time: {batch_time:.4f}s')
            print(f'  [Batch 1] Breakdown:')
            print(f'    - Neighbor search (KDTree): {t_dist:.4f}s ({100*t_dist/batch_time:.1f}%)')
            print(f'    - Neighbor selection loop: {t_neighbor:.4f}s ({100*t_neighbor/batch_time:.1f}%)')
            print(f'    - Kernel regression: {t_kr:.4f}s ({100*t_kr/batch_time:.1f}%)')
            print(f'    - Append: {t_append:.4f}s ({100*t_append/batch_time:.1f}%)')
            print()
        
        # Progress reporting with timing
        if end % max(50000, batch_size * 10) == 0 or end == n_query:
            progress_pct = 100.0 * end / n_query
            batches_processed = (end + batch_size - 1) // batch_size
            avg_time_per_batch = total_time / batches_processed if batches_processed > 0 else batch_time
            remaining_batches = max(0, (n_query - end + batch_size - 1) // batch_size)
            est_remaining = remaining_batches * avg_time_per_batch
            print(f'  Progress: {end}/{n_query} ({progress_pct:.1f}%) | '
                  f'Batch time: {batch_time:.3f}s | '
                  f'Est. remaining: {est_remaining:.1f}s')
    

    # Final timing summary
    t0 = time.time()
    KR = torch.cat(KR)
    # Ensure output is on the same device as input data (for autograd compatibility)
    if use_hybrid and data_xyz.is_cuda:
        KR = KR.to(data_xyz.device)
    KR = KR.view(X_queryShape)
    time_other += time.time() - t0
    
    # Print timing diagnostics
    print(f'\n=== Runtime Diagnostics for opt_kernel_regression ===')
    print(f'Total time: {total_time:.3f}s')
    print(f'  Neighbor search (KDTree): {time_distances:.3f}s ({100*time_distances/total_time:.1f}%)')
    print(f'  Neighbor selection loop: {time_neighbor_selection:.3f}s ({100*time_neighbor_selection/total_time:.1f}%)')
    print(f'  Kernel regression computation: {time_kernel_regression:.3f}s ({100*time_kernel_regression/total_time:.1f}%)')
    print(f'  Other operations: {time_other:.3f}s ({100*time_other/total_time:.1f}%)')
    print(f'Average time per batch: {total_time/((n_query + batch_size - 1) // batch_size):.3f}s')
    print('=' * 55)
    
    return KR