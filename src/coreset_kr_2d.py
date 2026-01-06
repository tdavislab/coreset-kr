import numpy as np
import math
import os
import time
import sys
import vtk
from pyevtk.hl import imageToVTK
from vtk.util.numpy_support import vtk_to_numpy
import time
from sklearn.neighbors import KDTree
import pandas as pd
import torch
import random
import ast
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


def locally_optimize(coreset, query_points, X_queryShape, KR_P, k,  windowsize, sigma, learning_rate, num_iterations, data_path, factor, scalarfield):
    """Optimize both spatial locations and function values using PyTorch."""
    # Detect and set device (GPU if available, otherwise CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    if device.type == 'cuda':
        print(f'  GPU: {torch.cuda.get_device_name(0)}')
        print(f'  CUDA Version: {torch.version.cuda}')
        print(f'  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
    
    coreset = np.array(coreset)
    n_query = len(query_points)
    print('Evaluation points:',n_query)
    
    # Create tensors on the appropriate device
    coreset = torch.tensor(coreset, requires_grad=True, dtype=torch.float64, device=device)
    KR_P = torch.tensor(KR_P.T, dtype=torch.float64, device=device)
    query_points = torch.tensor(query_points, dtype=torch.float64, device=device)
    
    best_l2_err = torch.tensor(torch.inf, dtype=torch.float64, device=device)
    save_KR_S = None

    optimizer = torch.optim.Adam([coreset], lr=learning_rate)
    
    err_lst = []
    runtime_lst = []

    min_val = KR_P.min()
    max_val = KR_P.max()
    range_val = max_val - min_val
    KR_P_norm = (KR_P - min_val) / range_val


    for iteration in range(num_iterations):
        print('-------------')
        print(f"Iteration {iteration + 1}/{num_iterations}")

        start_time = time.time()

        # Use set_to_none=True for better memory efficiency
        optimizer.zero_grad(set_to_none=True)

        # Compute kernel regression and L2 error
        KR_S = opt_kernel_regression(coreset, n_query, query_points, X_queryShape, windowsize, sigma)
        print(KR_P.shape, KR_S.shape)
        l2_err = torch.sqrt(torch.sum((KR_P - KR_S) ** 2))

        end_time = time.time() - start_time
        print(f"Iteration {iteration} spends: {end_time} seconds")
        runtime_lst.append(end_time)

        print(f"L2 Error at iteration {iteration + 1}: {l2_err.item()}")
        # Move to CPU before converting to numpy
        KR_S_cpu = KR_S.detach().cpu()
        KR_P_norm, KR_S_norm = normalize(KR_P.detach().cpu().numpy(), KR_S_cpu.numpy())
        print('L_inf normalized error:', np.max(np.abs(KR_P_norm - KR_S_norm)))
        print('-------------')
        
        # Clean up CPU tensors immediately
        del KR_S_cpu, KR_P_norm, KR_S_norm

        # Backpropagation
        l2_err.backward()
        optimizer.step()
        
        # Store best result if needed
        if l2_err.item() < best_l2_err.item(): 
            # Delete old best result to free memory
            if save_KR_S is not None:
                del save_KR_S
            best_l2_err = l2_err.detach().clone()  # Store a detached copy
            save_KR_S = KR_S.detach().clone()  # Save the best kernel regression so far
        
        # Clean up intermediate tensors to prevent memory accumulation
        err_lst.append(l2_err.item())
        del l2_err, KR_S
        
        # Clear GPU cache periodically (every 5 iterations) to prevent fragmentation
        if device.type == 'cuda' and (iteration + 1) % 5 == 0:
            torch.cuda.empty_cache()
    
    # Move tensors back to CPU for numpy conversion
    coreset_cpu = coreset.detach().cpu()
    save_KR_S_cpu = save_KR_S.cpu() if save_KR_S is not None else None
    
    return list(coreset_cpu.numpy()), save_KR_S_cpu.numpy() if save_KR_S_cpu is not None else None, err_lst, runtime_lst

def getGridData(gridsize, data):
    x_coords = data[:, 0]
    y_coords = data[:, 1]
    
    # Find the range of the data
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    
    # Compute the grid cell indices for each point
    cell_indices = np.floor((data[:,0:2] - [x_min, y_min]) / gridsize).astype(int)
    
    # Group points into grid cells
    grid_cells = {}
    for datapoint, (i, j) in zip(data, cell_indices):
        if (i, j) not in grid_cells:
            grid_cells[(i, j)] = []
        grid_cells[(i, j)].append(datapoint)
        
    return grid_cells

def getGridData_reverse2(rest_coresetsize, coresetsize, org_gridsize, data, weights):
    n_grids_per_dim = int(np.sqrt(rest_coresetsize) * 1.2)  # 20% buffer

    x_coords = data[:, 0]
    y_coords = data[:, 1]
    
    # Compute data range
    min_vals = np.min(data[:, :2], axis=0)
    max_vals = np.max(data[:, :2], axis=0)
    
    # Compute adaptive grid size
    gridsize = (max_vals - min_vals) / n_grids_per_dim
    
    # Compute grid indices for each point (vectorized for efficiency)
    cell_indices = np.floor((data[:, :2] - min_vals) / gridsize).astype(int)

    # Convert grid cell indices to a single unique index for easy sorting
    grid_ids, grid_counts = np.unique(cell_indices, axis=0, return_counts=True)
    
    # Sort grid cells by density (descending order) and select top `rest_coresetsize`
    top_grid_indices = np.argsort(-grid_counts)[:rest_coresetsize]  # Negative for descending order
    selected_grid_ids = grid_ids[top_grid_indices]
    
    # Construct optimized grid mapping
    grid_cells = {tuple(grid_id): [] for grid_id in selected_grid_ids}
    grid_weights = {tuple(grid_id): [] for grid_id in selected_grid_ids}
    
    # Assign points to selected grid cells
    for i in range(len(data)):
        cell_id = tuple(cell_indices[i])
        if cell_id in grid_cells:
            grid_cells[cell_id].append(data[i])
            grid_weights[cell_id].append(weights[i])
        
    return grid_cells, grid_weights

def getGridData_reverse(rest_coresetsize, coresetsize, org_gridsize, data, weights):
    n_grids_per_dim = int(np.ceil(np.sqrt(rest_coresetsize)))

    x_coords = data[:, 0]
    y_coords = data[:, 1]
    
    # Find the range of the data
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    max_vals = np.array([x_max, y_max])
    min_vals = np.array([x_min, y_min])
    gridsize = (max_vals - min_vals) / n_grids_per_dim

    if coresetsize == rest_coresetsize:
        gridsize = org_gridsize
    
    print('Interal gridsize is', gridsize)
    
    # Compute the grid cell indices for each point
    cell_indices = np.floor((data[:,0:2] - min_vals) / gridsize).astype(int)
    
    # Group points into grid cells
    grid_cells = {}
    grid_weights = {}
    for datapoint, (i, j), weight in zip(data, cell_indices, weights):
        if (i, j) not in grid_cells:
            grid_cells[(i, j)] = []
        if (i, j) not in grid_weights:
            grid_weights[(i, j)] = []
        grid_cells[(i, j)].append(datapoint)
        grid_weights[(i, j)].append(weight)
        
    return grid_cells, grid_weights

def OptCoresetGrid(method, gridsize, data, gradient, criticalpoints, nx, ny, weight_critical):

    grid_cells = getGridData(gridsize, data)
    if method == 'randomSample':
        indices = np.random.choice(len(data),len(grid_cells), replace=False)
        coreset = data[indices]
        print('Actual coreset size:', len(coreset))
    if method == 'RS':
        coreset = [random.choice(grid) for grid in grid_cells.values()]
        print('Actual coreset size:', len(coreset))
    # Grid-Aggregate
    if method == 'GA':
        coreset = [np.mean(np.array(grid), axis=0) for grid in grid_cells.values()]
        print('Actual coreset size:', len(coreset))
        coresetsize = len(grid_cells)
        print('Expected coreset size:', coresetsize)
    # probabilistic critical points selection
    if method == 'Base':
        coresetsize = len(grid_cells)
        rest_coresetsize = coresetsize - len(criticalpoints)
        coreset_1 = list(criticalpoints[:,0:3]) #pre-select all critical points
        weights = np.ones(len(data)) # probability = 1 for all regular points
        indices = [(int(x * ny + y)) for x, y, _ in criticalpoints[:,0:3]]
        weight_critical = 0 #exclude selecting the critical points
        weights[indices] *= weight_critical
        grid_cells, grid_weights = getGridData_reverse(rest_coresetsize, coresetsize, gridsize, data, weights)
        coreset_2 = [np.mean(np.array(grid), axis=0) for grid in grid_cells.values()]
        coreset = coreset_1 + coreset_2
        print('Actual coreset size:', len(coreset))
        print('Expected coreset size:', coresetsize)
    
    return coreset, len(coreset)

def locally_optCoreset(gridsize, data, KR_P, criticalpoints, nx, ny, sigma, knn, windowsize, num_iterations, data_path, factor, scalarfield, lr):
    grid_cells = getGridData(gridsize, data)
    coresetsize = len(grid_cells)
    rest_coresetsize = coresetsize - len(criticalpoints)
    if rest_coresetsize > 0:
        coreset_2 = [np.mean(np.array(grid), axis=0) for grid in grid_cells.values()]
        learning_rate = lr
        data = np.array(data)
        query_points, X_query, Y_query = getQueryPoints(data[:,0:2], xeval=math.floor(nx/factor), yeval=math.floor(ny/factor))
        optCoreset_2, KR_S, err_lst, runtime_lst = locally_optimize(coreset_2, query_points, X_query.shape, KR_P, int(knn), int(windowsize), sigma, learning_rate, num_iterations, data_path, factor, scalarfield)
        coreset = optCoreset_2
        # coreset = coreset_1 + optCoreset_2
        
    print('Actual coreset size:', len(coreset))
    print('Expected coreset size:', coresetsize)
    return coreset, len(coreset), KR_S.T, err_lst, runtime_lst

def main():
    dataName = sys.argv[1]
    data_path = 'data/'+dataName+'/'
    gridsize = 20
    factor = 8 # mesh xy eval: 250 375 Evaluation points: 93750
    knn = 1
    sigma = int(sys.argv[2])
    windowsize = 5 # a paramter to determine the window size that are queired to calcluate the kr values
    lr = float(sys.argv[3])
    num_iterations = 30
    run_opt = ast.literal_eval(sys.argv[4])
    run_random = ast.literal_eval(sys.argv[5])
    run_randomSample = ast.literal_eval(sys.argv[6])
    scalarfield = sys.argv[7]
    savepath = 'data/'+dataName+'/output_figs/' 
    print('Profile: ', dataName, data_path, 'grid size:'+str(gridsize), 'factor:'+str(factor), 'knn:'+str(knn), 'sigma:'+str(sigma), 'window size:'+str(windowsize), scalarfield, savepath)

    data_file = data_path+dataName+'.vti'
    rawData = readVTKFile(data_file).T

    print(rawData.shape)
    nx, ny = rawData.shape
    print('Total', nx*ny, 'points')
    data, gradient = getData(rawData)
    data = np.array(data)
    print(data[:10])

    KR_P = data_path+'VTKfiles/KR_P/Sigma_'+str(sigma)+'/KR_P_'+str(factor)+'_'+str(sigma)+'.vti'
    if not os.path.exists(KR_P):
        make_dir(data_path+'VTKfiles/KR_P/')
        make_dir(data_path+'VTKfiles/KR_P/Sigma_'+str(sigma))
        KR_P, X_query_P, Y_query_P, numEvalPts = fastKR_NN(data, data, float(sigma), int(knn), xeval=math.floor(nx/factor), yeval=math.floor(ny/factor), size=int(windowsize), leafsize=100)
        Numpy2VTK(KR_P, data_path+'VTKfiles/KR_P/Sigma_'+str(sigma)+'/KR_P_'+str(factor)+'_'+str(sigma), scalarfield)
    else:
        KR_P = readVTKFile(KR_P)
    print('KR_P shape', KR_P.shape)

    criticalpoints = []

    if run_randomSample:
        method = 'randomSample'
        make_dir(data_path+'VTKfiles/KR_S_randomSample/')
        make_dir(data_path+'VTKfiles/KR_S_randomSample/Sigma_'+str(sigma))

        for i in range(5):
            print('---------------')
            coreset, coresetsize = OptCoresetGrid(method, int(gridsize), data, gradient, criticalpoints, nx, ny, 0)
            
            KR_S, X_query, Y_query, numEvalPts = fastKR_NN(coreset, data, float(sigma), int(knn), xeval=math.floor(nx/factor), yeval=math.floor(ny/factor), size=int(windowsize), leafsize=100)

            KR_S = KR_S.T
            Numpy2VTK(KR_S, data_path+'VTKfiles/KR_S_randomSample/Sigma_'+str(sigma)+'/KR_S_randomSample'+str(factor)+'_'+str(sigma)+'_i'+str(i), scalarfield)
            print('KR_S_randomSample data saved!')

            # error = np.abs(KR_P - KR_S)
            # print('L infinity:', np.max(error))
            KR_P_norm, KR_S_norm= normalize(KR_P, KR_S)
            print('normalized L infinity:', np.max(np.abs(KR_P_norm - KR_S_norm)))
            print('---------------')

    if run_random:
        method = 'RS'
        make_dir(data_path+'VTKfiles/KR_S_rand/')
        make_dir(data_path+'VTKfiles/KR_S_rand/Sigma_'+str(sigma))

        for i in range(5):
            print('---------------')
            coreset, coresetsize = OptCoresetGrid(method, int(gridsize), data, gradient, criticalpoints, nx, ny, 0)
            
            KR_S, X_query, Y_query, numEvalPts = fastKR_NN(coreset, data, int(sigma), int(knn), xeval=math.floor(nx/factor), yeval=math.floor(ny/factor), size=int(windowsize), leafsize=100)

            KR_S = KR_S.T
            Numpy2VTK(KR_S, data_path+'VTKfiles/KR_S_rand/Sigma_'+str(sigma)+'/KR_S_rand_'+str(factor)+'_'+str(sigma)+'_i'+str(i), scalarfield)
            print('KR_S_rand data saved!')

            # error = np.abs(KR_P - KR_S)
            # print('L infinity:', np.max(error))
            KR_P_norm, KR_S_norm= normalize(KR_P, KR_S)
            print('normalized L infinity:', np.max(np.abs(KR_P_norm - KR_S_norm)))
            print('---------------')

    if run_opt:
        method = 'GA'
        make_dir(data_path+'VTKfiles/KR_S_opt/')
        make_dir(data_path+'VTKfiles/KR_S/')
        make_dir(data_path+'VTKfiles/KR_S_opt/Sigma_'+str(sigma))
        make_dir(data_path+'VTKfiles/KR_S/Sigma_'+str(sigma))

        coreset, coresetsize = OptCoresetGrid(method, int(gridsize), data, gradient, criticalpoints, nx, ny, 0)
        print('Coreset size:', coresetsize)

        coreset_opt, coresetsize, KR_S_opt, err_lst, runtime_lst = locally_optCoreset(int(gridsize), data, KR_P, criticalpoints, nx, ny, sigma, knn, windowsize, num_iterations, data_path, factor, scalarfield, lr)
       
        Numpy2VTK(KR_S_opt, data_path+'VTKfiles/KR_S_opt/Sigma_'+str(sigma)+'/KR_S_'+str(factor)+'_'+str(sigma)+'_opt_w'+str(windowsize), scalarfield)
        print('KR_S_opt data saved!')

    else:
        print('No optimization run. Exit.')
        

if __name__ == '__main__':

    main()