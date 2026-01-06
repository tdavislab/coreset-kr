import numpy as np
import math
import matplotlib.pyplot as plt
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
import torch.optim.lr_scheduler as lr_scheduler

import ast

def normalize(KR_P, KR_S):
    min_val_P = np.min(KR_P)
    max_val_P = np.max(KR_P)

    # Normalize KR_P between [0,1]
    KR_P_norm = (KR_P - min_val_P) / (max_val_P - min_val_P)

    # Normalize KR_S using KR_P's min and max
    KR_S_norm = (KR_S - min_val_P) / (max_val_P - min_val_P) 

    return KR_P_norm, KR_S_norm

def make_dir(new_dir):
    if not os.path.exists(new_dir):
        print('Make new dir', new_dir)
        os.makedirs(new_dir)

def readVTKFile(filename, scalarfield, load_KR_P):
    """Read a .vti (VTK ImageData) file and convert it to a 3D numpy array."""
    if load_KR_P:
        reader = vtk.vtkStructuredPointsReader()
    else:
        reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(filename)
    reader.Update()  # Read the .vti file
    image_data = reader.GetOutput()

    dims = image_data.GetDimensions()  # Get the dimensions (Nx, Ny, Nz)

    vtk_array = image_data.GetPointData().GetScalars(scalarfield)
    numpy_array = vtk_to_numpy(vtk_array)

    # Reshape to match VTK dimensions (Z, Y, X)
    numpy_array_reshaped = numpy_array.reshape(dims[2], dims[1], dims[0])

    return numpy_array_reshaped

def Numpy2VTK(data, newName, scalarfield):
    """Convert a 3D numpy array to VTK format and save it."""
    imageToVTK(newName, pointData={scalarfield: data})

def getData(data):
    """Extract 3D structured data and compute gradients in 3D."""
    dz, dy, dx = np.gradient(data)  # Compute gradients in x, y, z directions
    # gradient_magnitude = np.sqrt(dx**2 + dy**2 + dz**2)
    # gradient_flat = gradient_magnitude.ravel()

    x_indices, y_indices, z_indices = np.meshgrid(
        np.arange(data.shape[0]),
        np.arange(data.shape[1]),
        np.arange(data.shape[2]),
        indexing="ij"
    )

    x_indices = x_indices.ravel()
    y_indices = y_indices.ravel()
    z_indices = z_indices.ravel()
    data_flat = data.ravel()

    # Combine data into a structured format (x, y, z, scalar_value, gradient)
    newData = np.column_stack((x_indices, y_indices, z_indices, data_flat))
    return newData, []

def getQueryPoints(data, xeval=100, yeval=100, zeval=100):
    """Generate a uniform 3D grid of query points for interpolation."""
    x_min, x_max = data[:, 0].min(), data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min(), data[:, 1].max() + 1
    z_min, z_max = data[:, 2].min(), data[:, 2].max() + 1

    x_query = np.linspace(x_min, x_max, xeval, dtype=int)
    y_query = np.linspace(y_min, y_max, yeval, dtype=int)
    z_query = np.linspace(z_min, z_max, zeval, dtype=int)

    X_query, Y_query, Z_query = np.meshgrid(x_query, y_query, z_query, indexing="ij")
    
    print(f'Mesh grid eval: {xeval} x {yeval} x {zeval}')
    
    query_points = np.c_[X_query.ravel(), Y_query.ravel(), Z_query.ravel()]
    return query_points, X_query, Y_query, Z_query

def read_criticalpoints(filename, factor, data, nx, ny, nz):
    """Read critical points from CSV and scale to match 3D data."""
    cp_df = pd.read_csv(filename)
    cp_df.reset_index(inplace=True)

    # Assuming critical points are stored as (z, y, x) order
    criticalpoints = cp_df[['Points_2', 'Points_1', 'Points_0']].to_numpy()
    # criticalpoints[:, :3] *= factor  # Rescale to match original data grid
    criticalpoints[:, :3] = (criticalpoints[:, :3].astype(float) * factor).astype(int)
    indices = [(int(x * ny * nz + y * nz + z)) for x, y, z in criticalpoints]
    scalars = data[indices][:, 3]  # Extract function values

    criticalpoints = np.column_stack((criticalpoints, scalars))
    return criticalpoints

def gaussian_kernel(px, q, sigma):
    """Gaussian kernel function for 3D data (x,y,z)."""
    return np.exp(-np.linalg.norm((px - q).astype(np.float64), axis=1) ** 2 / (2 * sigma ** 2))

def kernel_regression_NN(Pxyz, values, qs, sigma):
    """Kernel Regression at query points for 3D data."""
    numerator = np.zeros(len(qs))
    denominator = np.zeros(len(qs))
    for idx, (px, py, q) in enumerate(zip(Pxyz, values, qs)):
        kernel = gaussian_kernel(px, q, sigma)
        numerator[idx] = sum(kernel * py)
        denominator[idx] = sum(kernel) 
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator!=0)

def fastKR_NN(data, ori_data, sigma, k, xeval, yeval, zeval, size, compute_KR_S, leafsize=100):
    """Fast kernel regression choosing the nearest neighbor(s) for 3D data."""
    data = np.array(data)
    data_xyz = data[:, 0:3]  # Now using x, y, z
    data_f = data[:, 3]  # Function values remain in column 3
    
    # Create query points in 3D space
    query_points, X_query, Y_query, Z_query = getQueryPoints(ori_data[:,0:3], xeval=xeval, yeval=yeval, zeval=zeval)
    print(query_points)
    
    if compute_KR_S:
        batch_size = 128
        n_query = len(query_points)
        data = torch.tensor(data, dtype=torch.float32)
        query_points = torch.tensor(query_points, dtype=torch.float32)
        data_xyz = data[:, 0:3]  # Now using x, y, z
        data_f = data[:, 3] 
        KR = []
        for start in range(0, n_query, batch_size):
            end = min(start + batch_size, n_query)
            batch_query_points = query_points[start:end]

            # Compute pairwise distances between coreset points and query points
            distances = torch.cdist(data_xyz.unsqueeze(0), batch_query_points.unsqueeze(0)).squeeze(0)

            # Identify neighbors within the radius
            radius = sigma * size
            neighbor_masks = distances < radius

            # Prepare lists to hold neighbors for each query point in the batch
            all_neighbors = []
            all_neighbors_f = []

            for i in range(batch_query_points.shape[0]):
                neighbor_indices = torch.where(neighbor_masks[:, i])[0]

                if len(neighbor_indices) == 0:
                    # No neighbors found, select the nearest 1 neighbor
                    nearest_idx = torch.topk(distances[:, i], k, largest=False).indices[0]
                    all_neighbors.append(data_xyz[nearest_idx].unsqueeze(0))
                    all_neighbors_f.append(data_f[nearest_idx].unsqueeze(0))
                else:
                    # Use neighbors within the window
                    all_neighbors.append(data_xyz[neighbor_indices])
                    all_neighbors_f.append(data_f[neighbor_indices])
                
            if end % 5000 == 0:
                print(end, 'continue computing KR values...')

            # Compute the kernel regression for the current mini-batch
            batch_KR = opt_kernel_regression_NN(all_neighbors, all_neighbors_f, batch_query_points, sigma)
            
            KR.append(batch_KR)
        KR = torch.cat(KR)
        return KR.numpy().reshape(X_query.shape), X_query, Y_query, Z_query, len(query_points)

    tree = KDTree(data_xyz, leaf_size=leafsize)
    radius = size * sigma
    n_query = len(query_points)
    KR = np.zeros(n_query)
    batch_size = 2000
    print('Evaluation points:', len(query_points))

    for start in range(0, n_query, batch_size):
        end = min(start + batch_size, n_query)
        batch_query_points = query_points[start:end]

        # Query neighbors within a radius
        all_indices = tree.query_radius(batch_query_points, r=radius)

        all_neighbors = [None] * len(batch_query_points)
        all_neighbors_f = [None] * len(batch_query_points)

        empty_mask = np.array([len(indices) == 0 for indices in all_indices])
        empty_indices = np.where(empty_mask)[0]

        if len(empty_indices) > 0:
            _, nn_indices = tree.query(batch_query_points[empty_indices], k=k)
            for idx, query_idx in enumerate(empty_indices):
                all_neighbors[query_idx] = np.array(data_xyz[nn_indices[idx]])
                all_neighbors_f[query_idx] = np.array(data_f[nn_indices[idx]])

        for idx, indices in enumerate(all_indices):
            if not empty_mask[idx]:
                all_neighbors[idx] = np.array(data_xyz[indices])
                all_neighbors_f[idx] = np.array(data_f[indices])

        if end % 5000 == 0:
            print(end, 'continue computing KR values...')

        KR[start:end] = kernel_regression_NN(np.array(all_neighbors, dtype=object), np.array(all_neighbors_f, dtype=object), batch_query_points, sigma)

    KR = np.array(KR)
    print('KR shape:', KR.reshape(X_query.shape).shape)
    return KR.reshape(X_query.shape), X_query, Y_query, Z_query, len(query_points)

def opt_gaussian_kernel(px, q, sigma):
    """Gaussian kernel function using PyTorch tensors for 3D data."""
    return torch.exp(-torch.norm(px - q, dim=1) ** 2 / (2 * sigma ** 2))

def opt_kernel_regression_NN(Pxyz, values, qs, sigma):
    """Compute the kernel regression at query points using selected neighbors."""
    numerator = []
    denominator = []

    for px, py, q in zip(Pxyz, values, qs):
        kernel = opt_gaussian_kernel(px, q, sigma)
        numerator.append(torch.sum(kernel * py))
        denominator.append(torch.sum(kernel))

    numerator = torch.stack(numerator)
    denominator = torch.stack(denominator)
    return torch.where(denominator != 0, numerator / denominator, torch.zeros_like(numerator))

def opt_kernel_regression(data, n_query, query_points, X_queryShape, windowsize, sigma, k):
    """
    Compute the kernel regression of the coreset using KDTree for fast neighbor search (3D data).
    
    Uses KDTree (numpy, no gradients) to find neighbor indices, then selects from PyTorch tensors
    (with gradients) for kernel regression, maintaining autograd compatibility.
    """
    import time
    
    data_xyz = data[:, 0:3]  # (x, y, z) - 3D coordinates
    data_f = data[:, 3]      # Function values
    KR = []
    
    radius = sigma * windowsize
    batch_size = 32
    
    # Hybrid GPU/CPU strategy: For smaller datasets, do neighbor selection and kernel regression 
    # on CPU (faster due to less overhead). For larger datasets, keep everything on GPU.
    use_hybrid = True
    if torch.cuda.is_available() and data_xyz.is_cuda:
        if n_query < 200_000:  # Smaller datasets
            use_hybrid = True
            print(f"  Hybrid GPU/CPU mode: KDTree on CPU, kernel regression on CPU (small dataset: {n_query:,} queries)")
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

def locally_optimize(coreset, query_points, X_queryShape, KR_P, k, windowsize, sigma, learning_rate, num_iterations, data_path, factor, scalarfield):
    """Optimize both spatial locations and function values using PyTorch for 3D data."""
    # Detect and set device (GPU if available, otherwise CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    if device.type == 'cuda':
        print(f'  GPU: {torch.cuda.get_device_name(0)}')
        print(f'  CUDA Version: {torch.version.cuda}')
        print(f'  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
    
    coreset = np.array(coreset)
    n_query = len(query_points)
    print('Evaluation points:', n_query)
    
    # Create tensors on the appropriate device
    coreset = torch.tensor(coreset, requires_grad=True, dtype=torch.float64, device=device)
    KR_P = torch.tensor(KR_P.T, dtype=torch.float64, device=device)
    query_points = torch.tensor(query_points, dtype=torch.float64, device=device)
    
    best_l2_err = torch.tensor(torch.inf, dtype=torch.float64, device=device)
    save_KR_S = None

    optimizer = torch.optim.Adam([coreset], lr=learning_rate)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    min_val = KR_P.min()
    max_val = KR_P.max()
    range_val = max_val - min_val  # Avoid division by zero

    # Normalize KR_P and KR_S using KR_P's stats
    KR_P_norm = (KR_P - min_val) / range_val
    
    err_lst = []
    runtime_lst = []
    
    best_lf_err = 999999
    for iteration in range(num_iterations):
        print('-------------')
        print(f"Iteration {iteration + 1}/{num_iterations}")

        start_time = time.time()

        # Use set_to_none=True for better memory efficiency
        optimizer.zero_grad(set_to_none=True)

        # Compute kernel regression and L2 error
        KR_S = opt_kernel_regression(coreset, n_query, query_points, X_queryShape, windowsize, sigma, k)
        l2_err = torch.sqrt(torch.sum((KR_P_norm - KR_S) ** 2))

        end_time = time.time() - start_time
        print(f"Iteration {iteration} spends: {end_time} seconds")
        runtime_lst.append(end_time)

        print(f"L2 Error at iteration {iteration + 1}: {l2_err.item()}")
        # Move to CPU before converting to numpy
        KR_P_cpu = KR_P.detach().cpu()
        KR_S_cpu = KR_S.detach().cpu()
        KR_P_norm_cpu, KR_S_norm_cpu = normalize_KR_P_KR_S(KR_P_cpu.numpy(), KR_S_cpu.numpy())
        print('L_inf error:', np.max(np.abs(KR_P_norm_cpu - KR_S_norm_cpu)))
        print('-------------')
        
        # Clean up CPU tensors immediately
        del KR_P_cpu, KR_S_cpu, KR_P_norm_cpu, KR_S_norm_cpu

        # Backpropagation
        l2_err.backward()
        optimizer.step()
        scheduler.step()

        # Store best result if needed
        if l2_err.item() < best_l2_err.item(): 
            # Delete old best result to free memory
            if save_KR_S is not None:
                del save_KR_S
            best_l2_err = l2_err.detach().clone()  # Store a detached copy
            save_KR_S = KR_S.detach().clone()  # Save the best kernel regression so far
        
        error = torch.max(torch.abs(KR_P - KR_S))
        normalizer = torch.max(torch.abs(KR_P))
        normalized_error = (error / normalizer).item()
        print('Normalized L_inf error:', normalized_error)
        
        # Clean up intermediate tensors to prevent memory accumulation
        err_lst.append(l2_err.item())
        del l2_err, KR_S
        
        # Clear GPU cache periodically (every 5 iterations) to prevent fragmentation
        if device.type == 'cuda' and (iteration + 1) % 5 == 0:
            torch.cuda.empty_cache()
        
        # if iteration + 1 == 4:
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] *= 1
        #         print(f"Iteration {iteration+1}, Adjusted Learning Rate: {optimizer.param_groups[0]['lr']}")
        # if iteration + 1 in [10, 20, 30]:
        #     Numpy2VTK(save_KR_S.numpy(), data_path+'VTKfiles/KR_S_opt/Sigma_'+str(sigma)+'/KR_S_'+str(factor)+'_'+str(sigma)+'_opt_w'+str(windowsize)+'_iter'+str(iteration+1), scalarfield)
        #     print('KR_S_opt data saved!')
    
    # Move tensors back to CPU for numpy conversion
    coreset_cpu = coreset.detach().cpu()
    save_KR_S_cpu = save_KR_S.cpu() if save_KR_S is not None else None
    
    return list(coreset_cpu.numpy()), save_KR_S_cpu.numpy() if save_KR_S_cpu is not None else None, err_lst, runtime_lst

def getGridData(gridsize, data):
    """Groups 3D data points into grid cells."""
    x_coords, y_coords, z_coords = data[:, 0], data[:, 1], data[:, 2]
    
    # Find the range of the data
    min_vals = np.min(data[:, 0:3], axis=0)
    max_vals = np.max(data[:, 0:3], axis=0)

    # Compute the grid cell indices for each point
    cell_indices = np.floor((data[:, 0:3] - min_vals) / gridsize).astype(int)
    
    # Group points into grid cells
    grid_cells = {}
    for datapoint, (i, j, k) in zip(data, cell_indices):
        if (i, j, k) not in grid_cells:
            grid_cells[(i, j, k)] = []
        grid_cells[(i, j, k)].append(datapoint)
        
    return grid_cells

def getGridData_reverse(rest_coresetsize, coresetsize, org_gridsize, data, weights):
    """Groups 3D data points into grid cells and selects the densest regions."""
    n_grids_per_dim = int(np.cbrt(rest_coresetsize) * 1.2)  # 20% buffer

    # Compute data range
    min_vals = np.min(data[:, :3], axis=0)
    max_vals = np.max(data[:, :3], axis=0)

    # Compute adaptive grid size
    gridsize = (max_vals - min_vals) / n_grids_per_dim

    print('Interal gridsize is', gridsize)
    
    # Compute grid indices for each point
    cell_indices = np.floor((data[:, :3] - min_vals) / gridsize).astype(int)

    # Convert grid cell indices to a single unique index for sorting
    grid_ids, grid_counts = np.unique(cell_indices, axis=0, return_counts=True)

    # Sort grid cells by density (descending order) and select top `rest_coresetsize`
    top_grid_indices = np.argsort(-grid_counts)[:rest_coresetsize]
    selected_grid_ids = grid_ids[top_grid_indices]
    
    # Construct optimized grid mapping
    grid_cells = {tuple(grid_id): [] for grid_id in selected_grid_ids}
    
    # Assign points to selected grid cells
    for i in range(len(data)):
        cell_id = tuple(cell_indices[i])
        if cell_id in grid_cells:
            grid_cells[cell_id].append(data[i])
        
    return grid_cells

def OptCoresetGrid(method, gridsize, data, gradient, criticalpoints, nx, ny, nz, weight_critical):
    """Optimized Coreset Selection Using Grid-Based Methods for 3D Data."""
    
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
    # Probabilistic critical points selection
    elif method == 'Base':
        coresetsize = len(grid_cells)
        rest_coresetsize = coresetsize - len(criticalpoints)
        coreset_1 = list(criticalpoints[:, 0:4])  # Pre-select all critical points
        weights = np.ones(len(data))  # Probability = 1 for all regular points
        indices = [(int(x * ny * nz + y * nz + z)) for x, y, z, _ in criticalpoints[:, 0:4]]
        weight_critical = 0  # Exclude selecting the critical points
        weights[indices] *= weight_critical
        grid_cells = getGridData_reverse(rest_coresetsize, coresetsize, gridsize, data, weights)
        coreset_2 = [np.mean(np.array(grid), axis=0) for grid in grid_cells.values()]
        coreset = coreset_1 + coreset_2
        print('Actual coreset size:', len(coreset))
        print('Expected coreset size:', coresetsize)
    
    return coreset, len(coreset)

def locally_optCoreset(gridsize, data, KR_P, criticalpoints, nx, ny, nz, sigma, knn, windowsize, num_iterations, data_path, factor, scalarfield, lr):
    """Optimizes Coreset Selection for Kernel Regression in 3D."""
    
    grid_cells = getGridData(gridsize, data)
    coresetsize = len(grid_cells)
    rest_coresetsize = coresetsize - len(criticalpoints)
    
    if rest_coresetsize > 0:
        coreset_2 = [np.mean(np.array(grid), axis=0) for grid in grid_cells.values()]
        learning_rate = lr
        data = np.array(data)
        query_points, X_query, Y_query, Z_query = getQueryPoints(data[:,0:3], xeval=math.floor(nx/factor), yeval=math.floor(ny/factor), zeval=math.floor(nz/factor))
        # print(query_points[:10])
        optCoreset_2, KR_S, err_lst, runtime_lst = locally_optimize(coreset_2, query_points, X_query.shape, KR_P, int(knn), int(windowsize), sigma, learning_rate, num_iterations, data_path, factor, scalarfield)
        coreset = optCoreset_2

    print('Actual coreset size:', len(coreset))
    print('Expected coreset size:', coresetsize)
    return coreset, len(coreset), KR_S.T, err_lst, runtime_lst

def normalize_KR_P_KR_S(KR_P, KR_S):
    min_val_P = np.min(KR_P)
    max_val_P = np.max(KR_P)

    # Normalize KR_P between [0,1]
    KR_P_norm = (KR_P - min_val_P) / (max_val_P - min_val_P)

    # Normalize KR_S using KR_P's min and max
    KR_S_norm = (KR_S - min_val_P) / (max_val_P - min_val_P) 

    return KR_P_norm, KR_S_norm

def main():
    # profile
    dataName = sys.argv[1]
    data_path = 'data/'+dataName+'/'
    method = 'GA' 
    gridsize = 4
    factor = 2 # mesh xy eval: 250 375 Evaluation points: 93750
    knn = 1
    sigma = int(sys.argv[2])
    windowsize = 5 # a paramter to determine the window size that are queired to calcluate the kr values
    lr = float(sys.argv[3])
    num_iterations = 30
    compute_KR_S = ast.literal_eval(sys.argv[4])
    run_random = ast.literal_eval(sys.argv[5])
    randomSample = ast.literal_eval(sys.argv[6])
    scalarfield = sys.argv[7]
    savepath = 'data/'+dataName+'/output_figs/' 
    print('Profile: ', dataName, data_path, method, 'grid size:'+str(gridsize), 'factor:'+str(factor), 'knn:'+str(knn), 'sigma:'+str(sigma), 'window size:'+str(windowsize), scalarfield, savepath)

    data_file = data_path+dataName+'.vtk'
    rawData = readVTKFile(data_file, scalarfield, load_KR_P=True).T

    print(rawData.shape)
    nx, ny, nz = rawData.shape
    print('Total', nx*ny*nz, 'points')
    data, gradient = getData(rawData)
    data = np.array(data)
    print(data[:10])

    KR_P = data_path+'VTKfiles/KR_P/Sigma_'+str(sigma)+'/KR_P_'+str(factor)+'_'+str(sigma)+'.vti'
    if not os.path.exists(KR_P):
        make_dir(data_path+'VTKfiles/KR_P/')
        make_dir(data_path+'VTKfiles/KR_P/Sigma_'+str(sigma))
        KR_P, X_query_P, Y_query_P, Z_query_P, numEvalPts = fastKR_NN(data, data, float(sigma), int(knn), xeval=math.floor(nx/factor), yeval=math.floor(ny/factor), zeval=math.floor(nz/factor), size=int(windowsize), compute_KR_S=compute_KR_S, leafsize=100)
        Numpy2VTK(KR_P, data_path+'VTKfiles/KR_P/Sigma_'+str(sigma)+'/KR_P_'+str(factor)+'_'+str(sigma), scalarfield)
    else:
        KR_P = readVTKFile(KR_P, scalarfield, load_KR_P=False)
    print('KR_P shape', KR_P.shape)

    criticalpoints = []

    if randomSample:
        method = 'randomSample'
        make_dir(data_path+'VTKfiles/KR_S_randomSample/Sigma_'+str(sigma))

        for i in range(5):
            print('---------------')
            coreset, coresetsize = OptCoresetGrid(method, int(gridsize), data, gradient, criticalpoints, nx, ny, nz, 0)

            KR_S, X_query, Y_query, Z_query, numEvalPts = fastKR_NN(coreset, data, float(sigma), int(knn), xeval=math.floor(nx/factor), yeval=math.floor(ny/factor), zeval=math.floor(nz/factor), size=int(windowsize), compute_KR_S=compute_KR_S, leafsize=100)
            KR_S = KR_S.T
            Numpy2VTK(KR_S, data_path+'VTKfiles/KR_S_randomSample/Sigma_'+str(sigma)+'/KR_S_randomSample'+str(factor)+'_'+str(sigma)+'_i'+str(i), scalarfield)
            print('KR_S_randomSample data saved!')

            KR_P_norm, KR_S_norm = normalize(KR_P, KR_S)
            print('normalized L infinity:', np.max(np.abs(KR_P_norm - KR_S_norm)))
            print('---------------')
        

    if run_random:
        method = 'RS'
        make_dir(data_path+'VTKfiles/KR_S_rand/Sigma_'+str(sigma))

        for i in range(5):
            print('---------------')
            coreset, coresetsize = OptCoresetGrid(method, int(gridsize), data, gradient, criticalpoints, nx, ny, nz, 0)

            KR_S, X_query, Y_query, Z_query, numEvalPts = fastKR_NN(coreset, data, float(sigma), int(knn), xeval=math.floor(nx/factor), yeval=math.floor(ny/factor), zeval=math.floor(nz/factor), size=int(windowsize), compute_KR_S=compute_KR_S, leafsize=100)
            KR_S = KR_S.T
            Numpy2VTK(KR_S, data_path+'VTKfiles/KR_S_rand/Sigma_'+str(sigma)+'/KR_S_rand_'+str(factor)+'_'+str(sigma)+'_i'+str(i), scalarfield)
            print('KR_S_rand data saved!')

            KR_P_norm, KR_S_norm = normalize(KR_P, KR_S)
            print('normalized L infinity:', np.max(np.abs(KR_P_norm - KR_S_norm)))
            print('---------------')

    if compute_KR_S:
        method = 'GA'
        make_dir(data_path+'VTKfiles/KR_S_opt/')
        make_dir(data_path+'VTKfiles/KR_S/')
        make_dir(data_path+'VTKfiles/KR_S_opt/Sigma_'+str(sigma))
        make_dir(data_path+'VTKfiles/KR_S/Sigma_'+str(sigma))

        KR_P = data_path+'VTKfiles/KR_P/Sigma_'+str(sigma)+'/KR_P_'+str(factor)+'_'+str(sigma)+'.vti'
        KR_P = readVTKFile(KR_P, scalarfield, load_KR_P=False)
        print('KR_P shape', KR_P.shape)

        coreset, coresetsize = OptCoresetGrid(method, int(gridsize), data, gradient, criticalpoints, nx, ny, nz, 0)
       
        coreset_opt, coresetsize, KR_S_opt, err_lst, runtime_lst = locally_optCoreset(int(gridsize), data, KR_P, criticalpoints, nx, ny, nz, sigma, knn, windowsize, num_iterations, data_path, factor, scalarfield, lr)

        Numpy2VTK(KR_S_opt, data_path+'VTKfiles/KR_S_opt/Sigma_'+str(sigma)+'/KR_S_'+str(factor)+'_'+str(sigma)+'_opt_w'+str(windowsize), scalarfield)
        print('KR_S_opt data saved!')

    else:
        print('No optimization run. Exit.')

if __name__ == '__main__':

    main()