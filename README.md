# Coreset-KR

This repository contains the implementation of "A Topology-Preserving Coreset for Kernel Regression in Scientific Visualization", accepted to PacificVis 2026 TVCG Journal track.

`coreset_kr_2d.py` implements coreset for kernel regression based on gradient descent optimization for 2D scalar field data. The script constructs and optimizes coresets (compressed representations of large datasets) that preserve the kernel regression behavior of the original data. 

`coreset_kr_3d.py` implements coreset for kernel regression based on gradient descent optimization for 3D scalar field data. 

## 

## USAGE

The dataset GTOPO-ME is used here as an example. You need to replace it with your dataset.

### Create data path

Run the following commands. Need to run this only once.
```
cd data/GTOPO-ME/
mkdir VTKfiles
```

### Python Environment

Create a virtual environment and install required dependencies:

```bash
# Install required packages
pip install numpy torch vtk pyevtk scikit-learn pandas
```

**Required packages:**
- `numpy`: Numerical computations
- `torch`: PyTorch for GPU acceleration and optimization
- `vtk`: VTK for reading/writing visualization data
- `pyevtk`: VTK file writing utilities
- `scikit-learn`: KDTree for neighbor search
- `pandas`: CSV file reading 

**Note:** For GPU support, ensure you have CUDA installed and install the appropriate PyTorch version with CUDA support from [pytorch.org](https://pytorch.org/get-started/locally/).


### Optimization

To optimize coreset for kernel regression using gradient descent, use command
```
python coreset_kr_2d.py <dataName> <sigma> <learning_rate> <run_KR_S> <run_random> <randomSample> <scalarfield>
```

**Parameters:**
- `dataName`: Name of the dataset (e.g., 'GTOPO-ME')
- `sigma`: Kernel bandwidth parameter
- `learning_rate`: Learning rate for optimization
- `run_opt`: Boolean flag to run coreset optimization
- `run_random`: Boolean flag to run GR baseline
- `run_randomSample`: Boolean flag to run RS baseline
- `scalarfield`: Name of the scalar field in VTK files (e.g., 'scalar')

### Example
```
python coreset_kr_2d.py GTOPO-ME 5 1 True True True scalar
```

## CITATION

If you found this work useful, please consider citing it as
```
@article{lyu2026coreset,
  title={A Topology-Preserving Coreset for Kernel Regression in Scientific Visualization},
  author={Lyu, Weiran and Gorski, Nathaniel and M. Phillips, Jeff and Wang, Bei},
  journal={IEEE Transactions on Visualization and Computer Graphics},
  year={2026}
}
```

## Notes

- The script supports both CPU and GPU execution
- Code comments are still in progress

