import numpy as np
import os
import vtk
from pyevtk.hl import imageToVTK
from vtk.util.numpy_support import vtk_to_numpy
import pandas as pd


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

def readVTKFile(filename):
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(filename)
    reader.Update()  # Read the .vti file
    image_data = reader.GetOutput()

    dims = image_data.GetDimensions()

    vtk_array = image_data.GetPointData().GetScalars()
    numpy_array = vtk_to_numpy(vtk_array)

    numpy_array_reshaped = np.squeeze(numpy_array.reshape(dims[1], dims[0]))
    return numpy_array_reshaped

def Numpy2VTK(data, newName, scalarfield):
    imageToVTK(newName, pointData = {scalarfield:data.reshape(data.shape[0], data.shape[1], 1)})

def getData(data):
    dx, dy = np.gradient(data)
    gradient_magnitude = np.sqrt(dx**2 + dy**2)
    gradient = gradient_magnitude.reshape(-1)
    x_indices, y_indices = np.meshgrid(np.arange(data.shape[0]), np.arange(data.shape[1]), indexing="ij")

    x_indices = x_indices.ravel()
    y_indices = y_indices.ravel()
    data_flat = data.ravel()

    # newData = np.column_stack((x_indices, y_indices, data_flat, gradient_flat))
    newData = np.column_stack((x_indices, y_indices, data_flat))
    return newData, gradient

def getQueryPoints(data, xeval=100, yeval=100):
    x_min, x_max = data[:, 0].min(), data[:, 0].max()+1
    y_min, y_max = data[:, 1].min(), data[:, 1].max()+1
    x_query, y_query = np.linspace(x_min, x_max, xeval, dtype=int), np.linspace(y_min, y_max, yeval, dtype=int)
    # x_query, y_query = np.arange(x_min, x_max + 1, xeval), np.arange(y_min, y_max + 1, yeval)
    X_query, Y_query = np.meshgrid(x_query, y_query)
    # X_query, Y_query = np.meshgrid(y_query, x_query)
    print('mesh xy eval:', xeval, yeval)

    query_points = np.c_[X_query.ravel(), Y_query.ravel()]
    return query_points, X_query, Y_query

def read_criticalpoints(filename, factor, data, nx , ny):
    cp_df = pd.read_csv(filename)
    cp_df.reset_index(inplace=True)
    criticalpoints = cp_df[['Points_0', 'Points_1', 'Scalar']].to_numpy()
    criticalpoints = cp_df[['Points_1', 'Points_0']].to_numpy()
    criticalpoints[:,:2] *= factor #rescale by the evaluation factor to map back to the original data locations
    indices = [(int(x * ny + y)) for x, y in criticalpoints]
    scalars = data[indices][:,2]
    criticalpoints = np.column_stack((criticalpoints, scalars))
    # criticalpoints = cp_df[['Points_1', 'Points_0', 'Scalar', 'VertexId']].to_numpy()
    return criticalpoints

def read_maxima_minima(filename):
    cp_df = pd.read_csv(filename)
    cp_df.reset_index(inplace=True)
    criticalpoints = cp_df[['Points_1', 'Points_0', 'CriticalType']].to_numpy()
    maxima_minima = criticalpoints[(criticalpoints[:, 2] == 0) | (criticalpoints[:, 2] == 3),:2]
    return maxima_minima