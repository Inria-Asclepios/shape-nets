# Shape-nets

This is the repository for the experiments from the paper: "Improving Network Surface Processing with Principal Curvatures"


## Models
Most of the code comes from the repositories of the models that we used. For anyone wanting to use them, we strongly suggest to go straight to these implementations:

- [Diffusion-net](https://github.com/nmwsharp/diffusion-net/tree/master)
- [Delta-net](https://github.com/rubenwiersma/deltaconv)
- [Point-net++](https://github.com/pyg-team/pytorch_geometric/tree/master/examples) -- following the pytorch-geometric implementation

## Data
We followed the instructions from the [Diffusion-net repo](https://github.com/nmwsharp/diffusion-net/tree/master/experiments) to get all three datasets.

## Shape representations

For **principal curvatures** (kmin, kmax), and **gaussian curvature**, we suggest following implementations from [libigl](https://libigl.github.io/libigl-python-bindings/):
```python
import igl
import pyvista as pv

mesh = pv.read('\path\to\mesh.vtk') # We use pyvista for loading and visualisation
vertices = mesh.points
faces = mesh.faces.reshape(-1, 4)[:, 1:]

_, _, kmin, kmax = igl.principal_curvature(verts, faces)
k = igl.gaussian_curvature(verts, faces)
```


For the **Heat Kernel Signature**, we follow the implementation using the [Robust-laplacian](https://github.com/nmwsharp/robust-laplacians-py) :

```python
import numpy as np
import scipy
import scipy.sparse.linalg as sla
import robust_laplacian

L, M = robust_laplacian.mesh_laplacian(verts, faces)
massvec = M.diagonal()
L_eigsh = (L + scipy.sparse.identity(L.shape[0]) * eps).tocsc()
Mmat = scipy.sparse.diags(massvec)
evals, evecs = sla.eigsh(L_eigsh, k=128, M=Mmat, sigma=eps) # k=number of eigenvectors
evals = np.clip(evals, a_min=0., a_max=float('inf'))
scales = np.logspace(-2, 0, num=16)  # num=dimension of the HKS
power_coefs = np.exp(-evals * scales)
hks = power_coefs * (evecs * evecs)
```

For the **SHOT Descriptors**, we used the implementation proposed by [Point Cloud Library](https://github.com/PointCloudLibrary/pcl). 
To be found [here](https://pointclouds.org/documentation/classpcl_1_1_s_h_o_t_estimation_base.html)


## Installation

If you wish to install the same environment and use shapecentral directly:

```
git clone https://github.com/Inria-Asclepios/shape-nets
cd shape-nets
pip install -e .
```
