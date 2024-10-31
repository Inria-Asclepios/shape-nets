import pyvista as pv
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import polyscope as ps
import numpy as np

def main():
    surface = pv.read('Datasets/Shapes/RNA/shapes/169.vtk')
    vertices = surface.points
    faces = surface.faces.reshape(-1, 4)[:, 1:]
    labels = surface['labels']
    print(len(np.unique(labels)))
    ps.init()
    ps_mesh = ps.register_surface_mesh("test mesh", vertices, faces)
    ps_mesh.add_scalar_quantity("labels", labels, defined_on='vertices')
    ps.show()

    pl = pv.Plotter()
    pl.add_mesh(surface, scalars='labels', cmap=plt.cm.get_cmap("jet", 118), categories=True, interpolate_before_map=False)
    pl.show()

if __name__ == "__main__":
    main()
