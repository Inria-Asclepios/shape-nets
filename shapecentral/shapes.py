import pyvista as pv
import numpy as np
import scipy as sp
import igl
import torch
from scipy.ndimage import gaussian_filter
from networks import diffusion_net

from .representation import Shot


class Surface:
    def __init__(self, vertices=None, faces=None, shape=None):
        self.vertices = vertices
        self.faces = faces
        if self.vertices is not None:
            self.normals = igl.per_vertex_normals(vertices, faces)

        if shape is not None:
            self.load_shape(shape)
        else:
            self.shape = pv.PolyData.from_regular_faces(vertices, faces)

        self.k = None
        self.h = None
        self.kmin = None
        self.kmax = None
        self.hks = None
        self.shot = None

    def load_shape(self, shape):
        if isinstance(shape, str):
            self.shape = pv.read(shape)
        elif isinstance(shape, pv.DataSet):
            self.shape = shape.copy(deep=True)
        else:
            raise ValueError(f'Unexpected type {type(shape)!r}, only string or pyvista dataset allowed')
        self.vertices = self.shape.points
        self.faces = self.shape.faces.reshape(-1, 4)[:, 1:]
        self.normals = igl.per_vertex_normals(self.vertices, self.faces)

    def compute_curvature_representers(self, filter_sigma=None):
        self.compute_gaussian_curvature()
        self.compute_mean_curvature()
        self.compute_principal_curvature()

        if filter_sigma is not None:
            self.k = gaussian_filter(self.k, sigma=filter_sigma)
            self.h = gaussian_filter(self.h, sigma=filter_sigma)
            self.kmin = gaussian_filter(self.kmin, sigma=filter_sigma)
            self.kmax = gaussian_filter(self.kmax, sigma=filter_sigma)

    def compute_mean_curvature(self, method='laplace'):
        if method == 'laplace':
            l = igl.cotmatrix(self.vertices, self.faces)
            m = igl.massmatrix(self.vertices, self.faces, igl.MASSMATRIX_TYPE_VORONOI)
            minv = sp.sparse.diags(1 / m.diagonal())
            hn = -minv.dot(l.dot(self.vertices))
            self.h = np.linalg.norm(hn, axis=1) / 2
        elif method == 'quadric':
            if self.kmin is None:
                self.compute_principal_curvature()
            self.h = (self.kmin + self.kmax) / 2
        else:
            raise ValueError('method should be either "quadric" or "laplace"')


    def compute_gaussian_curvature(self, method='rusinkiewicz'):
        if method=='rusinkiewicz':
            self.k = igl.gaussian_curvature(self.vertices, self.faces)
        elif method=='quadric':
            if self.kmin is None:
                self.compute_principal_curvature()
            self.k = self.kmin * self.kmax
        else:
            raise ValueError('method should be either "quadric" or "rusinkiewicz"')


    def compute_principal_curvature(self):
        _, _, self.kmin, self.kmax = igl.principal_curvature(self.vertices, self.faces)


    def compute_signature(self, n_dim=16, n_eigvecs=128):
        verts = torch.tensor(np.ascontiguousarray(self.vertices)).float()
        faces = torch.tensor(np.ascontiguousarray(self.faces))
        _, _, _, evals, evecs, _, _ = diffusion_net.geometry.get_operators(verts, faces, k_eig=n_eigvecs)
        self.hks = diffusion_net.geometry.compute_hks_autoscale(evals, evecs, count=n_dim)



    def compute_shot_descriptors(self, radius=12, n_desc=128, frame_radius=12, n_process=8, verbose=True):
        mr = igl.avg_edge_length(self.vertices, self.faces)
        rad = radius*mr
        frame_rad = frame_radius*mr

        with Shot(n_procs=n_process, disable_progress_bar=not verbose) as shot_cpt:
            self.shot = shot_cpt.compute_descriptor_bi_scale(self.vertices, self.normals, frame_rad, rad)
        assert n_desc in [64, 128, 256, 352], 'choice of number of descripted is restricted to 64, 128, 256 or 352'
        if n_desc == 64:
            self.shot = self.shot[:, :4, :4, :, :]
        if n_desc == 128:
            self.shot = self.shot[:, :8, :4, :, :]
        if n_desc == 256:
            self.shot = self.shot[:, :8, :, :, :]

        self.shot = self.shot.reshape((len(self.vertices), -1))


    def as_polydata(self):
        representers = [self.k, self.h, self.kmin, self.kmax,
                        self.vmin, self.vmax, self.shape_op,
                        self.hks, self.wks, self.shot, self.normals]
        representer_names = ['k', 'h', 'kmin', 'kmax', 'vmin', 'vmax', 'shape_op', 'hks', 'wks', 'shot', 'nor']

        for i in range(len(representers)):
            r = representers[i]
            if r is not None:
                n = representer_names[i]
                self.shape[n] = r
        return self.shape
