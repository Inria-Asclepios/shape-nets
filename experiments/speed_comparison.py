import time
import pyvista as pv
import igl
import robust_laplacian
import scipy
import numpy as np
import scipy.sparse.linalg as sla


def compute_hks(v, f, n_eig=16, eps=1e-8):
    L, M = robust_laplacian.mesh_laplacian(v, f)
    massvec = M.diagonal()
    L_eigsh = (L + scipy.sparse.identity(L.shape[0]) * eps).tocsc()
    Mmat = scipy.sparse.diags(massvec)
    evals, evecs = sla.eigsh(L_eigsh, k=n_eig, M=Mmat, sigma=eps)
    evals = np.clip(evals, a_min=0., a_max=float('inf'))
    scales = np.logspace(-2, 0, num=16)
    power_coefs = np.exp(-evals * scales)
    hks = power_coefs * (evecs * evecs)
    return hks

def main():
    bunny_coarse = pv.examples.download_bunny_coarse().clean()  # 502
    plane = pv.examples.load_airplane().clean()  # 1335
    jupyter = pv.examples.planets.load_jupiter(lon_resolution=50).triangulate().clean()  # 2500
    human_0 = pv.read('Datasets/Shapes/Human_pose/shapes/0.vtk').clean()  # 4706
    shrec_0 = pv.read('Datasets/Shapes/Shrec_16/shapes/0.vtk').clean()  # 9281
    rna_0 = pv.read('Datasets/Shapes/RNA/shapes/0.vtk')  # 15099
    bunny = pv.examples.download_bunny().clean()  # 35947
    lucy = pv.examples.download_lucy().clean()   # 49987
    face = pv.examples.download_face2().clean()   # 112862
    armadillo = pv.examples.download_armadillo().clean()   # 172 974
    louis = pv.examples.download_louis_louvre().clean()   # 210873
    dragon = pv.examples.download_dragon().clean()   # 437645

    meshes = [bunny_coarse, plane, jupyter, human_0, shrec_0, rna_0, bunny, lucy, face, armadillo, louis, dragon]

    print(f'Starting')
    k_times = []
    c = 0
    for mesh in meshes:
        c += 1
        print(f'Starting mesh {c}/12')
        vertices = np.array(mesh.points)
        faces = np.array(mesh.faces.reshape(-1, 4)[:, 1:])
        t0 = time.time()
        _, _, k1, k2 = igl.principal_curvature(vertices, faces)
        t1 = time.time()
        k_times.append(np.round(t1 - t0, 3))

    print(f'Curvature done in {np.sum(k_times)}')
    hks_times = []
    c = 0
    for mesh in meshes:
        c += 1
        print(f'Starting mesh {c}/12')
        vertices = np.array(mesh.points)
        faces = np.array(mesh.faces.reshape(-1, 4)[:, 1:])
        t0 = time.time()
        hks = compute_hks(vertices, faces)
        t1 = time.time()
        hks_times.append(np.round(t1 - t0, 3))

    print(f'HKS done in {np.sum(hks_times)}')
    breakpoint()


if __name__ == '__main__':
    main()
