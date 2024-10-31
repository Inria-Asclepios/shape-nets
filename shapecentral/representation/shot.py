from dataclasses import dataclass
from multiprocessing import Pool

import numpy as np
from sklearn.neighbors import KDTree
from tqdm import tqdm

from .shot_utils import get_local_rf, compute_single_shot_descriptor


@dataclass
class Shot:
    """
    Base class to compute SHOT descriptors in parallel on multiple processes.
    """
    def __init__(self, min_neighborhood_size=100, n_procs=8, disable_progress_bar=False):
        self.normalize = True
        self.share_local_rfs = True
        self.min_neighborhood_size = min_neighborhood_size
        self.n_procs = n_procs
        self.disable_progress_bar = disable_progress_bar

    def __enter__(self):
        self.pool = Pool(processes=self.n_procs)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.pool.terminate()
        else:
            self.pool.close()
        self.pool.join()

    def compute_local_rf(self, keypoints, neighborhoods, support, radius):
        """
        Parallelization of the function get_local_rf.

        Args:
            support: The supporting point cloud.
            keypoints: The keypoints to compute local reference frames on.
            radius: The radius used to compute the local reference frames.
            neighborhoods: The neighborhoods associated with each keypoint. neighborhoods[i] should be an array of ints.

        Returns:
            The local reference frames computed on every keypoint.
        """
        kps, nbs = keypoints, neighborhoods
        rf = tqdm(self.pool.imap(get_local_rf, [(kp, support[nbs[i]], radius) for i, kp in enumerate(kps)]),
                  desc=f"Local RFs with radius {radius}", total=keypoints.shape[0], disable=self.disable_progress_bar)
        return np.array(list(rf))

    def compute_descriptor(self, keypoints, normals, neighborhoods, local_rfs, radius):
        """
        Parallelization of the function compute_single_shot_descriptor.

        Args:
            keypoints: The keypoints to compute descriptors on.
            normals: The normals of points in the support.
            neighborhoods: The neighborhoods associated with each keypoint. neighborhoods[i] should be an array of ints.
            local_rfs: The local reference frames associated with each keypoint.
            support: The supporting point cloud.
            radius: The radius used to compute SHOT.

        Returns:
            The descriptor computed on every keypoint.
        """

        d = tqdm(self.pool.imap(compute_single_shot_descriptor, [(keypoint,
                                                                  keypoints[neighborhoods[i]],
                                                                  normals[neighborhoods[i]],
                                                                  radius,
                                                                  local_rfs[i],
                                                                  self.normalize,
                                                                  self.min_neighborhood_size)
                                                                 for i, keypoint in enumerate(keypoints)],
                                chunksize=int(np.ceil(keypoints.shape[0] / (2 * self.n_procs)))),
                 desc=f"SHOT desc with radius {radius}", total=keypoints.shape[0], disable=self.disable_progress_bar)
        return np.array(list(d))

    def compute_descriptor_single_scale(self, point_cloud, normals, radius):
        """
        Computes the SHOT descriptor on a single scale.
        Normals are expected to be normalized to 1.

        Args:
            point_cloud: The entire point cloud.
            normals: The normals computed on the point cloud.
            keypoints: The keypoints to compute descriptors on.
            radius: Radius used to compute the SHOT descriptors.

        Returns:
            The descriptor as a (self.keypoints.shape[0], 352) array.
        """

        neighborhoods = KDTree(point_cloud).query_radius(point_cloud, radius)
        local_rfs = self.compute_local_rf(point_cloud, neighborhoods, point_cloud, radius)

        return self.compute_descriptor(point_cloud, normals, neighborhoods, local_rfs, radius)

    def compute_descriptor_bi_scale(self, point_cloud, normals, local_rf_radius, shot_radius):
        """
        Computes the SHOT descriptor on a point cloud with two distinct radii: one for the computation of the local
        reference frames and the other one for the computation of the descriptor.
        Normals are expected to be normalized to 1.

        Args:
            point_cloud: The entire point cloud.
            normals: The normals computed on the point cloud.
            keypoints: The keypoints to compute descriptors on.
            local_rf_radius: Radius used to compute the local reference frames.
            shot_radius: Radius used to compute the SHOT descriptors.

        Returns:
            The descriptor as a (self.keypoints.shape[0], 352) array.
        """

        neighborhoods = KDTree(point_cloud).query_radius(point_cloud, local_rf_radius)
        local_rfs = self.compute_local_rf(point_cloud, neighborhoods, point_cloud, local_rf_radius)
        neighborhoods = KDTree(point_cloud).query_radius(point_cloud, shot_radius)
        return self.compute_descriptor(point_cloud, normals, neighborhoods, local_rfs, shot_radius)

    def compute_descriptor_multiscale(self, point_cloud, normals, keypoints, radii, weights):
        """
        Computes the SHOT descriptor on multiple scales.
        Normals are expected to be normalized to 1.

        Args:
            point_cloud: The entire point cloud.
            normals: The normals computed on the point cloud.
            keypoints: The keypoints to compute descriptors on.
            radii: The radii to compute the descriptors with.
            weights: The weights to multiply each scale with. Leave empty to multiply by 1.

        Returns:
            The descriptor as a (self.keypoints.shape[0], 352 * n_scales) array.
        """
        if weights is None:
            weights = np.ones(len(radii))

        all_descriptors = np.zeros((len(radii), keypoints.shape[0], 256)) # 352

        local_rfs = None
        for scale, radius in enumerate(radii):
            neighborhoods = KDTree(point_cloud).query_radius(keypoints, radius)
            # if shared, only using the smallest radius to determine the local RF
            if local_rfs is None or not self.share_local_rfs:
                local_rfs = self.compute_local_rf(keypoints, neighborhoods, point_cloud, radius)

            desc_scale = self.compute_descriptor(keypoints, normals, neighborhoods, local_rfs, radius, point_cloud)
            all_descriptors[scale, :, :] = desc_scale * weights[scale]

        return all_descriptors.reshape(keypoints.shape[0], 256 * len(radii)) # 352
