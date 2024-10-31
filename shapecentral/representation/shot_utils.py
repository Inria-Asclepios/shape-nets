import numpy as np


def get_local_rf(values):
    """
    Extracts a local reference frame based on the eigendecomposition of the weighted covariance matrix.
    Arguments are given in a tuple to allow for multiprocessing using multiprocessing.Pool.
    """
    point, neighbors, radius = values
    if neighbors.shape[0] == 0:
        return np.eye(3)

    centered_points = neighbors - point

    radius_minus_distances = radius - np.linalg.norm(centered_points, axis=1)
    weighted_cov_matrix = (centered_points.T @ (centered_points * radius_minus_distances[:, None])
                           / radius_minus_distances.sum())
    eigenvalues, eigenvectors = np.linalg.eigh(weighted_cov_matrix)

    # disambiguating the axes
    # TODO: deal with the equality case (where the two sums below are equal)
    x_orient = centered_points @ eigenvectors[:, 2]
    if (x_orient < 0).sum() > (x_orient >= 0).sum():
        eigenvectors[:, 2] *= -1
    z_orient = centered_points @ eigenvectors[:, 0]
    if (z_orient < 0).sum() > (z_orient >= 0).sum():
        eigenvectors[:, 0] *= -1
    eigenvectors[:, 1] = np.cross(eigenvectors[:, 0], eigenvectors[:, 2])

    return np.flip(eigenvectors, axis=1)


def get_azimuth_idx(x, y):
    """
    Finds the bin index of the azimuth of a point in a division in 8 bins.
    Bins are indexed clockwise, and the first bin is between pi and 3 * pi / 4.
    To use 4 bins instead of 8, divide each factor by 2 and remove the last one that compares |y| and |x|.
    To use 2 bins instead of 8, repeat the process to only keep the condition (y > 0) | ((y == 0) & (x < 0)).
    """
    a = (y > 0) | ((y == 0) & (x < 0))
    b = np.where((x * y > 0) | (x == 0), np.abs(x) < np.abs(y), np.abs(x) > np.abs(y))
    return 4 * a + 2 * np.logical_xor((x > 0) | ((x == 0) & (y > 0)), a) + b


def interpolate_on_adjacent_husks(distance, radius):
    """
    Interpolates on the adjacent husks.
    Assumes there are only two husks, centered around radius / 4 and 3 * radius / 4.

    Args:
        distance: distance or array of distances to the center of the sphere.
        radius: radius of the neighborhood sphere.

    Returns:
        outer_bin: value or array of values to add to the outer bin.
        Equal to 0 if the point is in the outer bin.
        inner_bin: value or array of values to add to the inner bin.
        Equal to 0 if the point is in the inner bin.
        current_bin: value or array of values to add to the current bin.
    """
    radial_bin_size = radius / 2
    inner_bin = ((distance > radius / 2) & (distance < radius * 3 / 4)) * (radius * 3 / 4 - distance) / radial_bin_size

    outer_bin = ((distance < radius / 2) & (distance > radius / 4)) * (distance - radius / 4) / radial_bin_size
    current_bin_inner = (distance < radius / 2) * (1 - np.abs(distance - radius / 4) / radial_bin_size)
    current_bin_outer = (distance > radius / 2) * (1 - np.abs(distance - radius * 3 / 4) / radial_bin_size)
    current_bin = current_bin_inner + current_bin_outer

    return outer_bin, inner_bin, current_bin


def interpolate_vertical_volumes(phi, z):
    """
    Interpolates on the adjacent vertical volumes.
    Assumes there are only two volumes, centered around pi / 4 and 3 * pi / 4.
    The upper volume is the one found for z > 0.

    Args:
        phi: elevation or array of elevations.
        z: vertical coordinate.

    Returns:
        outer_volume: value or array of values to add to the outer volume.
        inner_volume: value or array of values to add to the inner volume.
        current_volume: value or array of values to add to the current volume.
    """
    phi_bin_size = np.pi / 2
    upper_volume = ((phi > np.pi / 2) | ((np.abs(phi - np.pi / 2) < 1e-10) & (z <= 0))) & (phi <= np.pi * 3 / 4)
    upper_volume = upper_volume * (np.pi * 3 / 4 - phi) / phi_bin_size
    lower_volume = ((phi < np.pi / 2) & ((np.abs(phi - np.pi / 2) >= 1e-10) | (z > 0))) & (phi >= np.pi / 4)
    lower_volume = lower_volume * (phi - np.pi / 4) / phi_bin_size

    current_vol_upper = (phi < np.pi / 2) * (1 - np.abs(phi - np.pi / 4) / phi_bin_size)
    current_vol_lower = (phi >= np.pi / 2) * (1 - np.abs(phi - np.pi * 3 / 4) / phi_bin_size)
    current_volume = current_vol_upper + current_vol_lower

    return upper_volume, lower_volume, current_volume


def compute_single_shot_descriptor(values):
    # point, neighbors, normals, radius, eigenvectors, normalize, min_neighborhood_size
    """
    Computes a single SHOT descriptor.
    Arguments are given in a tuple to allow for multiprocessing using multiprocessing.Pool.
    """
    n_cosine_bins, n_azimuth_bins, n_elevation_bins, n_radial_bins = 11, 8, 2, 2

    descriptor = np.zeros((n_cosine_bins, n_azimuth_bins, n_elevation_bins, n_radial_bins))
    point, neighbors, normals, radius, eigenvectors, normalize, min_neighborhood_size = values

    rho = np.linalg.norm(neighbors - point, axis=1)
    if (rho > 0).sum() > min_neighborhood_size:
        neighbors = neighbors[rho > 0]
        local_coordinates = (neighbors - point) @ eigenvectors
        cosine = np.clip(normals[rho > 0] @ eigenvectors[:, 2].T, -1, 1)
        rho = rho[rho > 0]

        order = np.argsort(rho)
        rho = rho[order]
        local_coordinates = local_coordinates[order]
        cosine = cosine[order]

        # computing the spherical coordinates in the local coordinate system
        theta = np.arctan2(local_coordinates[:, 1], local_coordinates[:, 0])
        phi = np.arccos(np.clip(local_coordinates[:, 2] / rho, -1, 1))

        # computing the indices in the histograms
        cos_bin_pos = (cosine + 1.0) * n_cosine_bins / 2.0 - 0.5
        cos_bin_idx = np.rint(cos_bin_pos).astype(int)
        theta_bin_idx = get_azimuth_idx(local_coordinates[:, 0], local_coordinates[:, 1])
        # the two arrays below have to be cast as ints, otherwise they will be treated as masks
        phi_bin_idx = (local_coordinates[:, 2] > 0).astype(int)
        rho_bin_idx = (rho > radius / 2).astype(int)

        # interpolation on the local bins
        delta_cos = cos_bin_pos - cos_bin_idx  # normalized distance with the neighbor bin
        delta_cos_sign = np.sign(delta_cos)  # left-neighbor or right-neighbor
        abs_delta_cos = delta_cos_sign * delta_cos  # probably faster than np.abs
        # noinspection PyRedundantParentheses
        cos_bin_idx_bis = (cos_bin_idx + delta_cos_sign).astype(int) % n_cosine_bins
        descriptor_bis = abs_delta_cos * ((cos_bin_idx > -0.5) & (cos_bin_idx < n_cosine_bins - 0.5))
        descriptor[cos_bin_idx_bis, theta_bin_idx, phi_bin_idx, rho_bin_idx] += descriptor_bis
        descriptor[cos_bin_idx, theta_bin_idx, phi_bin_idx, rho_bin_idx] += (1 - abs_delta_cos)

        # interpolation on the adjacent husks
        outer_bin, inner_bin, current_bin = interpolate_on_adjacent_husks(rho, radius)
        descriptor[cos_bin_idx, theta_bin_idx, phi_bin_idx, 1] += outer_bin * (rho_bin_idx == 0)
        descriptor[cos_bin_idx, theta_bin_idx, phi_bin_idx, 0] += inner_bin * (rho_bin_idx == 1)
        descriptor[cos_bin_idx, theta_bin_idx, phi_bin_idx, rho_bin_idx] += current_bin

        # interpolation between adjacent vertical volumes
        upper_volume, lower_volume, current_volume = interpolate_vertical_volumes(phi, local_coordinates[:, 2])
        descriptor[cos_bin_idx, theta_bin_idx, 1, rho_bin_idx] += upper_volume * (phi_bin_idx == 0)
        descriptor[cos_bin_idx, theta_bin_idx, 0, rho_bin_idx] += lower_volume * (phi_bin_idx == 1)
        descriptor[cos_bin_idx, theta_bin_idx, phi_bin_idx, rho_bin_idx] += current_volume

        # interpolation between adjacent horizontal volumes
        # local_coordinates[:, 0] * local_coordinates[:, 1] != 0
        theta_bin_size = 2 * np.pi / n_azimuth_bins
        delta_theta = np.clip((theta - (-np.pi + theta_bin_idx * theta_bin_size)) / theta_bin_size - 0.5, -0.5, 0.5, )
        delta_theta_sign = np.sign(delta_theta)  # left-neighbor or right-neighbor
        abs_delta_theta = delta_theta_sign * delta_theta

        theta_bin_idx_bis = (theta_bin_idx + delta_theta_sign).astype(int) % n_azimuth_bins
        descriptor[cos_bin_idx, theta_bin_idx_bis, phi_bin_idx, rho_bin_idx] += abs_delta_theta
        descriptor[cos_bin_idx, theta_bin_idx, phi_bin_idx, rho_bin_idx] += (1 - abs_delta_theta)

        # normalizing the descriptor to Euclidian norm 1
        if (descriptor_norm := np.linalg.norm(descriptor)) > 0:
            if normalize:
                return descriptor / descriptor_norm  # .ravel()
            else:
                return descriptor  # .ravel()
    return np.zeros(n_cosine_bins * n_azimuth_bins * n_elevation_bins * n_radial_bins)
