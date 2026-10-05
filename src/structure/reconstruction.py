from __future__ import annotations

import numpy as np


def classical_mds(
    distance_matrix: np.ndarray,
    dimensions: int = 3,
) -> np.ndarray:
    """
    Reconstruct Cartesian coordinates from a distance matrix
    using classical multidimensional scaling.
    """
    distance_matrix = np.asarray(
        distance_matrix,
        dtype=np.float64,
    )

    if distance_matrix.ndim != 2:
        raise ValueError(
            "Distance matrix must be 2D."
        )

    if (
        distance_matrix.shape[0]
        != distance_matrix.shape[1]
    ):
        raise ValueError(
            "Distance matrix must be square."
        )

    distance_matrix = (
        distance_matrix
        + distance_matrix.T
    ) / 2.0

    np.fill_diagonal(
        distance_matrix,
        0.0,
    )

    distance_matrix = np.maximum(
        distance_matrix,
        0.0,
    )

    n = distance_matrix.shape[0]

    centering = (
        np.eye(n)
        - np.ones((n, n)) / n
    )

    squared_distances = (
        distance_matrix ** 2
    )

    gram_matrix = (
        -0.5
        * centering
        @ squared_distances
        @ centering
    )

    eigenvalues, eigenvectors = np.linalg.eigh(
        gram_matrix
    )

    order = np.argsort(
        eigenvalues
    )[::-1]

    eigenvalues = eigenvalues[
        order
    ]

    eigenvectors = eigenvectors[
        :,
        order,
    ]

    eigenvalues = np.maximum(
        eigenvalues,
        0.0,
    )

    dimensions = min(
        dimensions,
        n,
    )

    coordinates = (
        eigenvectors[:, :dimensions]
        * np.sqrt(
            eigenvalues[:dimensions]
        )
    )

    return coordinates.astype(
        np.float32
    )


def kabsch_align(
    predicted: np.ndarray,
    target: np.ndarray,
    allow_reflection: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Align predicted coordinates to target coordinates.

    When allow_reflection=False, only proper rotations are allowed.

    When allow_reflection=True, mirror transformations are also allowed.
    """
    predicted = np.asarray(
        predicted,
        dtype=np.float64,
    )

    target = np.asarray(
        target,
        dtype=np.float64,
    )

    if predicted.shape != target.shape:
        raise ValueError(
            "Predicted and target coordinates "
            "must have the same shape."
        )

    if predicted.ndim != 2:
        raise ValueError(
            "Coordinates must be a 2D array."
        )

    if predicted.shape[1] != 3:
        raise ValueError(
            "Coordinates must have shape [N, 3]."
        )

    predicted_center = predicted.mean(
        axis=0
    )

    target_center = target.mean(
        axis=0
    )

    predicted_centered = (
        predicted
        - predicted_center
    )

    target_centered = (
        target
        - target_center
    )

    covariance = (
        predicted_centered.T
        @ target_centered
    )

    u, _, vt = np.linalg.svd(
        covariance
    )

    rotation = (
        u @ vt
    )

    if not allow_reflection:
        if np.linalg.det(rotation) < 0:
            vt[-1, :] *= -1

            rotation = (
                u @ vt
            )

    aligned = (
        predicted_centered
        @ rotation
    )

    aligned += target_center

    return (
        aligned.astype(np.float32),
        rotation.astype(np.float32),
    )


def calculate_rmsd(
    predicted: np.ndarray,
    target: np.ndarray,
) -> float:
    """
    Calculate RMSD between two coordinate sets.
    """
    predicted = np.asarray(
        predicted,
        dtype=np.float64,
    )

    target = np.asarray(
        target,
        dtype=np.float64,
    )

    if predicted.shape != target.shape:
        raise ValueError(
            "Predicted and target coordinates "
            "must have the same shape."
        )

    difference = (
        predicted
        - target
    )

    return float(
        np.sqrt(
            np.mean(
                np.sum(
                    difference ** 2,
                    axis=1,
                )
            )
        )
    )


def reconstruct_and_align(
    distance_matrix: np.ndarray,
    target_coordinates: np.ndarray,
    allow_reflection: bool = False,
) -> tuple[np.ndarray, float]:
    """
    Reconstruct coordinates from predicted distances,
    align them to experimental coordinates,
    and calculate RMSD.
    """
    predicted_coordinates = classical_mds(
        distance_matrix
    )

    aligned_coordinates, _ = kabsch_align(
        predicted_coordinates,
        target_coordinates,
        allow_reflection=allow_reflection,
    )

    rmsd = calculate_rmsd(
        aligned_coordinates,
        target_coordinates,
    )

    return (
        aligned_coordinates,
        rmsd,
    )


def calculate_distance_matrix(
    coordinates: np.ndarray,
) -> np.ndarray:
    """
    Calculate pairwise Euclidean distances.
    """
    coordinates = np.asarray(
        coordinates,
        dtype=np.float32,
    )

    if coordinates.ndim != 2:
        raise ValueError(
            "Coordinates must be a 2D array."
        )

    if coordinates.shape[1] != 3:
        raise ValueError(
            "Coordinates must have shape [N, 3]."
        )

    differences = (
        coordinates[:, None, :]
        - coordinates[None, :, :]
    )

    return np.linalg.norm(
        differences,
        axis=-1,
    )