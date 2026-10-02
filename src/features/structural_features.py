from __future__ import annotations

import numpy as np


def contacts_to_distance_map(
    contact_map: np.ndarray,
    contact_distance: float = 8.0,
    non_contact_distance: float = 20.0,
) -> np.ndarray:
    """
    Convert a binary contact map into an approximate distance map.
    """
    contact_map = np.asarray(contact_map)

    if contact_map.ndim != 2:
        raise ValueError("Contact map must be a 2D array.")

    if contact_map.shape[0] != contact_map.shape[1]:
        raise ValueError("Contact map must be square.")

    distance_map = np.where(
        contact_map > 0,
        contact_distance,
        non_contact_distance,
    ).astype(np.float32)

    np.fill_diagonal(distance_map, 0.0)

    return distance_map


def distance_map_to_contacts(
    distance_map: np.ndarray,
    threshold: float = 8.0,
) -> np.ndarray:
    """
    Convert a distance map into a binary contact map.
    """
    distance_map = np.asarray(distance_map)

    if distance_map.ndim != 2:
        raise ValueError("Distance map must be a 2D array.")

    if distance_map.shape[0] != distance_map.shape[1]:
        raise ValueError("Distance map must be square.")

    contact_map = (
        distance_map <= threshold
    ).astype(np.float32)

    np.fill_diagonal(contact_map, 0.0)

    return contact_map


def validate_distance_map(
    distance_map: np.ndarray,
) -> None:
    """
    Validate a protein pairwise distance matrix.
    """
    distance_map = np.asarray(distance_map)

    if distance_map.ndim != 2:
        raise ValueError("Distance map must be a 2D array.")

    if distance_map.shape[0] != distance_map.shape[1]:
        raise ValueError("Distance map must be square.")

    if np.any(distance_map < 0):
        raise ValueError("Distance values cannot be negative.")

    if not np.allclose(
        distance_map,
        distance_map.T,
        atol=1e-5,
    ):
        raise ValueError("Distance map must be symmetric.")


def normalize_distance_map(
    distance_map: np.ndarray,
    max_distance: float = 20.0,
) -> np.ndarray:
    """
    Normalize pairwise distances to [0, 1].
    """
    validate_distance_map(distance_map)

    if max_distance <= 0:
        raise ValueError("max_distance must be positive.")

    normalized = np.clip(
        distance_map / max_distance,
        0.0,
        1.0,
    )

    return normalized.astype(np.float32)