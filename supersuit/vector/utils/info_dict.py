"""Helpers for Gymnasium-style vectorized info dicts.

Gymnasium vector environments return ``infos`` as a dict of arrays (with
boolean ``_key`` masks) rather than a list of per-env dicts. See
``gymnasium.vector.VectorEnv._add_info`` and
``gymnasium.wrappers.vector.DictInfoToList``.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def list_infos_to_dict(infos: list[dict[str, Any]]) -> dict[str, Any]:
    """Convert a list of per-sub-env info dicts into Gymnasium vector info form."""
    vector_infos: dict[str, Any] = {}
    for env_num, env_info in enumerate(infos):
        if not env_info:
            continue
        _add_info(vector_infos, env_info, env_num, len(infos))
    return vector_infos


def merge_vector_infos(
    infos_list: list[dict[str, Any]], num_envs: int
) -> dict[str, Any]:
    """Merge several Gymnasium-style vector info dicts (concat along env axis).

    Each entry in ``infos_list`` is assumed to cover a contiguous block of
    sub-environments whose sizes sum to ``num_envs``. Empty dicts are treated
    as containing no keys for that block.
    """
    if not infos_list:
        return {}

    # Fast path: already a single dict covering everything
    if len(infos_list) == 1:
        return infos_list[0]

    # Determine block sizes from mask/array lengths when present; fall back to
    # equal split only if every block is empty.
    block_sizes: list[int] = []
    remaining = num_envs
    for i, info in enumerate(infos_list):
        if info:
            # Prefer a mask key (starts with _) then any ndarray value
            size = None
            for key, value in info.items():
                if key.startswith("_") and isinstance(value, np.ndarray):
                    size = len(value)
                    break
            if size is None:
                for key, value in info.items():
                    if isinstance(value, np.ndarray):
                        size = len(value)
                        break
                    if isinstance(value, dict):
                        # nested; try one level
                        for v2 in value.values():
                            if isinstance(v2, np.ndarray):
                                size = len(v2)
                                break
                    if size is not None:
                        break
            if size is None:
                size = 0
            block_sizes.append(size)
            remaining -= size
        else:
            block_sizes.append(-1)  # fill later

    # Distribute remaining envs across empty blocks
    empty_idxs = [i for i, s in enumerate(block_sizes) if s < 0]
    if empty_idxs:
        # If we know other sizes, remaining goes to empties; else equal split
        known = sum(s for s in block_sizes if s >= 0)
        leftover = num_envs - known
        base, extra = divmod(leftover, len(empty_idxs))
        for j, i in enumerate(empty_idxs):
            block_sizes[i] = base + (1 if j < extra else 0)

    if sum(block_sizes) != num_envs:
        # As a last resort, try to use equal sizes
        if (
            all(s == block_sizes[0] for s in block_sizes)
            and block_sizes[0] * len(block_sizes) == num_envs
        ):
            pass
        else:
            # Fall back: convert via list form if possible
            raise ValueError(
                f"Cannot merge vector infos: block sizes {block_sizes} do not sum to num_envs={num_envs}"
            )

    offsets = []
    running = 0
    for s in block_sizes:
        offsets.append(running)
        running += s

    merged: dict[str, Any] = {}
    for info, offset, size in zip(infos_list, offsets, block_sizes):
        if not info or size == 0:
            continue
        _merge_block(merged, info, offset, size, num_envs)
    return merged


def _merge_block(
    dest: dict[str, Any],
    src: dict[str, Any],
    offset: int,
    size: int,
    num_envs: int,
) -> None:
    for key, value in src.items():
        if key.startswith("_"):
            continue

        mask_key = f"_{key}"
        src_mask = src.get(mask_key)

        if isinstance(value, dict):
            nested = dest.get(key, {})
            _merge_block(nested, value, offset, size, num_envs)
            # Also merge nested masks that live under nested dict via recursion
            dest[key] = nested
            if src_mask is not None:
                dest_mask = dest.get(mask_key, np.zeros(num_envs, dtype=np.bool_))
                dest_mask[offset : offset + size] = src_mask
                dest[mask_key] = dest_mask
            continue

        if not isinstance(value, np.ndarray):
            # Scalar unexpected at vector level; wrap
            arr = np.full(num_envs, fill_value=None, dtype=object)
            arr[offset : offset + size] = value
            dest[key] = arr
            mask = dest.get(mask_key, np.zeros(num_envs, dtype=np.bool_))
            if src_mask is not None:
                mask[offset : offset + size] = src_mask
            else:
                mask[offset : offset + size] = True
            dest[mask_key] = mask
            continue

        if key not in dest:
            if value.dtype == object:
                dest_arr = np.full(num_envs, fill_value=None, dtype=object)
            else:
                dest_arr = np.zeros((num_envs,) + value.shape[1:], dtype=value.dtype)
            dest[key] = dest_arr
        else:
            dest_arr = dest[key]

        dest_arr[offset : offset + size] = value
        dest[key] = dest_arr

        mask = dest.get(mask_key, np.zeros(num_envs, dtype=np.bool_))
        if src_mask is not None:
            mask[offset : offset + size] = src_mask
        else:
            mask[offset : offset + size] = True
        dest[mask_key] = mask


def _add_info(
    vector_infos: dict[str, Any],
    env_info: dict[str, Any],
    env_num: int,
    num_envs: int,
) -> dict[str, Any]:
    """Port of gymnasium.vector.VectorEnv._add_info (without needing self)."""
    for key, value in env_info.items():
        if key == "final_obs" or key == "terminal_observation":
            if key in vector_infos:
                array = vector_infos[key]
            else:
                array = np.full(num_envs, fill_value=None, dtype=object)
            array[env_num] = value
        elif isinstance(value, dict):
            array = _add_info(vector_infos.get(key, {}), value, env_num, num_envs)
        else:
            if key not in vector_infos:
                if type(value) in [int, float, bool] or issubclass(
                    type(value), np.number
                ):
                    array = np.zeros(num_envs, dtype=type(value))
                elif isinstance(value, np.ndarray):
                    array = np.zeros((num_envs, *value.shape), dtype=value.dtype)
                else:
                    array = np.full(num_envs, fill_value=None, dtype=object)
            else:
                array = vector_infos[key]
            array[env_num] = value

        array_mask = vector_infos.get(f"_{key}", np.zeros(num_envs, dtype=np.bool_))
        array_mask[env_num] = True
        vector_infos[key], vector_infos[f"_{key}"] = array, array_mask
    return vector_infos
