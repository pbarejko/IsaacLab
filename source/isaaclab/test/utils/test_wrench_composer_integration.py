# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Integration tests for WrenchComposer with mock assets.

These tests verify that the WrenchComposer integrates correctly with the mock asset
classes (MockRigidObjectCollection), focusing on the dual-buffer API, the compose_to_body_frame
method, the add_raw_buffers_from method, and the global_only flag.
"""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

import numpy as np
import pytest
import torch
import warp as wp

from isaaclab.test.mock_interfaces.assets import MockRigidObjectCollection
from isaaclab.utils.wrench_composer import WrenchComposer


# --- Helper functions ---


def create_mock_asset(
    num_envs: int,
    num_bodies: int,
    device: str,
    link_pos: torch.Tensor | None = None,
    link_quat: torch.Tensor | None = None,
) -> MockRigidObjectCollection:
    """Create a MockRigidObjectCollection with optional custom link poses."""
    mock = MockRigidObjectCollection(num_instances=num_envs, num_bodies=num_bodies, device=device)

    if link_pos is None:
        pos = torch.zeros(num_envs, num_bodies, 3, dtype=torch.float32)
    else:
        pos = link_pos.float()

    if link_quat is None:
        # Identity quaternion in (x, y, z, w) format = (0, 0, 0, 1)
        quat = torch.zeros(num_envs, num_bodies, 4, dtype=torch.float32)
        quat[..., 3] = 1.0
    else:
        quat = link_quat.float()

    pose = torch.cat([pos, quat], dim=-1)  # (N, B, 7)
    mock.data.set_body_link_pose_w(pose)
    return mock


def quat_rotate_inv_np(quat_xyzw: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate a vector by the inverse of a quaternion (numpy)."""
    xyz = quat_xyzw[..., 0:3]
    w = quat_xyzw[..., 3:4]
    t = 2.0 * np.cross(-xyz, vec, axis=-1)
    return vec + w * t + np.cross(-xyz, t, axis=-1)


def random_unit_quaternion_np(rng: np.random.Generator, shape: tuple) -> np.ndarray:
    """Generate random unit quaternions in (x, y, z, w) format."""
    q = rng.standard_normal(shape + (4,)).astype(np.float32)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    return q


# ============================================================================
# Tests for compose_to_body_frame
# ============================================================================


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_compose_to_body_frame_local_only(device: str):
    """Test that compose_to_body_frame correctly copies local buffers to output."""
    rng = np.random.default_rng(seed=100)
    num_envs, num_bodies = 4, 3

    mock_asset = create_mock_asset(num_envs, num_bodies, device)
    wc = WrenchComposer(mock_asset)

    # Add local forces and torques
    forces_np = rng.uniform(-50.0, 50.0, (num_envs, num_bodies, 3)).astype(np.float32)
    torques_np = rng.uniform(-50.0, 50.0, (num_envs, num_bodies, 3)).astype(np.float32)
    forces = wp.from_numpy(forces_np, dtype=wp.vec3f, device=device)
    torques = wp.from_numpy(torques_np, dtype=wp.vec3f, device=device)

    wc.add_forces_and_torques(forces=forces, torques=torques, is_global=False)

    assert wc.active
    assert wc._dirty
    assert wc._has_local
    assert not wc._has_global

    wc.compose_to_body_frame()

    assert not wc._dirty
    assert np.allclose(wc.out_force_b.numpy(), forces_np, atol=1e-5)
    assert np.allclose(wc.out_torque_b.numpy(), torques_np, atol=1e-5)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_compose_to_body_frame_global_only(device: str):
    """Test that compose_to_body_frame rotates global forces to body frame."""
    rng = np.random.default_rng(seed=101)
    num_envs, num_bodies = 4, 3

    link_quat_np = random_unit_quaternion_np(rng, (num_envs, num_bodies))
    link_quat_torch = torch.from_numpy(link_quat_np)
    mock_asset = create_mock_asset(num_envs, num_bodies, device, link_quat=link_quat_torch)
    wc = WrenchComposer(mock_asset)

    forces_global_np = rng.uniform(-50.0, 50.0, (num_envs, num_bodies, 3)).astype(np.float32)
    forces_global = wp.from_numpy(forces_global_np, dtype=wp.vec3f, device=device)

    wc.add_forces_and_torques(forces=forces_global, is_global=True)

    assert wc.global_only

    wc.compose_to_body_frame()

    expected_forces_local = quat_rotate_inv_np(link_quat_np, forces_global_np)
    assert np.allclose(wc.out_force_b.numpy(), expected_forces_local, atol=1e-4)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_compose_clears_dirty_flag(device: str):
    """Test that compose_to_body_frame clears the dirty flag."""
    num_envs, num_bodies = 2, 2
    mock_asset = create_mock_asset(num_envs, num_bodies, device)
    wc = WrenchComposer(mock_asset)

    forces = wp.from_numpy(
        np.ones((num_envs, num_bodies, 3), dtype=np.float32), dtype=wp.vec3f, device=device
    )
    wc.add_forces_and_torques(forces=forces, is_global=False)

    assert wc._dirty
    wc.compose_to_body_frame()
    assert not wc._dirty


# ============================================================================
# Tests for add_raw_buffers_from
# ============================================================================


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_add_raw_buffers_from_local(device: str):
    """Test add_raw_buffers_from merges local buffers from another composer."""
    rng = np.random.default_rng(seed=200)
    num_envs, num_bodies = 4, 3

    mock1 = create_mock_asset(num_envs, num_bodies, device)
    mock2 = create_mock_asset(num_envs, num_bodies, device)

    wc1 = WrenchComposer(mock1)
    wc2 = WrenchComposer(mock2)

    forces1_np = rng.uniform(-50.0, 50.0, (num_envs, num_bodies, 3)).astype(np.float32)
    forces2_np = rng.uniform(-50.0, 50.0, (num_envs, num_bodies, 3)).astype(np.float32)

    wc1.add_forces_and_torques(
        forces=wp.from_numpy(forces1_np, dtype=wp.vec3f, device=device), is_global=False
    )
    wc2.add_forces_and_torques(
        forces=wp.from_numpy(forces2_np, dtype=wp.vec3f, device=device), is_global=False
    )

    # Merge wc2 into wc1
    wc1.add_raw_buffers_from(wc2)

    # Local force buffer should be sum of both
    expected = forces1_np + forces2_np
    assert np.allclose(wc1.local_force_b.numpy(), expected, atol=1e-5)

    # After compose, output should match
    wc1.compose_to_body_frame()
    assert np.allclose(wc1.out_force_b.numpy(), expected, atol=1e-5)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_add_raw_buffers_from_global(device: str):
    """Test add_raw_buffers_from merges global buffers from another composer."""
    rng = np.random.default_rng(seed=201)
    num_envs, num_bodies = 4, 3

    link_quat_np = random_unit_quaternion_np(rng, (num_envs, num_bodies))
    link_quat_torch = torch.from_numpy(link_quat_np)

    mock1 = create_mock_asset(num_envs, num_bodies, device, link_quat=link_quat_torch)
    mock2 = create_mock_asset(num_envs, num_bodies, device, link_quat=link_quat_torch)

    wc1 = WrenchComposer(mock1)
    wc2 = WrenchComposer(mock2)

    forces1_np = rng.uniform(-50.0, 50.0, (num_envs, num_bodies, 3)).astype(np.float32)
    forces2_np = rng.uniform(-50.0, 50.0, (num_envs, num_bodies, 3)).astype(np.float32)

    wc1.add_forces_and_torques(
        forces=wp.from_numpy(forces1_np, dtype=wp.vec3f, device=device), is_global=True
    )
    wc2.add_forces_and_torques(
        forces=wp.from_numpy(forces2_np, dtype=wp.vec3f, device=device), is_global=True
    )

    wc1.add_raw_buffers_from(wc2)

    # Global at com buffer should be sum
    expected_global = forces1_np + forces2_np
    assert np.allclose(wc1.global_force_at_com_w.numpy(), expected_global, atol=1e-5)

    # After compose, output should be rotated sum
    wc1.compose_to_body_frame()
    expected_local = quat_rotate_inv_np(link_quat_np, expected_global)
    assert np.allclose(wc1.out_force_b.numpy(), expected_local, atol=1e-4)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_add_raw_buffers_from_flags(device: str):
    """Test that add_raw_buffers_from propagates flags correctly."""
    num_envs, num_bodies = 2, 2

    mock1 = create_mock_asset(num_envs, num_bodies, device)
    mock2 = create_mock_asset(num_envs, num_bodies, device)

    wc1 = WrenchComposer(mock1)
    wc2 = WrenchComposer(mock2)

    forces = wp.from_numpy(
        np.ones((num_envs, num_bodies, 3), dtype=np.float32), dtype=wp.vec3f, device=device
    )

    wc2.add_forces_and_torques(forces=forces, is_global=True)

    assert not wc1.active
    assert not wc1._has_global

    wc1.add_raw_buffers_from(wc2)

    assert wc1.active
    assert wc1._dirty
    assert wc1._has_global


# ============================================================================
# Tests for global_only flag
# ============================================================================


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_global_only_flag_true(device: str):
    """Test global_only returns True when only global forces are added."""
    num_envs, num_bodies = 2, 2
    mock_asset = create_mock_asset(num_envs, num_bodies, device)
    wc = WrenchComposer(mock_asset)

    forces = wp.from_numpy(
        np.ones((num_envs, num_bodies, 3), dtype=np.float32), dtype=wp.vec3f, device=device
    )
    wc.add_forces_and_torques(forces=forces, is_global=True)

    assert wc.global_only


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_global_only_flag_false_with_local(device: str):
    """Test global_only returns False when local forces are added."""
    num_envs, num_bodies = 2, 2
    mock_asset = create_mock_asset(num_envs, num_bodies, device)
    wc = WrenchComposer(mock_asset)

    forces = wp.from_numpy(
        np.ones((num_envs, num_bodies, 3), dtype=np.float32), dtype=wp.vec3f, device=device
    )
    wc.add_forces_and_torques(forces=forces, is_global=False)

    assert not wc.global_only


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_global_only_flag_false_with_mixed(device: str):
    """Test global_only returns False when both local and global forces are added."""
    num_envs, num_bodies = 2, 2
    mock_asset = create_mock_asset(num_envs, num_bodies, device)
    wc = WrenchComposer(mock_asset)

    forces = wp.from_numpy(
        np.ones((num_envs, num_bodies, 3), dtype=np.float32), dtype=wp.vec3f, device=device
    )
    wc.add_forces_and_torques(forces=forces, is_global=True)
    wc.add_forces_and_torques(forces=forces, is_global=False)

    assert not wc.global_only


# ============================================================================
# Tests for reset with 7 buffers
# ============================================================================


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_reset_clears_all_7_buffers(device: str):
    """Test that reset zeros all 5 input + 2 output buffers and clears flags."""
    rng = np.random.default_rng(seed=300)
    num_envs, num_bodies = 4, 3

    mock_asset = create_mock_asset(num_envs, num_bodies, device)
    wc = WrenchComposer(mock_asset)

    # Add both local and global forces
    forces_local = wp.from_numpy(
        rng.uniform(-50.0, 50.0, (num_envs, num_bodies, 3)).astype(np.float32),
        dtype=wp.vec3f, device=device,
    )
    forces_global = wp.from_numpy(
        rng.uniform(-50.0, 50.0, (num_envs, num_bodies, 3)).astype(np.float32),
        dtype=wp.vec3f, device=device,
    )
    wc.add_forces_and_torques(forces=forces_local, is_global=False)
    wc.add_forces_and_torques(forces=forces_global, is_global=True)

    # Compose so output buffers are non-zero
    wc.compose_to_body_frame()

    # Reset
    wc.reset()

    # Check all 7 buffers are zero
    zeros = np.zeros((num_envs, num_bodies, 3), dtype=np.float32)
    assert np.allclose(wc.global_force_w.numpy(), zeros)
    assert np.allclose(wc.global_torque_w.numpy(), zeros)
    assert np.allclose(wc.global_force_at_com_w.numpy(), zeros)
    assert np.allclose(wc.local_force_b.numpy(), zeros)
    assert np.allclose(wc.local_torque_b.numpy(), zeros)
    assert np.allclose(wc.out_force_b.numpy(), zeros)
    assert np.allclose(wc.out_torque_b.numpy(), zeros)

    # Check flags
    assert not wc.active
    assert not wc._dirty
    assert not wc._has_local
    assert not wc._has_global


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_reset_partial_by_index(device: str):
    """Test partial reset by environment index zeros only specified environments."""
    num_envs, num_bodies = 4, 2

    mock_asset = create_mock_asset(num_envs, num_bodies, device)
    wc = WrenchComposer(mock_asset)

    # Add forces to all envs
    forces_np = np.ones((num_envs, num_bodies, 3), dtype=np.float32) * 10.0
    forces = wp.from_numpy(forces_np, dtype=wp.vec3f, device=device)
    wc.add_forces_and_torques(forces=forces, is_global=False)
    wc.compose_to_body_frame()

    # Reset only envs 0 and 2
    env_ids = wp.array([0, 2], dtype=wp.int32, device=device)
    wc.reset(env_ids=env_ids)

    # Envs 0 and 2 should be zero, envs 1 and 3 should still have values
    local_force = wc.local_force_b.numpy()
    out_force = wc.out_force_b.numpy()
    zeros = np.zeros((num_bodies, 3), dtype=np.float32)

    assert np.allclose(local_force[0], zeros)
    assert np.allclose(local_force[2], zeros)
    assert np.allclose(out_force[0], zeros)
    assert np.allclose(out_force[2], zeros)

    # Envs 1 and 3 should still have the original values
    assert np.allclose(local_force[1], forces_np[1])
    assert np.allclose(local_force[3], forces_np[3])
    assert np.allclose(out_force[1], forces_np[1])
    assert np.allclose(out_force[3], forces_np[3])


# ============================================================================
# Tests for deprecated aliases
# ============================================================================


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_deprecated_composed_force_alias(device: str):
    """Test that composed_force is an alias for out_force_b."""
    num_envs, num_bodies = 2, 2
    mock_asset = create_mock_asset(num_envs, num_bodies, device)
    wc = WrenchComposer(mock_asset)

    forces_np = np.ones((num_envs, num_bodies, 3), dtype=np.float32) * 5.0
    forces = wp.from_numpy(forces_np, dtype=wp.vec3f, device=device)
    wc.add_forces_and_torques(forces=forces, is_global=False)
    wc.compose_to_body_frame()

    # composed_force should return the same warp array as out_force_b
    assert wc.composed_force.ptr == wc.out_force_b.ptr
    assert np.allclose(wc.composed_force.numpy(), wc.out_force_b.numpy())


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_deprecated_composed_torque_alias(device: str):
    """Test that composed_torque is an alias for out_torque_b."""
    num_envs, num_bodies = 2, 2
    mock_asset = create_mock_asset(num_envs, num_bodies, device)
    wc = WrenchComposer(mock_asset)

    torques_np = np.ones((num_envs, num_bodies, 3), dtype=np.float32) * 5.0
    torques = wp.from_numpy(torques_np, dtype=wp.vec3f, device=device)
    wc.add_forces_and_torques(torques=torques, is_global=False)
    wc.compose_to_body_frame()

    assert wc.composed_torque.ptr == wc.out_torque_b.ptr
    assert np.allclose(wc.composed_torque.numpy(), wc.out_torque_b.numpy())


# ============================================================================
# Tests for out_force_b_as_torch / out_torque_b_as_torch
# ============================================================================


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_out_force_b_as_torch(device: str):
    """Test that out_force_b_as_torch returns a torch view of the output force buffer."""
    num_envs, num_bodies = 2, 2
    mock_asset = create_mock_asset(num_envs, num_bodies, device)
    wc = WrenchComposer(mock_asset)

    forces_np = np.ones((num_envs, num_bodies, 3), dtype=np.float32) * 3.0
    forces = wp.from_numpy(forces_np, dtype=wp.vec3f, device=device)
    wc.add_forces_and_torques(forces=forces, is_global=False)
    wc.compose_to_body_frame()

    torch_result = wc.out_force_b_as_torch
    assert isinstance(torch_result, torch.Tensor)
    assert torch_result.shape == (num_envs, num_bodies, 3)
    assert np.allclose(torch_result.cpu().numpy(), forces_np, atol=1e-5)
