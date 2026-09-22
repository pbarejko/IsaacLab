# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for simulation-owned renderers and their rendering orchestration."""

from __future__ import annotations

import contextlib
import gc
import re
import weakref
from types import SimpleNamespace
from unittest.mock import Mock, call

import numpy as np
import pytest
import torch
import warp as wp

from isaaclab.renderers import render_context
from isaaclab.renderers.base_renderer import BaseRenderer
from isaaclab.renderers.render_context import RenderContext
from isaaclab.renderers.renderer_cfg import RendererCfg
from isaaclab.scene import InteractiveScene
from isaaclab.sensors import Camera
from isaaclab.sensors.camera.camera_data import CameraData
from isaaclab.sim import BackendCfg, SimulationContext
from isaaclab.utils.warp import ProxyArray

pytest.importorskip("isaaclab_physx")
pytest.importorskip("isaaclab_newton")

from isaaclab_newton.renderers import NewtonWarpRendererCfg
from isaaclab_physx.renderers import IsaacRtxRendererCfg

pytestmark = [pytest.mark.integration, pytest.mark.rendering]


def _renderer(cfg):
    renderer = Mock(spec=BaseRenderer)
    renderer.visual_material_writer = None
    return renderer


@pytest.fixture
def sim():
    sim = object.__new__(SimulationContext)
    sim._backend_registry = []
    sim._render_context = RenderContext(sim._backend_registry)
    return sim


def test_renderer_registry_sharing_and_early_clone_requirements(sim):
    constructor = Mock(side_effect=_renderer)
    cfg = IsaacRtxRendererCfg(class_type=constructor, cloning_contexts=("example:CloneContext",))
    renderer = sim.get_or_create_backend(cfg)

    assert sim.get_or_create_backend(cfg) is renderer
    assert sim.get_or_create_backend(cfg.replace()) is renderer
    constructor.assert_called_once_with(cfg)
    assert constructor.call_args.args[0] is cfg
    assert sim.render_context.clone_contexts == set(cfg.cloning_contexts)
    renderer.initialize.assert_not_called()

    assert sim.get_or_create_backend(cfg.replace(semantic_filter="class:robot")) is not renderer
    assert sim.get_or_create_backend(NewtonWarpRendererCfg(class_type=_renderer)) is not renderer
    sim.get_or_create_backend(BackendCfg(class_type=lambda cfg: object()))
    assert sim.render_context.renderer_types == ("isaac_rtx", "isaac_rtx", "newton_warp")


def test_renderer_initializes_once_before_or_after_physics_ready(sim):
    cfg = RendererCfg(class_type=_renderer)
    first = sim.get_or_create_backend(cfg)
    sim.get_or_create_backend(BackendCfg(class_type=lambda cfg: object()))
    sim.render_context.ensure_initialize()
    sim.render_context.ensure_initialize()
    first.initialize.assert_called_once_with()

    second_cfg = cfg.replace(renderer_type="second")
    second = sim.get_or_create_backend(second_cfg)
    second.initialize.assert_called_once_with()
    assert sim.get_or_create_backend(second_cfg) is second
    sim.render_context.ensure_initialize()
    first.initialize.assert_called_once_with()
    second.initialize.assert_called_once_with()


def test_conflicting_global_settings_are_rejected_before_construction(sim):
    constructor = Mock(side_effect=_renderer)
    cfg = IsaacRtxRendererCfg(class_type=constructor)
    renderer = sim.get_or_create_backend(cfg)
    conflicting = cfg.replace(global_settings=cfg.global_settings.replace(enable_shadows=False))

    with pytest.raises(ValueError, match="global settings differ"):
        sim.get_or_create_backend(conflicting)

    constructor.assert_called_once_with(cfg)
    assert sim.get_or_create_backend(cfg) is renderer
    assert sim.render_context.renderer_types == ("isaac_rtx",)


@pytest.mark.parametrize("has_materials", [False, True])
def test_finalized_consumers_allow_cache_hits_but_reject_new_renderers_with_materials(sim, has_materials):
    constructor = Mock(side_effect=_renderer)
    cfg = RendererCfg(class_type=constructor)
    renderer = sim.get_or_create_backend(cfg)
    if has_materials:
        sim.render_context.register_visual_material(
            SimpleNamespace(
                channels=("roughness",),
                _material_paths=("/World/Material",),
                _shader_paths=("/World/Material/Shader",),
                _input_names={"roughness": "roughness"},
                _values={"roughness": torch.zeros(1)},
                _offsets={},
            )
        )
    sim.render_context.finalize_consumers([])

    assert sim.get_or_create_backend(cfg.replace()) is renderer
    late_cfg = cfg.replace(renderer_type="late")
    if has_materials:
        with pytest.raises(RuntimeError, match="before rendering consumers are finalized"):
            sim.get_or_create_backend(late_cfg)
        constructor.assert_called_once_with(cfg)
    else:
        assert sim.get_or_create_backend(late_cfg) is not renderer
        assert constructor.call_count == 2


def test_close_backend_removes_renderer_from_orchestration(sim):
    cfg = RendererCfg(class_type=_renderer)
    renderer = sim.get_or_create_backend(cfg)
    sim.render_context.ensure_prepare_stage(None, 4)
    sim.render_context.update_scene_state(1)

    sim.close_backend(renderer)
    renderer.close.assert_called_once_with()
    assert not sim.render_context.renderer_types
    sim.render_context.update_scene_state(2)
    renderer.update_transforms.assert_called_once_with()
    renderer.update_geometries.assert_called_once_with()
    with pytest.raises(RuntimeError, match="renderer must be registered"):
        sim.render_context.ensure_prepare_stage(None, 4)

    replacement = sim.get_or_create_backend(cfg)
    assert replacement is not renderer
    sim.render_context.ensure_prepare_stage(None, 4)
    sim.render_context.update_scene_state(2)
    replacement.prepare_stage.assert_called_once_with(None, 4)
    replacement.update_transforms.assert_called_once_with()
    sim.render_context.close()
    renderer.close.assert_called_once_with()
    replacement.close.assert_not_called()


def test_prepare_stage_is_idempotent_and_checks_env_count_until_reset(sim):
    renderer = sim.get_or_create_backend(RendererCfg(class_type=_renderer))
    sim.render_context.ensure_prepare_stage(None, 4)
    sim.render_context.ensure_prepare_stage(None, 4)
    renderer.prepare_stage.assert_called_once_with(None, 4)
    with pytest.raises(RuntimeError, match="different num_envs"):
        sim.render_context.ensure_prepare_stage(None, 8)

    sim.render_context.reset_stage_prepare_flag()
    sim.render_context.ensure_prepare_stage(None, 8)
    assert renderer.prepare_stage.call_args_list == [call(None, 4), call(None, 8)]


def test_scene_state_updates_once_per_step_until_cadence_reset(sim):
    renderer = sim.get_or_create_backend(RendererCfg(class_type=_renderer))
    for step in (1, 1, 2):
        sim.render_context.update_scene_state(step)
    assert renderer.update_transforms.call_count == renderer.update_geometries.call_count == 2

    sim.render_context.reset_scene_state_cadence()
    sim.render_context.update_scene_state(2)
    assert renderer.update_transforms.call_count == renderer.update_geometries.call_count == 3


@pytest.mark.parametrize("profile", [False, True])
def test_render_into_camera_call_order_and_profile_output(sim, monkeypatch, capsys, profile):
    """Profiling preserves call order and prints the renderer benchmark's timing format."""
    monkeypatch.setattr(render_context, "_RENDER_PROFILE_ENABLED", profile)
    renderer = sim.get_or_create_backend(RendererCfg(class_type=_renderer))
    data, camera = object(), CameraData()

    sim.render_context.render_into_camera(renderer, data, camera, physics_step_count=1)
    sim.render_context.render_into_camera(renderer, data, camera, physics_step_count=1)

    assert renderer.mock_calls == [
        call.update_transforms(),
        call.update_geometries(),
        call.render([data]),
        call.read_output(data, camera),
        call.render([data]),
        call.read_output(data, camera),
    ]
    timing = rf"{re.escape(render_context.RENDER_PROFILE_SCOPE)} took [\d.]+ ms"
    assert len(re.findall(timing, capsys.readouterr().out)) == (2 if profile else 0)


@pytest.mark.parametrize("fail_writer", [False, True])
def test_context_close_only_releases_writers_and_resets_bookkeeping(sim, fail_writer):
    cfg = RendererCfg(class_type=_renderer, cloning_contexts=("example:CloneContext",))
    renderer = sim.get_or_create_backend(cfg)
    context = sim.render_context
    context.ensure_initialize()
    context.ensure_prepare_stage(None, 4)
    context.update_scene_state(1)
    writers = (Mock(), Mock())
    if fail_writer:
        writers[0].close.side_effect = RuntimeError("writer failed")
    context._visual_material_writers = writers

    if fail_writer:
        with pytest.raises(RuntimeError, match=r"1 material writer\(s\) failed to close"):
            context.close()
    else:
        context.close()
    context.close()
    for writer in writers:
        writer.close.assert_called_once_with()
    renderer.close.assert_not_called()
    assert sim.get_or_create_backend(cfg) is renderer
    assert not context.clone_contexts

    context.ensure_initialize()
    context.ensure_prepare_stage(None, 8)
    context.update_scene_state(1)
    assert renderer.initialize.call_count == renderer.prepare_stage.call_count == 2
    assert renderer.update_transforms.call_count == renderer.update_geometries.call_count == 2


class _CpuCamera(Camera):
    """Exercise camera capture timing with CPU buffers and an in-memory pose source."""

    def __init__(self, renderer, name, update_period=0.0):
        self.cfg = SimpleNamespace(update_period=update_period, update_latest_camera_pose=True)
        self._device = "cpu"
        self._num_envs = 2
        self._is_initialized = True
        self._is_visualizing = False
        self._renderer = renderer
        self._render_data = SimpleNamespace(name=name, pose=None)
        self._data = CameraData()
        self._data.create_buffers(2, "cpu")
        self._data.info = {}
        self._frame = ProxyArray(wp.zeros(2, dtype=wp.int64, device="cpu"))
        self._ALL_INDICES = wp.array([0, 1], dtype=wp.int32, device="cpu")
        self._ALL_ENV_MASK = wp.ones(2, dtype=wp.bool, device="cpu")
        self._is_outdated = wp.ones(2, dtype=wp.bool, device="cpu")
        self._timestamp = wp.zeros(2, device="cpu")
        self._timestamp_last_update = wp.zeros(2, device="cpu")
        self._data_generation = 0
        self._data_generation_last_update = -1
        self.pose = 0.0
        self._view = SimpleNamespace(count=2, xform_world_space_writer=self._pose_writer)
        self._queue_render()

    def __del__(self):
        pass

    @contextlib.contextmanager
    def _pose_writer(self):
        def set_poses(positions, orientations, indices):
            self.pose = float(positions.numpy()[0, 0])

        yield SimpleNamespace(set_poses=set_poses)

    def _update_poses(self, env_ids=None, env_mask=None, frame_op=0):
        self._render_data.pose = self.pose
        self._update_camera_state(env_ids=env_ids, env_mask=env_mask, frame_op=frame_op)


@pytest.fixture
def camera_batch_context(sim, monkeypatch):
    ctx = sim.render_context
    sim._physics_step_count = 1
    monkeypatch.setattr(SimulationContext, "_instance", sim)
    renderer = sim.get_or_create_backend(NewtonWarpRendererCfg(class_type=_renderer))
    batches = []
    renderer.render = lambda requests: batches.append([(rd.name, rd.pose) for rd in requests])
    renderer.read_output = lambda rd, data: data.info.update(pose=rd.pose)
    return ctx, renderer, batches


@pytest.mark.parametrize("lazy", [False, True])
def test_camera_updates_render_shared_batch_with_current_poses(camera_batch_context, lazy):
    """Scene updates prepare both cameras, submit once, and repeated reads reuse completed images."""
    ctx, renderer, batches = camera_batch_context
    cameras = [_CpuCamera(renderer, "wide"), _CpuCamera(renderer, "tele")]
    cameras[0].pose, cameras[1].pose = 1.0, 2.0
    scene = SimpleNamespace(
        sim=SimulationContext.instance(),
        cfg=SimpleNamespace(lazy_sensor_update=lazy),
        _sensors=dict(zip(("wide", "tele"), cameras)),
        **{
            name: {}
            for name in (
                "_articulations",
                "_cable_objects",
                "_deformable_objects",
                "_rigid_objects",
                "_rigid_object_collections",
                "_surface_grippers",
            )
        },
    )
    InteractiveScene.update(scene, 0.01)
    assert len(batches) == (0 if lazy else 1)
    assert cameras[0].data.info["pose"] == 1.0
    assert cameras[1].data.info["pose"] == 2.0
    assert batches == [[("wide", 1.0), ("tele", 2.0)]]
    for camera in cameras:
        np.testing.assert_array_equal(camera.frame.warp.numpy(), [1, 1])
        np.testing.assert_allclose(camera._timestamp_last_update.numpy(), [0.01, 0.01])
    assert not ctx._pending_cameras


def test_camera_batch_respects_period_partial_reset_and_same_step_pose(camera_batch_context):
    """A fresh peer stays cached while a due, reset, or explicitly moved camera captures again."""
    ctx, renderer, batches = camera_batch_context
    fast = _CpuCamera(renderer, "fast")
    slow = _CpuCamera(renderer, "slow", update_period=1.0)
    fast.data
    fast.update(0.1)
    slow.update(0.1, force_recompute=True)
    fast.data
    assert batches[-1] == [("fast", 0.0)]
    assert len(batches) == 2
    np.testing.assert_array_equal(slow.frame.warp.numpy(), [1, 1])
    slow.reset(env_mask=wp.array([False, True], dtype=wp.bool, device="cpu"))
    slow.data
    assert batches[-1] == [("slow", 0.0)]
    np.testing.assert_array_equal(slow.frame.warp.numpy(), [1, 1])
    np.testing.assert_allclose(slow._timestamp_last_update.numpy(), [0.0, 0.0])
    slow.set_world_poses(positions=torch.tensor([[3.0, 0.0, 0.0]]), env_ids=[1])
    assert slow.data.info["pose"] == 3.0
    assert batches[-1] == [("slow", 3.0)]
    np.testing.assert_array_equal(slow.frame.warp.numpy(), [1, 2])
    slow.update(1.0)
    slow.data
    np.testing.assert_array_equal(slow.frame.warp.numpy(), [2, 3])


@pytest.mark.parametrize("failure", ["render", "read"])
def test_camera_batch_failure_retries_without_completing_captures(camera_batch_context, failure):
    """Rendering or readback failure leaves all camera counters/timestamps pending until retry."""
    ctx, renderer, batches = camera_batch_context
    cameras = [_CpuCamera(renderer, "wide"), _CpuCamera(renderer, "tele")]
    for camera in cameras:
        camera.update(0.1)
    original = renderer.render if failure == "render" else renderer.read_output

    def fail(*args):
        if failure == "render" or args[0].name == "tele":
            raise RuntimeError("capture failed")
        original(*args)

    setattr(renderer, "render" if failure == "render" else "read_output", fail)
    with pytest.raises(RuntimeError, match="capture failed"):
        cameras[0].data
    for camera in cameras:
        np.testing.assert_array_equal(camera.frame.warp.numpy(), [0, 0])
        np.testing.assert_array_equal(camera._timestamp_last_update.numpy(), [0.0, 0.0])
    setattr(renderer, "render" if failure == "render" else "read_output", original)
    cameras[1].data
    for camera in cameras:
        np.testing.assert_array_equal(camera.frame.warp.numpy(), [1, 1])
        np.testing.assert_allclose(camera._timestamp_last_update.numpy(), [0.1, 0.1])
    assert not ctx._pending_cameras


def test_camera_batch_groups_backends_and_releases_queued_cameras(camera_batch_context):
    """Reading one backend leaves another pending, and deleted/removed cameras cannot render."""
    ctx, renderer, batches = camera_batch_context
    other = SimulationContext.instance().get_or_create_backend(IsaacRtxRendererCfg(class_type=_renderer))
    other_batches = []
    other.render = lambda requests: other_batches.append([rd.name for rd in requests])
    other.read_output = lambda rd, data: None
    first = _CpuCamera(renderer, "first")
    second = _CpuCamera(other, "second")
    expired = _CpuCamera(renderer, "expired")
    reference = weakref.ref(expired)
    del expired
    gc.collect()
    assert reference() is None
    first.data
    assert batches == [[("first", 0.0)]]
    assert not other_batches
    ctx.remove_camera(second)
    ctx.render_pending_cameras(1)
    assert not other_batches


def test_camera_data_read_completes_inside_deferred_scope(camera_batch_context):
    """Explicit reads never return stale data merely because eager updates are being collected."""
    ctx, renderer, batches = camera_batch_context
    camera = _CpuCamera(renderer, "camera")
    with ctx.defer_camera_renders(1):
        camera.update(0.1, force_recompute=True)
        assert not batches
        assert camera.data.info["pose"] == 0.0
        assert len(batches) == 1
    assert len(batches) == 1
