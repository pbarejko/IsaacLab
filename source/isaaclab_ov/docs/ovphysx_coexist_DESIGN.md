# ovphysx + ovrtx coexistence — IsaacLab integration design notes

Tracking ticket: OMPE-88037 — `presets=ovphysx,ovrtx_renderer,rgb`.

This branch (`erarnold/ovphysx-ovrtx-coexist`, off `pbarejko/ovphysx-ovrtx`)
contains only the changes that are clearly correct without first answering
the open questions to the kit/sdk and ovstage agents listed in
`~/tmp/OMPE-88037-ovphysx-ovrtx-isaaclab.md`. The harder pieces are
captured here as design notes.

## What this branch does land

**Renderer-side teardown before physics-side teardown.**
`SimulationContext.clear_instance()` now calls
`self._render_context.cleanup()` before `physics_manager.close()`. The
new `RenderContext.cleanup()` method walks the registered renderer
backends and invokes `BaseRenderer.cleanup(None)` on each, then drops
its registration list. Per-camera `Camera.__del__` cleanup remains in
place as a safety net but becomes a no-op once the central cleanup has
run.

Why: the kit/sdk and ovstage samples (see
`~/repos/kit/sdks/source/ovphysx_ovrtx_integration_example/README.md`
"Cleanup order matches C++ sample" and
`~/repos/ovstage/src/examples/render_physx.cpp:785`) are explicit that
ovrtx must release before ovphysx tears down their shared Carbonite.
Today IsaacLab releases physics first and lets cameras GC last; that
inversion crashes the second native teardown. The fix is small, has no
behavior change for non-ovrtx backends, and removes the sequencing
dependency on Python GC.

## What this branch defers

### A. Hoist OVRTXRenderer construction before OvPhysxManager._warmup_and_load

**Problem.** Today `ovrtx.Renderer(config)` is constructed lazily inside
`OVRTXRenderer.initialize(spec)` (in `source/isaaclab_ov/.../ovrtx_renderer.py`),
which is called from `OVRTXRenderer.create_render_data()`, which is
called from `Camera._initialize_impl()` (in
`source/isaaclab/isaaclab/sensors/camera/camera.py:423`), which fires
on the `PHYSICS_READY` event dispatched at the end of
`OvPhysxManager.reset() → _warmup_and_load()`. By that point ovphysx has
already constructed `ovphysx.PhysX(device=...)` and claimed Carbonite.

Per the working samples (`render_physx.cpp:200-214`,
`ovlibs_sample.py:210-213`) ovrtx must claim Carbonite first, before
ovphysx exists. Two paths:

1. **Wheel side:** ovphysx 0.4.3 already flips the
   `OVPHYSX_COEXIST_DIAGNOSTICS` default (commit `0917441ef9`) so that
   ovphysx auto-detects another Carbonite owner and proceeds in
   coexistence mode. **Open question for the kit/sdk agent (Q1 in the
   triage doc):** does this default flip make ovphysx-first
   work-as-coexistence-tenant, or only suppress the env-var nag?
2. **IsaacLab side:** add a `BaseRenderer.early_init(stage)` hook that
   the renderer uses to construct its native object before
   `prepare_stage`. `RenderContext` exposes `early_init_all(stage)`
   that walks registered backends. Add a hook on the physics manager
   (or `SimulationContext.reset`) that calls it before
   `physics_manager.reset()` runs `_warmup_and_load`. Backends that
   don't need it (Newton, Isaac RTX) get a default no-op.

The catch with (2): Camera sensors only register their renderer cfg
during `_initialize_impl` (after PHYSICS_READY), so at the early hook
nothing is registered yet. We need to pre-walk the scene cfg for
`CameraCfg.renderer_cfg` entries during `InteractiveScene` build (or
`SimulationContext.__init__`) and register them up front.

**Don't implement until Q1 is answered.** If 0.4.3 does relax the order
constraint, (1) is enough and IsaacLab can keep the lazy construction
path.

### B. Wire OVRTXRenderer._setup_object_bindings to read OvPhysx pose bindings

**Problem.** `OVRTXRenderer._setup_object_bindings` (line 307) and
`OVRTXRenderer.update_transforms` (line 409) call
`SimulationContext.instance().initialize_scene_data_provider().get_newton_model()`
and `.get_newton_state().body_q`. With OvPhysx these return `None`, so
object transforms never reach OVRTX (camera transforms still work via
the omni:xform binding written from `update_camera`).

**Two ways forward:**

1. **Add an `OvPhysxSceneDataProvider`** that mirrors the Newton
   provider's interface (`get_newton_model()` returns a path/index list,
   `get_newton_state()` returns a body-q tensor). The provider wraps
   ovphysx tensor bindings — pose binding → body_q tensor in
   `wp.transformf` layout. Keep the same renderer code by aliasing
   "newton state" to "physics state" at the provider level.
2. **Add a `BaseSceneDataProvider.get_body_transforms()`** abstraction
   so OVRTXRenderer no longer reaches for Newton-specific accessors,
   then implement it on both Newton and a new OvPhysx provider.

(2) is cleaner but a bigger renderer-side change. (1) is fewer LOC and
keeps the scope focused on unblocking the bug. **Open question for the
ovstage agent (Q3 in the triage doc):** if there's a third option
(IsaacLab pulls in `ovstage` as a third dependency and uses the
ovstage→ovrtx attribute-write path the working samples use), is that
the recommended pattern or a heavier one?

### C. Migrate physx.clone() to attach_stage + stage.clone_subtree

**Problem.** `OvPhysxManager._warmup_and_load` calls
`cls._physx.clone(source, targets, transforms)` (line 278) for
replicate-physics. ovphysx 0.4.3 removes the public `physx.clone()` API
(see ovphysx changelog under 0.4 "Breaking changes") in favor of
`physx.attach_stage(stage) → stage.clone_subtree(source, [targets])`
followed by per-target `xformOp:translate` writes inside the same
`begin_frame`/`end_frame`. The replicate plugin is unchanged; only the
entry point moved.

**Migration sketch:**

```python
# Before (0.4.2 and older)
op_idx = cls._physx.clone(source, targets, transforms)
cls._physx.wait_op(op_idx)

# After (0.4.3+)
import ovstage  # new dependency
stage = ovstage.Stage()  # or get from sim context if hoisted
ovstage_op = stage.begin_frame()
stage.clone_subtree(source, targets)
for target_path, (x, y, z) in zip(targets, parent_positions):
    stage.write_attribute(ovstage_op, [target_path], "xformOp:translate", pack_vec3(x, y, z))
stage.end_frame(ovstage_op)
# OvstageBridge ingests this on next physx.step()
```

**Why deferred:**

1. Pulling `ovstage` into IsaacLab as a third dependency is a bigger
   ask than this branch should swallow without sign-off — it adds
   another wheel pin, plugin path, and Carbonite tenant.
2. The migration is wheel-version-dependent. The cleanest implementation
   is a feature gate: try `physx.clone()` first, fall back to the
   ovstage path if `AttributeError`. That keeps the 0.4.2 path working
   while landing the 0.4.3 path. But the gate logic only makes sense
   once we've actually exercised the 0.4.3 API end-to-end, which needs
   a built 0.4.3 wheel installed in IsaacLab.

**Action when 0.4.3 wheel is in hand:** add the version gate and
ovstage import, run the cartpole repro to confirm clones land.

### D. OVRTX renderer init order with respect to USD export

**Problem (potential — needs verification).** `OvPhysxManager._warmup_and_load`
exports the USD stage to a temp file before constructing ovphysx; that
file is what ovphysx ingests. `OVRTXRenderer.prepare_stage` separately
exports the stage to `/tmp/stage_before_ovrtx.usda` with cameras
injected. Two distinct files. The samples have both libs `open_usd`
the same on-disk file.

If the open question Q1 (init order in 0.4.3) lands as "ovrtx must
still be first", the USD file ovrtx opens has not yet been augmented
with the OvPhysx-required schemas (which `_configure_physx_scene_prim`
writes onto the in-memory stage right before export). One USD file
that both backends open is the cleanest fix; needs a small refactor of
the export path so OvPhysxManager and OVRTXRenderer share the export.

**Defer until Q1 + the actual repro on 0.4.3 says whether this matters.**

## Open questions blocking the deferred items

(Same numbering as `~/tmp/OMPE-88037-ovphysx-ovrtx-isaaclab.md` §3-§4.)

- **Q1 (kit/sdk):** Does ovphysx 0.4.3 *enforce* coexistence-mode
  regardless of init order, or only suppress the env-var nag? Decides
  whether item A is needed at all.
- **Q2 (kit/sdk):** Validated ovrtx pip build for ovphysx 0.4.x.
  Decides item D / pin update.
- **Q3 (ovstage):** Slimmest "ovphysx pose tensor → ovrtx CUDA tensor"
  recipe. Decides item B's choice between an OvPhysx scene-data
  provider vs pulling ovstage in.
