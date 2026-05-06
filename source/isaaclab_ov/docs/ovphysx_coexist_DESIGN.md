# ovphysx + ovrtx coexistence — IsaacLab integration design notes

Tracking ticket: OMPE-88037 — `presets=ovphysx,ovrtx_renderer,rgb`.

Branch: `erarnold/ovphysx-ovrtx-coexist`, off `pbarejko/ovphysx-ovrtx`.

Validated against ovphysx 0.4.3 (`dev/erarnold/ovstage-attach`,
commit `c728f7e740`) and ovrtx 0.3.0.307110 on Linux x86_64. The
cartpole repro
(`Isaac-Cartpole-Camera-Presets-Direct-v0`, 32 envs, 100×100 RGB tiled
camera, 2 epochs) runs end-to-end at ~670 fps in 16-17 seconds.

This doc tracks the four pieces the integration needed and links the
specific commits that landed each. Three questions to the kit/sdk and
ovstage teams informed the design:

1. **kit/sdk:** "does ovphysx 0.4.3's coexistence default flip relax
   the init-order requirement, or only suppress the env-var nag?"
   → **Only suppresses the nag.** ovrtx is the side that can't go
   second; it does no checks at construction time. Hence item 3 below.
2. **kit/sdk:** "validated ovrtx pip build for ovphysx 0.4.x?"
   → `0.3.0.307110` (kit demos) or `0.3.0.304843` (ovstage demo) —
   either works against the same shared deps.
3. **ovstage:** "slim ovphysx-pose → ovrtx-binding recipe without
   pulling ovstage in?" → "Don't pull ovstage in for the steady-state
   pose-forward path." Recipe: ovphysx pose binding → CUDA quat→mat4
   kernel → ovrtx `write_attribute(omni:xform, XFORM_MAT4x4)`. Used
   unmodified for item 4 below.

## 1. Renderer teardown before physics teardown — landed

**What.** `SimulationContext.clear_instance()` calls
`self._render_context.cleanup()` before `physics_manager.close()`. New
`RenderContext.cleanup()` walks registered backends, invokes
`BaseRenderer.cleanup(None)` on each, drops the registration list.
Per-camera `Camera.__del__` cleanup stays as a safety net but is a
no-op after central cleanup.

**Why.** ovrtx and ovphysx share a Carbonite framework; ovrtx-side
native objects must release before ovphysx tears down its Carbonite,
otherwise the second teardown crashes on freed plugin state. IsaacLab
released physics first and let cameras GC last, which inverted the
required order.

**Commit:** `a7ecbd0de18` "sim: tear down camera renderers before physics manager close".

## 2. ovphysx 0.4.3 clone-API removal — landed

**What.** ovphysx 0.4.3 removed the public `physx.clone()` (replaced
by `attach_stage` + `ovstage_clone_subtree`). Until that ovstage
bridge is wired into IsaacLab, `InteractiveScene` routes ovphysx
through USD-side cloning: `clone_usd=True`, `clone_physics=False`,
`physics_clone_fn=None`. Every env's physics prims live in the USD
that ovphysx ingests; no physics-runtime clone is needed.
`OvPhysxManager._warmup_and_load` now raises a clear error if a
pre-0.4.3 wheel and the legacy `clone_physics=True` path coincide,
rather than letting `AttributeError` bubble through `Camera`
initialization.

**Why.** Previously the ovphysx branch routed `clone_usd=False` so
that `physx.clone()` could populate env_1..N in the physics runtime;
the 0.4.3 wheel removed that entry point.

**Commit:** `f018386acda` "sim: ovphysx 0.4.3 + ovrtx coexistence — clone migration + renderer init hoist".

**Future work.** Pulling `ovstage` into IsaacLab as a third dependency
+ wiring `physx.attach_stage(stage)` + `stage.clone_subtree(...)`
restores physics-runtime cloning. Per the ovstage team's guidance,
this is *not* recommended for the steady-state pose-forward path
(item 4 below already does that without ovstage); it would only matter
if someone hits a perf or memory limit on USD-side cloning at large
`num_envs`.

## 3. OVRTXRenderer init order — landed

**What.** New `BaseRenderer.early_init()` hook (default no-op).
`OVRTXRenderer` overrides it to construct the C++ `Renderer(config)`
up front. `OVRTXRenderer.initialize(spec)` skips renderer construction
when `early_init` already ran. New `RenderContext.early_init_all()`
walks registered backends. New `InteractiveScene.early_init_renderers()`
pre-walks `Camera` sensors to register their `renderer_cfg` with the
context so `early_init` has something to act on. `DirectRLEnv._init_sim`
calls `scene.early_init_renderers()` between `_setup_scene` (which is
where the cartpole task adds its tiled camera) and `sim.reset()`.

**Why.** ovrtx does no coexistence checks at construction time and
SIGSEGVs inside `createRTXRenderer` if ovphysx (or any other Carbonite
owner) has already loaded its plugins. IsaacLab constructed OVRTX
lazily in `Camera._initialize_impl` (which fires on PHYSICS_READY,
*after* `OvPhysxManager._warmup_and_load`), inverting the required
order. kit/sdk Q1 confirmed: even with ovphysx 0.4.3's coexistence
default flip, ovrtx is the side that can't go second; the wheel-side
flip only suppresses the env-var nag, it doesn't make ovrtx tolerant
of being constructed second.

**Commit:** `f018386acda` (same as item 2; the renderer hoist and the
clone migration share the commit).

**Caveat.** The `early_init_renderers()` hook is currently called
explicitly by `DirectRLEnv._init_sim`. Other env types (e.g.
manager-based envs) that build cameras after `InteractiveScene.__init__`
need to call it themselves between sensor construction and
`sim.reset()`, otherwise the lazy renderer construction comes back and
the SIGSEGV returns under ovphysx. Worth a follow-up that hooks it
from a more central place (likely `SimulationContext.reset` walking the
scene cfg, but that needs scene-cfg discovery from the sim context,
which doesn't exist yet).

## 4. OvPhysx pose binding → OVRTX object transforms — landed

**What.** `OVRTXRenderer._setup_object_bindings` now falls through to
an OvPhysx path when the Newton scene-data provider returns no model.
The fallback walks the USD stage for `PhysicsRigidBodyAPI` prims under
`/World/envs/env_*` (excluding the camera and any ground plane),
creates a single ovphysx `RIGID_BODY_POSE` tensor binding for them,
and binds the same flat list to ovrtx `omni:xform`
(`PrimMode.EXISTING_ONLY`, `omni:resetXformStack=True` so writes are
treated as world transforms — same pattern as the camera binding).

`update_transforms` now branches on which fallback fired. The OvPhysx
path: read the ovphysx pose tensor on GPU into a pre-allocated
`[N, 7]` `wp.array`, run `sync_ovphysx_pose_to_mat44d_kernel` to
construct ovrtx-format `mat44d` rows (column-major rotation,
translation in row 3 — matches `create_camera_transforms_kernel`),
then `wp.copy` into the ovrtx attribute mapping. Zero host roundtrip,
single ovrtx write per frame.

**Why.** Without it, the renderer ran clean on ovphysx but every
object in the rendered frame was static — the renderer had no source
of per-frame pose updates because `get_newton_model()` returned `None`
under ovphysx and the Newton path was the only one wired.

**Recipe credit.** kit/sdk team — "don't pull ovstage in for the
steady-state pose-forward path" recommendation, the `omni:xform`
LOCAL/WORLD note, and the `PrimMode` caveat. We sidestepped the
parent-strip math by using `omni:resetXformStack=True` (matches what
the camera binding already did; the kit/sdk sample uses the same
trick). `PrimMode.EXISTING_ONLY` worked here; the kit/sdk note about
`EXISTING_ONLY` skipping the first write silently when the bucket has
no `omni:xform` column didn't bite for us because the cloned env
subtrees inherit the column from env_0.

**Commit:** `9039f0adb1a` "ovrtx: forward OvPhysx rigid-body poses to OVRTX object bindings".

**Caveats / future work.**

- Articulation-link poses for cartpole are read via `RIGID_BODY_POSE`
  rather than `ARTICULATION_LINK_POSE`. ovphysx 0.4.3 accepts this for
  the cartpole layout (32 envs × 3 bodies = 96 prims bound), but if a
  task hits a layout where `RIGID_BODY_POSE` rejects an articulation
  link, the fallback should switch to a per-articulation
  `ARTICULATION_LINK_POSE` binding, walk the USD for `PhysicsArticulationRootAPI`
  prims to get the articulation list, and use `binding.body_names` to
  map back to USD prim paths. Not implemented because it isn't needed
  yet.
- The visual smoke is by absence of update warnings + non-trivial
  RGB output, not by frame-by-frame inspection. A short `--video`
  capture across N steps would close that gap.
- Fixed-base assets (e.g. ground plane) are filtered by name. A
  task that names its ground plane something other than
  `GroundPlane`/`ground_plane` would slip through and try to bind a
  static body, which ovphysx may reject. Move the filter to a
  schema check (`PhysicsRigidBodyAPI.GetRigidBodyEnabledAttr`) if
  this becomes an issue.

## Cross-cutting notes

**ovrtx wheel pin.** kit/sdk's CMake comment about needing
"Fabric IStageReaderWriter v0.16+" remains relevant: the run prints
repeated `Warning: Possible version incompatibility. Attempting to
load omni::fabric::IStageReaderWriter with version v0.16 against v0.15`.
Non-fatal, but the kit/sdk team probably wants to know which dep is
still on v0.15 in the 0.4.3 + 0.3.0.307110 combo.

**`os._exit(0)` shutdown hack.** `OvPhysxManager._construct_physx`
still installs an `atexit` handler that calls `os._exit(0)` to
sidestep the dual-Carbonite static-destructor race. Keep until
ovphysx ships a namespace-isolated Carbonite (the existing HACK
comment in `ovphysx_manager.py` covers the rationale).

**Kit visualizer suspension.** Not touched in this branch. The
`SimulationContext` already has the kit-visualizer-aware paths
gated on `has_kit()`, so kitless ovphysx + ovrtx runs go through
the right branches without changes.
