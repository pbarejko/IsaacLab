# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utility to bake material colours for Newton Warp rendering.

Newton's USD importer resolves diffuse colour from a fixed set of shader
input names (``diffuse_color_constant``, ``base_color``, …) and loads textures
from connected shader inputs.  When the texture file cannot be resolved at
runtime, the shape falls back to a built-in palette.

This module extracts the best-effort diffuse colour from IsaacLab / Omniverse
materials (including ``diffuse_tint`` which Newton does not read) and writes it
into ``inputs:diffuse_color_constant`` — an input Newton **does** recognise —
while disconnecting texture inputs that Newton cannot load.  The result is a
flat-shaded colour that matches the artist intent.
"""

from __future__ import annotations

import logging

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from .stage import get_current_stage

logger = logging.getLogger(__name__)

# Shader input names we check (in priority order) to extract a diffuse colour.
# This is a superset of what Newton reads — we deliberately include inputs like
# diffuse_tint that Newton ignores, so we can forward them into an input Newton
# does understand (diffuse_color_constant).
_DIFFUSE_COLOR_INPUTS: list[str] = [
    "inputs:diffuse_tint",
    "inputs:diffuse_color_constant",
    "inputs:diffuse_reflection_color",
    "inputs:diffuseColor",
    "inputs:base_color",
    "inputs:baseColor",
]

# Shader inputs that Newton recognises as colour sources (from newton/_src/usd/utils.py).
# We write the resolved colour into the first of these.
_NEWTON_COLOR_INPUT = "inputs:diffuse_color_constant"


def bake_display_colors_from_materials(
    prim_path: str,
    default_color: tuple[float, float, float] = (0.5, 0.5, 0.5),
    stage: Usd.Stage | None = None,
) -> int:
    """Prepare material colours on template gprims for Newton Warp rendering.

    For each :class:`UsdGeom.Gprim` descendant under *prim_path*:

    1. Prims with :class:`UsdPhysics.CollisionAPI` are skipped (collision-only
       geometry).
    2. The bound visual material's shader is inspected.  The best diffuse
       colour is extracted from the first recognised input (see
       ``_DIFFUSE_COLOR_INPUTS``).
    3. ``inputs:diffuse_color_constant`` on the shader is (over)written with
       the resolved colour so that Newton's USD importer picks it up.
    4. Connected texture inputs on the shader are disconnected so that Newton
       does not attempt to load unresolvable textures and bypass the colour.
    5. ``primvars:displayColor`` is also authored as a fallback for any
       consumer that reads it.
    6. If no material is bound, *default_color* is applied via
       ``primvars:displayColor`` only.

    Prims marked ``instanceable`` are temporarily made non-instanceable so
    that their descendant geometry can be authored.

    Args:
        prim_path: Root prim path whose subtree will be processed.
        default_color: Fallback RGB colour for prims with no resolvable
            material colour. Each component is in ``[0, 1]``.
        stage: USD stage to operate on.  Defaults to the current stage.

    Returns:
        Number of geometry prims that were modified.
    """
    if stage is None:
        stage = get_current_stage()

    root = stage.GetPrimAtPath(prim_path)
    if not root.IsValid():
        logger.warning("bake_display_colors_from_materials: prim '%s' is not valid – skipping.", prim_path)
        return 0

    _clear_instanceable_flags(root)

    modified = 0
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdGeom.Gprim):
            continue

        prim_path_str = prim.GetPath().pathString

        if prim.HasAPI(UsdPhysics.CollisionAPI):
            print(f"  [bakeColor] SKIP {prim_path_str} — collision mesh (PhysicsCollisionAPI)")
            continue

        color, shader_prim = _resolve_material_color(prim, stage)

        if shader_prim is not None:
            if color is None:
                color = Gf.Vec3f(*default_color)
                print(f"  [bakeColor] DEFAULT {prim_path_str} — no diffuse input, using {default_color}")
            else:
                print(
                    f"  [bakeColor] BAKED {prim_path_str}"
                    f" — color: ({color[0]:.3f}, {color[1]:.3f}, {color[2]:.3f})"
                )

            _set_shader_color(shader_prim, color)
            _disconnect_texture_inputs(shader_prim)
        else:
            if color is None:
                color = Gf.Vec3f(*default_color)
            print(f"  [bakeColor] NO-MATERIAL {prim_path_str} — displayColor only ({color[0]:.3f}, {color[1]:.3f}, {color[2]:.3f})")

        _set_display_color(prim, color)
        modified += 1

    if modified:
        logger.info(
            "bake_display_colors_from_materials: processed %d visual prim(s) under '%s'.",
            modified,
            prim_path,
        )
    return modified


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _clear_instanceable_flags(root: Usd.Prim) -> None:
    """Recursively clear ``instanceable`` on *root* and all descendants."""
    queue: list[Usd.Prim] = [root]
    while queue:
        prim = queue.pop()
        if prim.IsInstance():
            prim.SetInstanceable(False)
        queue.extend(prim.GetChildren())


def _resolve_material_color(
    prim: Usd.Prim, stage: Usd.Stage
) -> tuple[Gf.Vec3f | None, Usd.Prim | None]:
    """Extract a diffuse colour and locate the shader prim for *prim*.

    Returns:
        ``(color, shader_prim)`` — either may be ``None``.
    """
    prim_path_str = prim.GetPath().pathString
    binding_api = UsdShade.MaterialBindingAPI(prim)
    material, _ = binding_api.ComputeBoundMaterial()
    if not material:
        print(f"    [resolve] {prim_path_str} — no bound material")
        return None, None

    mat_path = material.GetPath()
    print(f"    [resolve] {prim_path_str} — bound material: {mat_path}")

    shader_prim = _get_surface_shader(material, stage)
    if shader_prim is not None:
        print(f"    [resolve]   surface shader (connected): {shader_prim.GetPath()}")
    else:
        print(f"    [resolve]   no connected surface shader")
        candidate = stage.GetPrimAtPath(mat_path.AppendChild("Shader"))
        if candidate.IsValid():
            shader_prim = candidate
            print(f"    [resolve]   fallback Shader child: {shader_prim.GetPath()}")

    if shader_prim is None:
        print(f"    [resolve]   FAILED — no shader found")
        return None, None

    print(
        f"    [resolve]   shader attrs: "
        f"{[a.GetName() for a in shader_prim.GetAttributes() if a.GetName().startswith('inputs:')]}"
    )

    for attr_name in _DIFFUSE_COLOR_INPUTS:
        attr = shader_prim.GetAttribute(attr_name)
        if attr:
            has_val = attr.HasAuthoredValue()
            val = attr.Get() if has_val else None
            print(f"    [resolve]   checking {attr_name}: exists=True, authored={has_val}, value={val}")
            if has_val and val is not None:
                return Gf.Vec3f(float(val[0]), float(val[1]), float(val[2])), shader_prim
        else:
            print(f"    [resolve]   checking {attr_name}: exists=False")

    print(f"    [resolve]   no recognised diffuse input — will use default")
    return None, shader_prim


def _get_surface_shader(material: UsdShade.Material, stage: Usd.Stage) -> Usd.Prim | None:
    """Return the prim of the surface shader connected to *material*, or ``None``."""
    surface_output = material.GetSurfaceOutput()
    if not surface_output:
        return None
    connections = surface_output.GetConnectedSources()
    if not connections or not connections[0]:
        return None
    for info in connections[0]:
        source_prim = info.source.GetPrim()
        if source_prim.IsValid():
            return source_prim
    return None


def _set_shader_color(shader_prim: Usd.Prim, color: Gf.Vec3f) -> None:
    """Write *color* into ``inputs:diffuse_color_constant`` on the shader.

    Newton's USD importer reads this input by name, so setting it ensures
    the resolved colour is picked up regardless of what other inputs exist.
    """
    attr = shader_prim.GetAttribute(_NEWTON_COLOR_INPUT)
    if not attr:
        attr = shader_prim.CreateAttribute(
            _NEWTON_COLOR_INPUT, Sdf.ValueTypeNames.Color3f, custom=False
        )
    attr.Set(color)
    print(f"  [bakeColor] SET {shader_prim.GetPath()}.{_NEWTON_COLOR_INPUT} = ({color[0]:.3f}, {color[1]:.3f}, {color[2]:.3f})")


def _disconnect_texture_inputs(shader_prim: Usd.Prim) -> None:
    """Disconnect all connected shader inputs that could reference textures.

    Newton's ``_extract_shader_properties`` iterates all shader inputs looking
    for connected sources that resolve to textures.  If *any* texture reference
    is found, ``properties["texture"]`` is set and Newton uses texture sampling
    instead of ``diffuse_color_constant``.  By disconnecting these inputs we
    ensure Newton falls through to the flat colour.
    """
    shader = UsdShade.Shader(shader_prim)
    for inp in shader.GetInputs():
        if inp.HasConnectedSource():
            inp.DisconnectSource()
            print(f"  [bakeColor] DISCONNECT {shader_prim.GetPath()}.{inp.GetFullName()}")


def _set_display_color(prim: Usd.Prim, color: Gf.Vec3f) -> None:
    """Author a constant ``primvars:displayColor`` on *prim* as a fallback."""
    gprim = UsdGeom.Gprim(prim)
    display_color = gprim.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant)
    display_color.Set([color])
