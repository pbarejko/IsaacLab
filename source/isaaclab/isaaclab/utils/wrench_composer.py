# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import torch
import warp as wp

from isaaclab.utils.warp.kernels import (
    add_forces_to_dual_buffers_index,
    add_forces_to_dual_buffers_mask,
    add_raw_wrench_buffers,
    compose_wrench_to_body_frame,
    reset_wrench_composer_index,
    reset_wrench_composer_mask,
    set_forces_to_dual_buffers_index,
    set_forces_to_dual_buffers_mask,
)

if TYPE_CHECKING:
    from isaaclab.assets import BaseArticulation, BaseRigidObject, BaseRigidObjectCollection


class WrenchComposer:
    """Dual-buffer wrench composer.

    This class accumulates forces and torques from multiple sources into five separate input buffers
    (split by frame and application mode), then composes them into a single body-frame result on demand.

    The five input buffers are:

    - ``_global_force_w``: Positional global forces (world frame). When a position is provided,
      the resulting positional torque is computed during composition.
    - ``_global_torque_w``: Global torques stored about the world origin.
    - ``_global_force_at_com_w``: Global forces applied at the center of mass (no positional torque).
    - ``_local_force_b``: Body-frame forces.
    - ``_local_torque_b``: Body-frame torques.

    The two output buffers are:

    - ``_out_force_b``: Composed force in the body (link) frame.
    - ``_out_torque_b``: Composed torque in the body (link) frame.

    Call :meth:`compose_to_body_frame` to fetch the current link poses from the asset and run the
    composition kernel that transforms all global contributions into the body frame and sums
    everything together.
    """

    def __init__(self, asset: BaseArticulation | BaseRigidObject | BaseRigidObjectCollection) -> None:
        """Initialize the wrench composer.

        Args:
            asset: The asset whose bodies receive the composed wrenches.

        Raises:
            ValueError: If the asset type does not expose ``num_bodies`` or ``body_link_pose_w``.
        """
        self.num_envs = asset.num_instances
        # Avoid isinstance to prevent circular import issues; use attribute presence instead.
        if hasattr(asset, "num_bodies"):
            self.num_bodies = asset.num_bodies
        else:
            raise ValueError(f"Unsupported asset type: {asset.__class__.__name__}")
        self.device = asset.device
        self._asset = asset

        # -- Tracking flags --
        self._active: bool = False
        self._dirty: bool = False
        self._has_local: bool = False
        self._has_global: bool = False

        # Avoid isinstance here due to potential circular import issues; check by attribute presence instead.
        if hasattr(self._asset.data, "body_link_pose_w"):
            self._get_link_pose_fn = lambda a=self._asset: a.data.body_link_pose_w
        else:
            raise ValueError(f"Unsupported asset type: {self._asset.__class__.__name__}")

        shape = (self.num_envs, self.num_bodies)

        # -- 5 input buffers --
        self._global_force_w = wp.zeros(shape, dtype=wp.vec3f, device=self.device)
        self._global_torque_w = wp.zeros(shape, dtype=wp.vec3f, device=self.device)
        self._global_force_at_com_w = wp.zeros(shape, dtype=wp.vec3f, device=self.device)
        self._local_force_b = wp.zeros(shape, dtype=wp.vec3f, device=self.device)
        self._local_torque_b = wp.zeros(shape, dtype=wp.vec3f, device=self.device)

        # -- 2 output buffers --
        self._out_force_b = wp.zeros(shape, dtype=wp.vec3f, device=self.device)
        self._out_torque_b = wp.zeros(shape, dtype=wp.vec3f, device=self.device)

        # -- Pre-allocated index / mask helpers --
        self._ALL_ENV_INDICES = wp.array(np.arange(self.num_envs, dtype=np.int32), dtype=wp.int32, device=self.device)
        self._ALL_BODY_INDICES = wp.array(
            np.arange(self.num_bodies, dtype=np.int32), dtype=wp.int32, device=self.device
        )
        self._ALL_ENV_MASK = wp.ones((self.num_envs,), dtype=wp.bool, device=self.device)
        self._ALL_BODY_MASK = wp.ones((self.num_bodies,), dtype=wp.bool, device=self.device)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def active(self) -> bool:
        """Whether any forces or torques have been written since the last full reset."""
        return self._active

    @property
    def global_only(self) -> bool:
        """True when only the global input buffers have been written (no local contributions)."""
        return self._has_global and not self._has_local

    # -- Input buffer accessors (read-only) --

    @property
    def global_force_w(self) -> wp.array:
        """Positional global forces buffer. Shape ``(num_envs, num_bodies)``, dtype ``wp.vec3f``."""
        return self._global_force_w

    @property
    def global_torque_w(self) -> wp.array:
        """Global torques buffer (about world origin). Shape ``(num_envs, num_bodies)``, dtype ``wp.vec3f``."""
        return self._global_torque_w

    @property
    def global_force_at_com_w(self) -> wp.array:
        """Global forces at CoM buffer (no positional torque). Shape ``(num_envs, num_bodies)``, dtype ``wp.vec3f``."""
        return self._global_force_at_com_w

    @property
    def local_force_b(self) -> wp.array:
        """Body-frame forces buffer. Shape ``(num_envs, num_bodies)``, dtype ``wp.vec3f``."""
        return self._local_force_b

    @property
    def local_torque_b(self) -> wp.array:
        """Body-frame torques buffer. Shape ``(num_envs, num_bodies)``, dtype ``wp.vec3f``."""
        return self._local_torque_b

    # -- Output buffer accessors --

    @property
    def out_force_b(self) -> wp.array:
        """Composed force in the body (link) frame. Shape ``(num_envs, num_bodies)``, dtype ``wp.vec3f``.

        .. warning::
            If the composer is dirty (inputs were modified since the last composition), this property
            will trigger :meth:`compose_to_body_frame` automatically and emit a warning.
        """
        if self._dirty:
            warnings.warn(
                "Accessing out_force_b while the composer is dirty. Calling compose_to_body_frame() automatically."
                " Consider calling compose_to_body_frame() explicitly before reading outputs.",
                UserWarning,
                stacklevel=2,
            )
            self.compose_to_body_frame()
        return self._out_force_b

    @property
    def out_torque_b(self) -> wp.array:
        """Composed torque in the body (link) frame. Shape ``(num_envs, num_bodies)``, dtype ``wp.vec3f``.

        .. warning::
            If the composer is dirty (inputs were modified since the last composition), this property
            will trigger :meth:`compose_to_body_frame` automatically and emit a warning.
        """
        if self._dirty:
            warnings.warn(
                "Accessing out_torque_b while the composer is dirty. Calling compose_to_body_frame() automatically."
                " Consider calling compose_to_body_frame() explicitly before reading outputs.",
                UserWarning,
                stacklevel=2,
            )
            self.compose_to_body_frame()
        return self._out_torque_b

    @property
    def out_force_b_as_torch(self) -> torch.Tensor:
        """Composed force in body frame as a :class:`torch.Tensor`.

        Shape ``(num_envs, num_bodies, 3)``, dtype ``torch.float32``.
        """
        return wp.to_torch(self.out_force_b)

    @property
    def out_torque_b_as_torch(self) -> torch.Tensor:
        """Composed torque in body frame as a :class:`torch.Tensor`.

        Shape ``(num_envs, num_bodies, 3)``, dtype ``torch.float32``.
        """
        return wp.to_torch(self.out_torque_b)

    # -- Legacy composed_force / composed_torque properties for backward compat --

    @property
    def composed_force(self) -> wp.array:
        """Composed force at the body's link frame.

        .. deprecated::
            Use :attr:`out_force_b` instead. This property delegates to the output buffer.
        """
        return self.out_force_b

    @property
    def composed_torque(self) -> wp.array:
        """Composed torque at the body's link frame.

        .. deprecated::
            Use :attr:`out_torque_b` instead. This property delegates to the output buffer.
        """
        return self.out_torque_b

    # ------------------------------------------------------------------
    # Index-based methods
    # ------------------------------------------------------------------

    def add_forces_and_torques_index(
        self,
        forces: wp.array | torch.Tensor | None = None,
        torques: wp.array | torch.Tensor | None = None,
        positions: wp.array | torch.Tensor | None = None,
        body_ids: wp.array | torch.Tensor | list | None = None,
        env_ids: wp.array | torch.Tensor | list | None = None,
        is_global: bool = False,
    ):
        """Add forces and torques into the dual input buffers using index selection.

        Forces, torques, and positions are accumulated (added) into the appropriate input buffers
        depending on the ``is_global`` flag. The composer is marked dirty so that
        :meth:`compose_to_body_frame` must be called before reading outputs.

        Args:
            forces: Forces to add. Shape ``(len(env_ids), len(body_ids), 3)``. Defaults to None.
            torques: Torques to add. Shape ``(len(env_ids), len(body_ids), 3)``. Defaults to None.
            positions: Application positions for forces. Shape ``(len(env_ids), len(body_ids), 3)``.
                When ``is_global=True``, these are world-frame positions; the kernel computes
                positional torques from the lever arm. When ``is_global=False``, these are
                body-frame offsets. Defaults to None (forces applied at the link origin / CoM).
            body_ids: Body indices. Shape ``(num_selected_bodies,)``. Defaults to None (all bodies).
            env_ids: Environment indices. Shape ``(num_selected_envs,)``. Defaults to None (all envs).
            is_global: If True, forces/torques are in the world frame. Defaults to False.
        """
        # Resolve indices
        env_ids = self._resolve_env_ids(env_ids)
        body_ids = self._resolve_body_ids(body_ids)

        if forces is None and torques is None:
            warnings.warn(
                "No forces or torques provided. No force will be added.",
                UserWarning,
                stacklevel=2,
            )
            return

        self._active = True
        self._dirty = True
        if is_global:
            self._has_global = True
        else:
            self._has_local = True

        wp.launch(
            add_forces_to_dual_buffers_index,
            dim=(env_ids.shape[0], body_ids.shape[0]),
            inputs=[
                env_ids,
                body_ids,
                forces,
                torques,
                positions,
                is_global,
            ],
            outputs=[
                self._global_force_w,
                self._global_torque_w,
                self._global_force_at_com_w,
                self._local_force_b,
                self._local_torque_b,
            ],
            device=self.device,
        )

    def set_forces_and_torques_index(
        self,
        forces: wp.array | torch.Tensor | None = None,
        torques: wp.array | torch.Tensor | None = None,
        positions: wp.array | torch.Tensor | None = None,
        body_ids: wp.array | torch.Tensor | list | None = None,
        env_ids: wp.array | torch.Tensor | list | None = None,
        is_global: bool = False,
    ):
        """Set (overwrite) forces and torques in the dual input buffers using index selection.

        All five input buffers are cleared first, then the provided forces/torques are written.

        Args:
            forces: Forces to set. Shape ``(len(env_ids), len(body_ids), 3)``. Defaults to None.
            torques: Torques to set. Shape ``(len(env_ids), len(body_ids), 3)``. Defaults to None.
            positions: Application positions for forces. Defaults to None.
            body_ids: Body indices. Defaults to None (all bodies).
            env_ids: Environment indices. Defaults to None (all envs).
            is_global: If True, forces/torques are in the world frame. Defaults to False.
        """
        # Resolve indices
        env_ids = self._resolve_env_ids(env_ids)
        body_ids = self._resolve_body_ids(body_ids)

        if forces is None and torques is None:
            warnings.warn(
                "No forces or torques provided. No force will be added.",
                UserWarning,
                stacklevel=2,
            )
            return

        # Clear ALL 5 input buffers before setting
        self._global_force_w.zero_()
        self._global_torque_w.zero_()
        self._global_force_at_com_w.zero_()
        self._local_force_b.zero_()
        self._local_torque_b.zero_()

        self._active = True
        self._dirty = True
        if is_global:
            self._has_global = True
            self._has_local = False
        else:
            self._has_local = True
            self._has_global = False

        wp.launch(
            set_forces_to_dual_buffers_index,
            dim=(env_ids.shape[0], body_ids.shape[0]),
            inputs=[
                env_ids,
                body_ids,
                forces,
                torques,
                positions,
                is_global,
            ],
            outputs=[
                self._global_force_w,
                self._global_torque_w,
                self._global_force_at_com_w,
                self._local_force_b,
                self._local_torque_b,
            ],
            device=self.device,
        )

    # ------------------------------------------------------------------
    # Mask-based methods
    # ------------------------------------------------------------------

    def add_forces_and_torques_mask(
        self,
        forces: wp.array | torch.Tensor | None = None,
        torques: wp.array | torch.Tensor | None = None,
        positions: wp.array | torch.Tensor | None = None,
        body_mask: wp.array | torch.Tensor | None = None,
        env_mask: wp.array | torch.Tensor | None = None,
        is_global: bool = False,
    ):
        """Add forces and torques into the dual input buffers using mask selection.

        Only entries where the corresponding environment and body masks are True are processed.
        Forces and torques are expected to be full-sized ``(num_envs, num_bodies, 3)``.

        Args:
            forces: Forces to add. Shape ``(num_envs, num_bodies, 3)``. Defaults to None.
            torques: Torques to add. Shape ``(num_envs, num_bodies, 3)``. Defaults to None.
            positions: Application positions. Shape ``(num_envs, num_bodies, 3)``. Defaults to None.
            body_mask: Boolean body mask. Shape ``(num_bodies,)``. Defaults to None (all bodies).
            env_mask: Boolean environment mask. Shape ``(num_envs,)``. Defaults to None (all envs).
            is_global: If True, forces/torques are in the world frame. Defaults to False.
        """
        if env_mask is None:
            env_mask = self._ALL_ENV_MASK
        if body_mask is None:
            body_mask = self._ALL_BODY_MASK

        if forces is None and torques is None:
            warnings.warn(
                "No forces or torques provided. No force will be added.",
                UserWarning,
                stacklevel=2,
            )
            return

        self._active = True
        self._dirty = True
        if is_global:
            self._has_global = True
        else:
            self._has_local = True

        wp.launch(
            add_forces_to_dual_buffers_mask,
            dim=(self.num_envs, self.num_bodies),
            inputs=[
                env_mask,
                body_mask,
                forces,
                torques,
                positions,
                is_global,
            ],
            outputs=[
                self._global_force_w,
                self._global_torque_w,
                self._global_force_at_com_w,
                self._local_force_b,
                self._local_torque_b,
            ],
            device=self.device,
        )

    def set_forces_and_torques_mask(
        self,
        forces: wp.array | torch.Tensor | None = None,
        torques: wp.array | torch.Tensor | None = None,
        positions: wp.array | torch.Tensor | None = None,
        body_mask: wp.array | torch.Tensor | None = None,
        env_mask: wp.array | torch.Tensor | None = None,
        is_global: bool = False,
    ):
        """Set (overwrite) forces and torques in the dual input buffers using mask selection.

        All five input buffers are cleared first, then the provided forces/torques are written.

        Args:
            forces: Forces to set. Shape ``(num_envs, num_bodies, 3)``. Defaults to None.
            torques: Torques to set. Shape ``(num_envs, num_bodies, 3)``. Defaults to None.
            positions: Application positions. Shape ``(num_envs, num_bodies, 3)``. Defaults to None.
            body_mask: Boolean body mask. Shape ``(num_bodies,)``. Defaults to None (all bodies).
            env_mask: Boolean environment mask. Shape ``(num_envs,)``. Defaults to None (all envs).
            is_global: If True, forces/torques are in the world frame. Defaults to False.
        """
        if env_mask is None:
            env_mask = self._ALL_ENV_MASK
        if body_mask is None:
            body_mask = self._ALL_BODY_MASK

        if forces is None and torques is None:
            warnings.warn(
                "No forces or torques provided. No force will be added.",
                UserWarning,
                stacklevel=2,
            )
            return

        # Clear ALL 5 input buffers before setting
        self._global_force_w.zero_()
        self._global_torque_w.zero_()
        self._global_force_at_com_w.zero_()
        self._local_force_b.zero_()
        self._local_torque_b.zero_()

        self._active = True
        self._dirty = True
        if is_global:
            self._has_global = True
            self._has_local = False
        else:
            self._has_local = True
            self._has_global = False

        wp.launch(
            set_forces_to_dual_buffers_mask,
            dim=(self.num_envs, self.num_bodies),
            inputs=[
                env_mask,
                body_mask,
                forces,
                torques,
                positions,
                is_global,
            ],
            outputs=[
                self._global_force_w,
                self._global_torque_w,
                self._global_force_at_com_w,
                self._local_force_b,
                self._local_torque_b,
            ],
            device=self.device,
        )

    # ------------------------------------------------------------------
    # Composition
    # ------------------------------------------------------------------

    def compose_to_body_frame(self):
        """Fetch current link poses and compose all input buffers into body-frame output buffers.

        This method reads ``asset.data.body_link_pose_w`` (a ``wp.transformf`` 2-D array) and
        launches :func:`compose_wrench_to_body_frame` which:

        1. Rotates global forces and torques into the body frame.
        2. Computes positional torques for ``_global_force_w`` using the lever arm from the
           link position.
        3. Adds the local-frame contributions directly.
        4. Writes the summed results into ``_out_force_b`` and ``_out_torque_b``.

        After this call the dirty flag is cleared.
        """
        link_poses = self._get_link_pose_fn()

        # Zero output buffers before composition (kernel accumulates via assignment)
        self._out_force_b.zero_()
        self._out_torque_b.zero_()

        wp.launch(
            compose_wrench_to_body_frame,
            dim=(self.num_envs, self.num_bodies),
            inputs=[
                self._global_force_w,
                self._global_torque_w,
                self._global_force_at_com_w,
                self._local_force_b,
                self._local_torque_b,
                link_poses,
            ],
            outputs=[
                self._out_force_b,
                self._out_torque_b,
            ],
            device=self.device,
        )

        self._dirty = False

    # ------------------------------------------------------------------
    # Buffer merging
    # ------------------------------------------------------------------

    def add_raw_buffers_from(self, other: WrenchComposer):
        """Element-wise add another composer's five input buffers into this one.

        This is useful for merging contributions from multiple sources (e.g., different action
        terms) before a single composition pass.

        Args:
            other: Another :class:`WrenchComposer` whose input buffers will be added into this one.
        """
        wp.launch(
            add_raw_wrench_buffers,
            dim=(self.num_envs, self.num_bodies),
            inputs=[
                other._global_force_w,
                other._global_torque_w,
                other._global_force_at_com_w,
                other._local_force_b,
                other._local_torque_b,
            ],
            outputs=[
                self._global_force_w,
                self._global_torque_w,
                self._global_force_at_com_w,
                self._local_force_b,
                self._local_torque_b,
            ],
            device=self.device,
        )

        if other._active:
            self._active = True
            self._dirty = True
        if other._has_global:
            self._has_global = True
        if other._has_local:
            self._has_local = True

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, env_ids: wp.array | torch.Tensor | list | None = None, env_mask: wp.array | None = None):
        """Reset input and output buffers to zero.

        Supports three modes:

        1. **Full reset** (``env_ids=None, env_mask=None``): zeros all 7 buffers and clears all flags.
        2. **Index reset** (``env_ids`` provided): zeros only the rows corresponding to the given
           environment indices across all 7 buffers.
        3. **Mask reset** (``env_mask`` provided): zeros rows where the mask is True.

        .. caution::
            If both ``env_ids`` and ``env_mask`` are provided, ``env_mask`` takes precedence.

        Args:
            env_ids: Environment indices to reset. Defaults to None (all environments).
            env_mask: Boolean environment mask. Defaults to None (all environments).
        """
        if env_ids is None and env_mask is None:
            # Full reset: zero everything
            self._global_force_w.zero_()
            self._global_torque_w.zero_()
            self._global_force_at_com_w.zero_()
            self._local_force_b.zero_()
            self._local_torque_b.zero_()
            self._out_force_b.zero_()
            self._out_torque_b.zero_()
            self._active = False
            self._dirty = False
            self._has_local = False
            self._has_global = False
        elif env_mask is not None:
            # Mask-based partial reset: launch reset kernel for each buffer pair
            # We reset all 7 buffers using the mask kernel. The kernel zeros both force and torque
            # arrays for the masked environments, so we call it for the 3 pairs + 1 extra.
            # Input buffers: (global_force_w, global_torque_w), (global_force_at_com_w, local_force_b),
            #                (local_torque_b, out_force_b) -- but that mixes semantics. Instead, launch
            # the kernel once per pair of buffers that share the same shape.
            wp.launch(
                reset_wrench_composer_mask,
                dim=(self.num_envs, self.num_bodies),
                inputs=[env_mask],
                outputs=[self._global_force_w, self._global_torque_w],
                device=self.device,
            )
            wp.launch(
                reset_wrench_composer_mask,
                dim=(self.num_envs, self.num_bodies),
                inputs=[env_mask],
                outputs=[self._global_force_at_com_w, self._local_force_b],
                device=self.device,
            )
            wp.launch(
                reset_wrench_composer_mask,
                dim=(self.num_envs, self.num_bodies),
                inputs=[env_mask],
                outputs=[self._local_torque_b, self._out_force_b],
                device=self.device,
            )
            wp.launch(
                reset_wrench_composer_mask,
                dim=(self.num_envs, self.num_bodies),
                inputs=[env_mask],
                outputs=[self._out_torque_b, self._out_torque_b],
                device=self.device,
            )
        else:
            # Index-based partial reset
            env_ids = self._resolve_env_ids(env_ids)
            wp.launch(
                reset_wrench_composer_index,
                dim=(env_ids.shape[0], self.num_bodies),
                inputs=[env_ids],
                outputs=[self._global_force_w, self._global_torque_w],
                device=self.device,
            )
            wp.launch(
                reset_wrench_composer_index,
                dim=(env_ids.shape[0], self.num_bodies),
                inputs=[env_ids],
                outputs=[self._global_force_at_com_w, self._local_force_b],
                device=self.device,
            )
            wp.launch(
                reset_wrench_composer_index,
                dim=(env_ids.shape[0], self.num_bodies),
                inputs=[env_ids],
                outputs=[self._local_torque_b, self._out_force_b],
                device=self.device,
            )
            wp.launch(
                reset_wrench_composer_index,
                dim=(env_ids.shape[0], self.num_bodies),
                inputs=[env_ids],
                outputs=[self._out_torque_b, self._out_torque_b],
                device=self.device,
            )

    # ------------------------------------------------------------------
    # Deprecated methods (delegate to index variants)
    # ------------------------------------------------------------------

    def add_forces_and_torques(
        self,
        forces: wp.array | torch.Tensor | None = None,
        torques: wp.array | torch.Tensor | None = None,
        positions: wp.array | torch.Tensor | None = None,
        body_ids: torch.Tensor | None = None,
        env_ids: torch.Tensor | None = None,
        is_global: bool = False,
    ):
        """Deprecated, same as :meth:`add_forces_and_torques_index`."""
        warnings.warn(
            "The function 'add_forces_and_torques' will be deprecated in a future release. Please"
            " use 'add_forces_and_torques_index' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.add_forces_and_torques_index(forces, torques, positions, body_ids, env_ids, is_global)

    def set_forces_and_torques(
        self,
        forces: wp.array | torch.Tensor | None = None,
        torques: wp.array | torch.Tensor | None = None,
        positions: wp.array | torch.Tensor | None = None,
        body_ids: wp.array | torch.Tensor | None = None,
        env_ids: wp.array | torch.Tensor | None = None,
        is_global: bool = False,
    ):
        """Deprecated, same as :meth:`set_forces_and_torques_index`."""
        warnings.warn(
            "The function 'set_forces_and_torques' will be deprecated in a future release. Please"
            " use 'set_forces_and_torques_index' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.set_forces_and_torques_index(forces, torques, positions, body_ids, env_ids, is_global)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_env_ids(self, env_ids: wp.array | torch.Tensor | list | None) -> wp.array:
        """Resolve environment indices to a warp int32 array.

        Args:
            env_ids: Environment indices as None, slice, list, torch.Tensor, or wp.array.

        Returns:
            wp.array of dtype wp.int32.
        """
        if env_ids is None or env_ids == slice(None):
            return self._ALL_ENV_INDICES
        if isinstance(env_ids, list):
            return wp.array(env_ids, dtype=wp.int32, device=self.device)
        if isinstance(env_ids, torch.Tensor):
            return wp.from_torch(env_ids.to(torch.int32).contiguous(), dtype=wp.int32)
        return env_ids

    def _resolve_body_ids(self, body_ids: wp.array | torch.Tensor | list | None) -> wp.array:
        """Resolve body indices to a warp int32 array.

        Args:
            body_ids: Body indices as None, slice, list, torch.Tensor, or wp.array.

        Returns:
            wp.array of dtype wp.int32.
        """
        if body_ids is None or body_ids == slice(None):
            return self._ALL_BODY_INDICES
        if isinstance(body_ids, list):
            return wp.array(body_ids, dtype=wp.int32, device=self.device)
        if isinstance(body_ids, torch.Tensor):
            return wp.from_torch(body_ids.to(torch.int32).contiguous(), dtype=wp.int32)
        return body_ids
