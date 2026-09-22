Changed
^^^^^^^

* **Breaking:** Changed ``IsaacRtxRenderer.render()`` to accept a sequence of
  render-data objects and update RTX once before extracting their outputs.
  Migrate direct calls from ``render(data)`` to ``render([data])`` for one camera,
  or pass a sequence for multiple cameras. An empty sequence performs no rendering.

Fixed
^^^^^

* Invalidated the RTX render-update stamp after explicit camera changes or resets so
  repeated captures within a physics step observed updated camera poses and intrinsics.
