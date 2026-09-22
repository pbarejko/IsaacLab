Changed
^^^^^^^

* **Breaking:** Changed ``NewtonWarpRenderer.render()`` to accept a sequence of
  render-data objects and submit their camera tasks together. Migrate direct
  calls from ``render(data)`` to ``render([data])`` for one camera, or pass a
  sequence for multiple cameras. An empty sequence performs no rendering.
