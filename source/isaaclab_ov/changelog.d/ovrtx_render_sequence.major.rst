Changed
^^^^^^^

* **Breaking:** Changed ``OVRTXRenderer.render`` to accept a sequence of camera render-data objects
  and submit all requested products in one native renderer step. Updated direct callers to pass
  ``renderer.render([render_data])`` for a single camera, or pass all prepared camera render-data
  objects together to batch their rendering. Empty sequences performed no work, and missing product
  frames raised an error instead of leaving stale output buffers marked as current.
