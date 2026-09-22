Changed
^^^^^^^

* **Breaking:** Changed ``BaseRenderer.render()`` to accept a sequence of render-data objects.
  Migrate direct calls from ``render(data)`` to ``render([data])`` for one camera, or pass a
  sequence for multiple cameras. Custom renderer implementations must handle every entry;
  an empty sequence performs no rendering.
* Batched pending camera captures by shared renderer through ``RenderContext``. Eager scene
  updates submitted captures together, while lazy reads also captured due peer cameras.
  Clone output tensors when retaining snapshots because peer captures reuse their buffers.
  Preserved individual update periods and committed capture timestamps and frame counters
  only after all rendering and output reads in a group succeeded.
