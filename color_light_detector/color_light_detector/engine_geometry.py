"""Resolve the actual TensorRT input contract once, on the YOLO worker.

Loading a predictor here does not run an extra inference. Both full frames and
crops use this one engine and the same letterbox size; results remain in source
image coordinates through Ultralytics postprocessing.
"""


def engine_contract(yolo, device, precision_options, configured_size):
    if yolo.predictor is None:
        arguments = dict(yolo.overrides)
        if yolo.task not in ("detect", "segment"):
            raise ValueError("Task1 engine must be a detection or segmentation model")
        arguments.update(device=device, task=yolo.task, mode="predict", batch=1,
                         imgsz=configured_size, rect=False, save=False, verbose=False)
        arguments.update(precision_options)
        predictor = yolo.task_map[yolo.task]["predictor"](
            overrides=arguments, _callbacks=yolo.callbacks)
        predictor.setup_model(model=yolo.model, verbose=False)
        yolo.predictor = predictor
    wrapper = yolo.predictor.model
    if set(wrapper.names) != {0}:
        raise ValueError("Task1 requires one beacon class at class ID 0")
    backend_task = getattr(wrapper, "task", None)
    if backend_task in ("detect", "segment") and backend_task != yolo.task:
        raise ValueError("TensorRT metadata task disagrees with model_task")
    backend = getattr(wrapper, "backend", wrapper)  # 8.4 and older AutoBackend
    bindings = getattr(backend, "bindings", {})
    if "images" not in bindings:
        raise ValueError("TensorRT images binding unavailable; cannot verify input geometry")
    shape = tuple(int(v) for v in bindings["images"].shape)
    dynamic = bool(getattr(backend, "dynamic", False))
    if dynamic:
        engine = backend.model
        if hasattr(engine, "get_tensor_profile_shape"):
            minimum, optimum, maximum = engine.get_tensor_profile_shape("images", 0)
        else:
            index = engine.get_binding_index("images")
            minimum, optimum, maximum = engine.get_profile_shape(0, index)
        if not int(minimum[0]) <= 1 <= int(maximum[0]):
            raise ValueError("TensorRT profile must support batch 1")
        shape = (1, int(optimum[1]), int(optimum[2]), int(optimum[3]))
    if len(shape) != 4 or shape[:2] != (1, 3):
        raise ValueError("Task1 requires a batch-1, three-channel BCHW TensorRT engine")
    stride = int(getattr(wrapper, "stride", 32))
    if any(v < 32 or v > 4096 or v % stride for v in shape[2:]):
        raise ValueError("TensorRT H/W must be stride-aligned and within 32..4096")
    return {"shape_bchw": list(shape), "dynamic": dynamic,
            "policy": "PROFILE_OPTIMUM" if dynamic else "FIXED_BINDING"}
