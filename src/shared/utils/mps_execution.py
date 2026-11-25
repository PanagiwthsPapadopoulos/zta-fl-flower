"""Opt-in device placement; never replace layers or change model mathematics."""
import os


def _pool_input(module, inputs):
    module._execution_input_device = inputs[0].device
    if inputs[0].device.type == "mps":
        return (inputs[0].to("cpu"),) + inputs[1:]
    return None


def _pool_output(module, inputs, output):
    return output.to(module._execution_input_device)


def configure_native_execution(model):
    """Run the existing adaptive pool on CPU only in the native MPS profile.

    Device copies remain in the autograd graph. Top-level hooks survive deepcopy
    and Flower worker imports; CPU inference copies stay entirely on CPU.
    No parameters, buffers, state_dict keys, or architecture are changed.
    """
    if os.getenv("ZTA_NATIVE_MPS") == "1":
        pool = model.adaptive_pool
        pool.register_forward_pre_hook(_pool_input)
        pool.register_forward_hook(_pool_output)
    return model
