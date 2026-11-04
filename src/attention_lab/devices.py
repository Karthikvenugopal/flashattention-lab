"""Small diagnostics: hardware visibility is specific to the calling process."""

import ctypes
import os
import platform
import sys

import torch


def resolve_device(name: str) -> torch.device:
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable in this process. Run 'attention-lab doctor' in your own Terminal; use --device cpu here.")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this process")
    if name not in ("cpu", "mps", "cuda"):
        raise ValueError("Device must be cpu, mps, or cuda")
    return torch.device(name)


def native_metal_probe():
    if sys.platform != "darwin":
        return {"status": "not applicable"}
    try:
        metal = ctypes.CDLL("/System/Library/Frameworks/Metal.framework/Metal")
        core = ctypes.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
        metal.MTLCopyAllDevices.restype = ctypes.c_void_p
        metal.MTLCreateSystemDefaultDevice.restype = ctypes.c_void_p
        core.CFArrayGetCount.argtypes = [ctypes.c_void_p]
        core.CFArrayGetCount.restype = ctypes.c_long
        core.CFRelease.argtypes = [ctypes.c_void_p]
        devices = metal.MTLCopyAllDevices()
        count = core.CFArrayGetCount(devices) if devices else 0
        if devices:
            core.CFRelease(devices)
        default = metal.MTLCreateSystemDefaultDevice()
        if default:
            core.CFRelease(default)
        return {"device_count": count, "default_device_visible": bool(default)}
    except (OSError, AttributeError) as exc:
        return {"error": str(exc)}


def environment():
    return {
        "python": platform.python_version(), "executable": sys.executable,
        "architecture": platform.machine(), "platform": platform.platform(),
        "macos": platform.mac_ver()[0], "torch": torch.__version__,
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
        "mps_device_count": torch.mps.device_count(),
        "cuda_available": torch.cuda.is_available(),
        "mps_cpu_fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
    }


def doctor():
    result = environment()
    result["native_metal"] = native_metal_probe()
    try:
        x = torch.ones(3, device="mps")
        y = x * 2
        torch.mps.synchronize()
        result["mps_operation"] = {"status": "passed", "result": y.cpu().tolist()}
    except (RuntimeError, TypeError, NotImplementedError) as exc:
        result["mps_operation"] = {"status": "unavailable or failed", "error": str(exc)}
    return result
