from __future__ import annotations

import importlib.util
import json
import sys
from importlib import metadata


MODULES = {
    "base": ["numpy", "cv2", "PIL", "docx", "yaml", "requests"],
    "geo": ["rasterio"],
    "det": ["torch", "torchvision", "ultralytics"],
    "api": ["openai"],
    "llm": ["transformers", "accelerate"],
    "dev": ["pytest", "ipykernel"],
    "notebook": ["jupyter"],
}


def module_status(module_name: str) -> dict[str, object]:
    found = importlib.util.find_spec(module_name) is not None
    version = None
    if found:
        package_name = {
            "PIL": "pillow",
            "cv2": "opencv-python",
            "docx": "python-docx",
            "yaml": "PyYAML",
        }.get(module_name, module_name)
        try:
            version = metadata.version(package_name)
        except metadata.PackageNotFoundError:
            version = "unknown"
    return {"installed": found, "version": version}


def torch_status() -> dict[str, object]:
    if importlib.util.find_spec("torch") is None:
        return {"installed": False}

    import torch

    return {
        "installed": True,
        "version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def main() -> int:
    report: dict[str, object] = {
        "python": sys.version,
        "executable": sys.executable,
        "profiles": {},
        "torch": torch_status(),
    }

    profiles: dict[str, object] = {}
    for profile, modules in MODULES.items():
        modules_status = {module: module_status(module) for module in modules}
        profiles[profile] = {
            "ok": all(item["installed"] for item in modules_status.values()),
            "modules": modules_status,
        }
    report["profiles"] = profiles

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if all(profile["ok"] for profile in profiles.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
