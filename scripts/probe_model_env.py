#!/usr/bin/env python3
"""Report optional model dependencies available in the active Python environment."""

from __future__ import annotations

import importlib.util
import json
import platform


def version(name: str) -> str | None:
    try:
        module = __import__(name)
    except Exception:
        return None
    return str(getattr(module, "__version__", "unknown"))


def main() -> None:
    packages = ("torch", "torchvision", "transformers", "groundingdino", "sam2", "qwen_vl_utils")
    report = {
        "python": platform.python_version(),
        "packages": {
            name: {
                "available": importlib.util.find_spec(name) is not None,
                "version": version(name),
            }
            for name in packages
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
