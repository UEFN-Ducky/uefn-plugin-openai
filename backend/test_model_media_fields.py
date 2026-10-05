"""OpenAI dashboard records report image/video/audio limits only when present."""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path

ROOT = Path(os.path.abspath(__file__)).parents[1]
_APP = None
for _p in ROOT.parents:
    for _name in ("UEFN-Ducky-Release", "UEFN-Ducky-video", "UEFN-Ducky"):
        _app = _p / _name / "ducky_app"
        if (_app / "backend" / "agent").is_dir():
            _APP = str(_app)
            break
    else:
        continue
    break


def _load():
    # The plugin dir is itself a ``backend`` package; let the host's win while importing.
    saved = {k: sys.modules.pop(k) for k in list(sys.modules) if k == "backend" or k.startswith("backend.")}
    saved_path = list(sys.path)
    sys.path[:] = [_APP] + [p for p in sys.path if os.path.abspath(p) != str(ROOT)]
    try:
        return _load_module()
    finally:
        sys.path[:] = saved_path
        for k in [k for k in sys.modules if k == "backend" or k.startswith("backend.")]:
            del sys.modules[k]
        sys.modules.update(saved)


def _load_module():
    pkg = types.ModuleType("openai_gw")
    pkg.__path__ = [str(ROOT / "backend")]
    sys.modules["openai_gw"] = pkg
    prov = types.ModuleType("openai_gw.openai_provider")  # heavy; thinking helpers are irrelevant here
    prov.model_supports_thinking_effort = lambda mid: False
    prov.thinking_menu = lambda mid: None
    sys.modules["openai_gw.openai_provider"] = prov
    spec = importlib.util.spec_from_file_location("openai_gw.model_fetch", ROOT / "backend" / "model_fetch.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _info(**record):
    return _load()._openai_info_from_dashboard(record, "gpt-x")


def test_audio_and_max_images_from_record():
    info = _info(features=["image_content", "audio_input"], max_images=10)
    assert (info.max_images, info.supports_audio) == (10, True)
    assert _info(features=["image_content", "audio"]).supports_audio is True
    assert _info(features=["image_content"], max_image_inputs=4).max_images == 4


def test_audio_unknown_when_features_lack_audio():
    assert _info(features=["function_calling"]).supports_audio is None


def test_media_fields_unknown_when_absent():
    info = _info()
    assert (info.max_images, info.supports_video, info.supports_audio) == (None, None, None)
