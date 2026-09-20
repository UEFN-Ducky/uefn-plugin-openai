from __future__ import annotations

import json
from pathlib import Path


def test_plugin_declares_nodes():
    data = json.loads((Path(__file__).resolve().parent.parent / "plugin.json").read_text(encoding="utf-8"))
    ids = {n["id"] for n in data["contributes"]["automations"]["nodes"]}
    assert ids == {"openai.complete", "openai.image"}
    model = next(f for n in data["contributes"]["automations"]["nodes"] if n["id"] == "openai.complete" for f in n["config_fields"] if f["id"] == "model")
    assert model["type"] == "model"


def test_image_writes_png(tmp_path, monkeypatch):
    from backend import graph_nodes

    png = b"\x89PNG\r\n\x1a\n"
    monkeypatch.setattr(graph_nodes, "_images_http", lambda body, key: {"data": [{"b64_json": __import__("base64").b64encode(png).decode()}]})
    monkeypatch.setattr(graph_nodes, "_secret", lambda *_a, **_k: "sk-test")

    saved = {}

    def _save(raw, **kw):
        dest = tmp_path / "out.png"
        dest.write_bytes(raw)
        saved["raw"] = raw
        return {"ok": True, "path": str(dest), "name": "out.png"}

    monkeypatch.setattr(graph_nodes, "_persist", _save)
    out = graph_nodes.handle_image({"config": {"prompt": "a duck"}, "payload": {}})
    assert out["ok"] is True
    assert Path(out["path"]).is_file()
    assert saved["raw"].startswith(b"\x89PNG")
    assert "brainrot" not in Path(graph_nodes.__file__).read_text(encoding="utf-8")
