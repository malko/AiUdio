import asyncio
import json

from fastapi.testclient import TestClient

from app.config import ConfigStore
from app.main import create_app
from app.stt_models import ModelIndex, iter_download, load_index

INDEX = {
    "schema_version": 1,
    "defaults": {"asr": "nvidia/nemotron-3.5-asr-streaming-0.6b"},
    "models": [
        {
            "repo": "nvidia/nemotron-3.5-asr-streaming-0.6b",
            "aliases": ["nemotron-3.5", "nemotron-asr"],
            "revision": "abc123",
            "artifacts": [
                {
                    "role": "asr",
                    "type": "file",
                    "filename": "model.q8_0.gguf",
                    "size": 100,
                    "sha256": "deadbeef",
                }
            ],
        }
    ],
}


def test_index_find_and_artifact():
    index = ModelIndex.from_json(INDEX)
    model = index.find("nemotron-3.5")
    assert model is not None and model.repo.startswith("nvidia/")
    assert index.find("nvidia/nemotron-3.5-asr-streaming-0.6b") is model
    assert index.find("unknown") is None
    artifact = model.asr_artifact()
    assert artifact is not None and artifact.filename == "model.q8_0.gguf"


def test_local_path_and_status(tmp_path):
    index = ModelIndex.from_json(INDEX)
    model = index.find("nemotron-3.5")
    artifact = model.asr_artifact()
    models_dir = tmp_path / "models"
    path = model.local_path(artifact, str(models_dir))
    assert str(path).endswith("nvidia/nemotron-3.5-asr-streaming-0.6b/abc123/model.q8_0.gguf")

    status = index.with_status(str(models_dir))
    assert status[0]["cached"] is False
    path.parent.mkdir(parents=True)
    path.write_bytes(b"\0" * 100)
    assert index.with_status(str(models_dir))[0]["cached"] is True


def test_iter_download_cached(tmp_path):
    index = ModelIndex.from_json(INDEX)
    model = index.find("nemotron-3.5")
    artifact = model.asr_artifact()
    path = model.local_path(artifact, str(tmp_path))
    path.parent.mkdir(parents=True)
    path.write_bytes(b"\0" * 100)

    async def collect():
        return [event async for event in iter_download(model, artifact, str(tmp_path), "https://example.invalid")]

    events = asyncio.run(collect())
    assert events[-1]["status"] == "cached"


def test_load_index_missing(tmp_path):
    assert load_index(str(tmp_path / "nope.json")) is None


def test_api_stt_models(tmp_path):
    index_path = tmp_path / "model-index.json"
    index_path.write_text(json.dumps(INDEX), encoding="utf-8")
    models_dir = tmp_path / "models"

    store = ConfigStore(tmp_path / "config.yaml")
    store.settings.stt.enabled = True
    store.settings.stt.model_index = str(index_path)
    store.settings.stt.models_dir = str(models_dir)

    app = create_app(store)
    with TestClient(app) as client:
        r = client.get("/api/stt/models")
        assert r.status_code == 200
        body = r.json()
        assert body["available"] is True
        assert body["models"][0]["cached"] is False
