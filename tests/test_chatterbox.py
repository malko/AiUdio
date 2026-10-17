import base64
import io
import os

import numpy as np
import soundfile as sf

from app.chatterbox_backend import resolve_python
from app.chatterbox_worker import _accepts, _detect_t3, _filter_kwargs
from app.config import Settings
from app.engine import TTSEngine


def test_accepts_adapts_to_signature():
    def without_t3(device):  # noqa: ARG001
        return None

    def with_t3(device, t3_model=None):  # noqa: ARG001
        return None

    def with_kwargs(device, **kwargs):  # noqa: ARG001
        return None

    assert _accepts(without_t3, "t3_model") is False
    assert _accepts(with_t3, "t3_model") is True
    assert _accepts(with_kwargs, "anything") is True
    assert _filter_kwargs(without_t3, {"device": "cpu", "t3_model": "v3"}) == {"device": "cpu"}
    assert _filter_kwargs(with_kwargs, {"t3_model": "v3"}) == {"t3_model": "v3"}


def test_detect_t3(tmp_path):
    (tmp_path / "t3_mtl23ls_v2.safetensors").write_bytes(b"x")
    assert _detect_t3(tmp_path, "v3") == "t3_mtl23ls_v2.safetensors"
    (tmp_path / "t3_mtl23ls_v3.safetensors").write_bytes(b"x")
    assert _detect_t3(tmp_path, "v3") == "t3_mtl23ls_v3.safetensors"


def test_resolve_python_missing():
    assert resolve_python("") in (None, "/opt/cb/venv/bin/python")
    assert resolve_python("/nonexistent/python") in (None, "/opt/cb/venv/bin/python")


def test_chatterbox_params_selection():
    engine = TTSEngine(Settings())
    params = engine._chatterbox_params(
        {"temperature": 0.8, "top_p": 0.9, "top_k": 50, "exaggeration": 0.6,
         "cfg_weight": 0.4, "min_p": 0.02, "max_new_tokens": 100}
    )
    assert params["temperature"] == 0.8
    assert params["exaggeration"] == 0.6
    assert "top_k" not in params        # Qwen-only
    assert "max_new_tokens" not in params


def test_chatterbox_ref_data_url(tmp_path):
    engine = TTSEngine(Settings())
    buf = io.BytesIO()
    sf.write(buf, np.zeros(800, dtype=np.float32), 16000, format="WAV")
    data = "data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode()
    path = engine._chatterbox_ref(data)
    assert path and os.path.isfile(path)
    os.unlink(path)


def test_chatterbox_ref_local_path():
    engine = TTSEngine(Settings())
    assert engine._chatterbox_ref("/tmp/ref.wav") == "/tmp/ref.wav"
    assert engine._chatterbox_ref(None) is None
