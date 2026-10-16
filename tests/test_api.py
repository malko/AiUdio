def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["engine"] == "dummy"


def test_config_endpoint(client):
    r = client.get("/api/config")
    assert r.status_code == 200
    body = r.json()
    assert "families" in body["models"]
    assert body["audio"]["sample_rate"] == 24000


def test_models_and_load(client):
    assert client.get("/api/models").status_code == 200
    r = client.post("/api/models/load", json={"family": "custom_voice"})
    assert r.status_code == 200
    assert "custom_voice" in r.json()["loaded"]


def test_voices(client):
    r = client.get("/api/voices?family=custom_voice")
    assert r.status_code == 200
    assert "Vivian" in r.json()["speakers"]


def test_tts_non_stream_wav(client):
    r = client.post("/api/tts", json={"text": "Bonjour le monde.", "format": "wav"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("audio/wav")
    assert r.content[:4] == b"RIFF"


def test_tts_stream_pcm(client):
    r = client.post("/api/tts", json={"text": "Bonjour. Comment ça va ?", "stream": True})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("audio/pcm")
    assert len(r.content) > 0


def test_openai_speech(client):
    r = client.post("/v1/audio/speech", json={"input": "Hello", "voice": "Vivian"})
    assert r.status_code == 200
    assert r.content[:4] == b"RIFF"


def test_tts_requires_text(client):
    assert client.post("/api/tts", json={}).status_code == 400


def test_auth_blocks_and_allows(auth_client):
    assert auth_client.get("/api/config").status_code == 401
    assert auth_client.get("/health").status_code == 200
    # API key via header
    assert auth_client.get("/api/config", headers={"X-API-Key": "k-123"}).status_code == 200
    # Basic
    r = auth_client.get("/api/config", auth=("admin", "s3cret"))
    assert r.status_code == 200


def _clone_body(ref_audio):
    return {"text": "hello", "family": "base", "ref_audio": ref_audio}


def test_ref_audio_defaults_are_restrictive(client):
    # Remote URLs and local paths are refused by default (SSRF / file disclosure).
    assert client.post("/api/tts", json=_clone_body("http://example.com/a.wav")).status_code == 400
    assert client.post("/api/tts", json=_clone_body("/etc/passwd")).status_code == 400
    assert client.post("/api/tts", json=_clone_body("file:///etc/passwd")).status_code == 400
    # Base64 data-URL is allowed.
    assert client.post("/api/tts", json=_clone_body("data:audio/wav;base64,AAAA")).status_code == 200


def test_ref_audio_permissive_and_traversal(permissive_media_client):
    c = permissive_media_client
    assert c.post("/api/tts", json=_clone_body("http://example.com/a.wav")).status_code == 200
    assert c.post("/api/tts", json=_clone_body("ref.wav")).status_code == 200
    # Path traversal outside the reference directory is refused.
    assert c.post("/api/tts", json=_clone_body("../../etc/passwd")).status_code == 400
