from datetime import timedelta

from app import apikeys
from app.apikeys import ApiKey, hash_key, new_key, utcnow


def test_new_key_is_hashed_not_stored_plaintext():
    raw, record = new_key(name="client")
    assert raw.startswith("aiud_")
    assert record.hash == hash_key(raw)
    assert raw not in record.model_dump_json()


def test_verify_matches_and_ignores_expired():
    raw, record = new_key(name="ok")
    assert apikeys.verify(raw, [record]).id == record.id
    assert apikeys.verify("nope", [record]) is None

    expired = ApiKey(name="old", hash=hash_key("x"), expires_at=utcnow() - timedelta(minutes=1))
    assert apikeys.verify("x", [expired]) is None


def test_session_registry():
    raw, record = new_key(name="session", ttl_minutes=5, admin=True)
    apikeys.clear_sessions()
    apikeys.add_session(record)
    assert record in apikeys.sessions()
    apikeys.drop_session(record.id)
    assert apikeys.sessions() == []


def test_public_view_hides_secret():
    raw, record = new_key(name="c")
    view = apikeys.public(record)
    assert "hash" not in view and "key" not in view
    assert view["prefix"] == raw[:12]
