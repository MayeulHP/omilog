from fastapi.testclient import TestClient


def test_health(client: TestClient):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_readiness(client: TestClient):
    r = client.get("/readiness")
    assert r.status_code == 200


def test_login_rejects_bad_password(client: TestClient):
    r = client.post(
        "/auth/jwt/login",
        data={"username": "test", "password": "wrong"},
    )
    assert r.status_code == 401


def test_login_rejects_unknown_user(client: TestClient, password: str):
    r = client.post(
        "/auth/jwt/login",
        data={"username": "nobody", "password": password},
    )
    assert r.status_code == 401


def test_login_succeeds(client: TestClient, auth_token: str):
    assert auth_token
    assert len(auth_token) > 20


def test_conversations_requires_auth(client: TestClient):
    r = client.get("/api/conversations")
    assert r.status_code == 401


def test_conversations_empty_when_authed(client: TestClient, auth_token: str):
    r = client.get(
        "/api/conversations",
        headers={"Authorization": f"Bearer {auth_token}"},
    )
    assert r.status_code == 200
    assert r.json() == []


def test_ws_rejects_missing_token(client: TestClient):
    # FastAPI's TestClient raises on close-during-handshake; use ws_connect and
    # expect a disconnect immediately.
    from starlette.testclient import WebSocketDenialResponse
    from starlette.websockets import WebSocketDisconnect

    try:
        with client.websocket_connect("/ws"):
            raise AssertionError("WS should not have connected without a token")
    except (WebSocketDisconnect, WebSocketDenialResponse):
        pass


def test_ws_accepts_with_token_and_writes_audio(client: TestClient, auth_token: str, tmp_path):
    import os
    from pathlib import Path

    payload = b"\x00\x01\x02\x03" * 64  # 256 bytes of nonsense "audio"
    with client.websocket_connect(f"/ws?codec=opus&token={auth_token}") as ws:
        ws.send_json({"type": "audio-start", "data": {"rate": 16000}})
        ws.send_bytes(payload)
        ws.send_json({"type": "audio-stop", "data": {}})

    # The file is now Ogg-wrapped, so the raw payload should appear inside the
    # container — assert via Ogg page parsing rather than byte-equality.
    from tests.test_ogg_opus import parse_pages

    storage = Path(os.environ["OMILOG_STORAGE_DIR"])
    opus_files = list(storage.glob("*.opus"))
    assert opus_files, "no audio file produced"
    found_payload = False
    found_valid_ogg = False
    for f in opus_files:
        data = f.read_bytes()
        if not data.startswith(b"OggS"):
            continue
        found_valid_ogg = True
        for page in parse_pages(data):
            if page["payload"] == payload:
                found_payload = True
                assert page["crc_stored"] == page["crc_computed"]
    assert found_valid_ogg, "no file had Ogg magic"
    assert found_payload, "sent payload not found in any Ogg page"
