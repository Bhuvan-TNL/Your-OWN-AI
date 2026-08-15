from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")

    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "your-own-ai"
    assert body["version"] == "0.1.0"
    assert body["llm_provider"]
    assert body["embedding_model"]
