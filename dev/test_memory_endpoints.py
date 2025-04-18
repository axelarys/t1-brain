import time
import pytest
from fastapi.testclient import TestClient

# import your FastAPI app
from main import app

client = TestClient(app)
API_KEY = "WsRocks1234"

@pytest.fixture(autouse=True)
def wait_for_startup():
    # give FAISS and Neo4j a moment to initialize if needed
    time.sleep(0.1)
    yield

def auth_headers():
    return {"X-API-KEY": API_KEY}

def test_store_and_retrieve_memory():
    # store a piece of text memory
    payload = {
        "session_id": "pytest_s1",
        "query": "I love testing",
        "response": "Testing is fun",
        "memory_type": "semantic",
        "sentiment": "positive"
    }
    r = client.post("/memory/store", json=payload, headers=auth_headers())
    assert r.status_code == 200 or r.status_code == 201

    # retrieve it
    r2 = client.post("/memory/retrieve",
                     json={"session_id":"pytest_s1","query":"I love testing"},
                     headers=auth_headers())
    assert r2.status_code == 200
    body = r2.json()
    assert body["status"] == "retrieved"
    assert body["count"] >= 1
    found = [m for m in body["memory"] if m["query"]=="I love testing"]
    assert found, "Stored memory not found on retrieve"

def test_find_similar_queries_endpoint_and_internal():
    # store two related queries
    for q in ("apple pie","banana bread"):
        client.post("/memory/store",
                    json={
                        "session_id":"pytest_sim",
                        "query":q,
                        "response":"ok",
                        "memory_type":"semantic",
                        "sentiment":"neutral"
                    }, headers=auth_headers())

    # call the aggregate endpoint (which uses find_similar_queries internally)
    r = client.post("/memory/aggregate",
                    json={"session_id":"pytest_sim","query":"I love pie"},
                    headers=auth_headers())
    assert r.status_code == 200
    agg = r.json()
    assert "vector" in agg
    # verify at least one similar result
    assert any("pie" in m["query"] for m in agg["vector"])


def test_tool_memory_fallback_and_storage():
    # fallback route (no tool)
    r = client.post("/tool/memory",
                    json={"session_id":"pytest_tool","user_input":"Just chat"},
                    headers=auth_headers())
    assert r.status_code == 200
    body = r.json()
    # fallback uses status "fallback" and tool "none"
    assert body.get("status") == "fallback"
    assert body.get("tool") == "none"

    # check that semantic memory was stored
    r2 = client.post("/memory/retrieve",
                     json={"session_id":"pytest_tool","query":"Just chat"},
                     headers=auth_headers())
    assert r2.status_code == 200
    assert r2.json()["count"] >= 1


@pytest.mark.parametrize("action,expected_tool", [
    ("DoWebSearch", "web_search"),
    # add other tool names here if available in your tools/ directory
])
def test_tool_memory_execution(action, expected_tool):
    # tool route
    r = client.post("/tool/memory",
                    json={"session_id":"pytest_tool2","user_input":action},
                    headers=auth_headers())
    assert r.status_code == 200
    body = r.json()
    # success from tool agent
    assert body.get("status") == "success"
    assert body.get("tool") == expected_tool

    # and stored as tool memory
    r2 = client.post("/memory/retrieve",
                     json={"session_id":"pytest_tool2","query":action},
                     headers=auth_headers())
    assert r2.status_code == 200
    mems = r2.json()["memory"]
    # at least one stored chunk with memory_type "tool"
    assert any(m["memory_type"]=="tool" for m in mems)

def test_summarize_session():
    # you should have stored metadata for pytest_tool2
    r = client.get("/memory/summarize?session_id=pytest_tool2",
                   headers=auth_headers())
    assert r.status_code == 200
    summ = r.json()
    assert summ["session_id"] == "pytest_tool2"
    # ensure the four lists are present
    for key in ("intents","topics","emotions","keywords"):
        assert key in summ and isinstance(summ[key], list)
