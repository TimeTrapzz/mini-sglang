import asyncio
from concurrent.futures import Future
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from minisgl.recommendation.scheduler import RecommendationWorker
from minisgl.recommendation.server import create_app, wait_result
from test_worker import FakeRuntime


class Tokenizer:
    def __len__(self):
        return 100

    def apply_chat_template(self, messages, **kwargs):
        return [1, 2, 3]

    def decode(self, tokens, **kwargs):
        return " ".join(map(str, tokens))


def test_api_ranked_choices_usage_and_validation():
    runtime = FakeRuntime()
    worker = RecommendationWorker(lambda: runtime)
    with TestClient(create_app(worker, Tokenizer(), "test-model")) as client:
        assert client.get("/health").status_code == 200
        response = client.post("/v1/chat/completions", json={"input_ids": [1, 2], "n": 2})
        assert response.status_code == 200, response.text
        body = response.json()
        assert [c["sglext"]["item_id"] for c in body["choices"]] == ["a", "b"]
        assert body["usage"]["completion_tokens"] == 8
        assert body["usage"]["total_tokens"] == 10
        assert all(c["finish_reason"] == "stop" for c in body["choices"])
        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "recommend"}], "max_tokens": 4},
        )
        assert response.status_code == 200
        for data, status in [
            ({"input_ids": []}, 400),
            ({"input_ids": [100]}, 400),
            ({"input_ids": [1], "n": 3}, 400),
            ({"input_ids": [1], "model": "wrong"}, 404),
            ({"input_ids": [1], "max_tokens": 1}, 400),
            ({"input_ids": [1], "stream": True}, 422),
            ({"input_ids": [1], "top_p": 0.5}, 422),
            ({"input_ids": [1], "messages": []}, 422),
        ]:
            assert client.post("/v1/chat/completions", json=data).status_code == status
    assert runtime.closed


def test_api_backpressure_and_health_failure():
    runtime = FakeRuntime(blocked=True)
    worker = RecommendationWorker(lambda: runtime, max_pending=1)
    running = worker.submit([1], 1)
    assert runtime.entered.wait(2)
    try:
        with TestClient(create_app(worker, Tokenizer(), "test")) as client:
            assert client.post("/v1/chat/completions", json={"input_ids": [1]}).status_code == 429
            runtime.resume.set()
            running.result(5)
            worker.close()
            assert client.get("/health").status_code == 503
    finally:
        runtime.resume.set()
        worker.close()


def test_disconnect_cancels_gpu_job_future():
    async def disconnected():
        return True

    future = Future()
    with pytest.raises(HTTPException) as error:
        asyncio.run(wait_result(SimpleNamespace(is_disconnected=disconnected), future))
    assert error.value.status_code == 499
    assert future.cancelled()
