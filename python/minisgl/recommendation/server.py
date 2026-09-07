from __future__ import annotations

import asyncio
import queue
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Message(BaseModel):
    role: str
    content: str


class CompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str | None = None
    messages: list[Message] | None = None
    input_ids: list[int] | None = None
    n: int = Field(default=1, ge=1)
    temperature: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    seed: int = Field(default=0, ge=0, le=2**63 - 1)
    max_tokens: int | None = Field(default=None, ge=1)
    stream: bool = False

    @model_validator(mode="after")
    def validate_input(self):
        if (self.messages is None) == (self.input_ids is None):
            raise ValueError("Provide exactly one of messages or input_ids")
        if self.stream:
            raise ValueError("Ranked SID responses do not support streaming")
        return self


async def wait_result(request: Request, future):
    wrapped = asyncio.wrap_future(future)
    try:
        while not wrapped.done():
            if await request.is_disconnected():
                future.cancel()
                raise HTTPException(499, "Client disconnected")
            await asyncio.wait([wrapped], timeout=0.1)
        return await wrapped
    except asyncio.CancelledError:
        future.cancel()
        raise


def create_app(worker, tokenizer, model_name: str):
    @asynccontextmanager
    async def lifespan(app):
        yield
        await asyncio.to_thread(worker.close)

    app = FastAPI(title="mini-sglang recommendation", lifespan=lifespan)

    @app.get("/health")
    async def health():
        if worker.error or worker.stopping.is_set():
            raise HTTPException(503, "GPU worker unavailable")
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": model_name, "object": "model"}]}

    @app.post("/v1/chat/completions")
    async def completions(body: CompletionRequest, request: Request):
        if body.model is not None and body.model != model_name:
            raise HTTPException(404, "Unknown model")
        depth = worker.runtime.catalog.depth
        if body.max_tokens is not None and body.max_tokens != depth:
            raise HTTPException(400, f"max_tokens must equal catalog depth {depth}")
        try:
            if body.input_ids is not None:
                prompt = body.input_ids
            else:
                prompt = await asyncio.to_thread(
                    tokenizer.apply_chat_template,
                    [m.model_dump() for m in body.messages],
                    tokenize=True,
                    add_generation_prompt=True,
                )
            if not prompt or min(prompt) < 0 or max(prompt) >= len(tokenizer):
                raise ValueError("Prompt token IDs are outside the model vocabulary")
            future = worker.submit(prompt, body.n, body.temperature, body.seed)
        except queue.Full as exc:
            raise HTTPException(429, "Recommendation queue is full") from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(503, "GPU worker unavailable") from exc
        try:
            result = await wait_result(request, future)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except (RuntimeError, MemoryError) as exc:
            raise HTTPException(503, "Recommendation execution failed") from exc
        choices = []
        for i, beam in enumerate(result["beams"]):
            content = await asyncio.to_thread(
                tokenizer.decode, beam["token_ids"], skip_special_tokens=False
            )
            choices.append(
                {
                    "index": i,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                    "sglext": beam,
                }
            )
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": choices,
            "usage": {
                "prompt_tokens": result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
                "total_tokens": result["prompt_tokens"] + result["completion_tokens"],
                "prompt_tokens_details": {"cached_tokens": result["cached_tokens"]},
            },
        }

    @app.post("/start_profile")
    async def start_profile():
        try:
            return await asyncio.wrap_future(worker.profile("start"))
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/stop_profile")
    async def stop_profile():
        try:
            return await asyncio.wrap_future(worker.profile("stop"))
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc

    return app
