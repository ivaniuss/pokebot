"""
app/api/routes.py
FastAPI routes for the PokeBot LangGraph agent.
"""

from collections.abc import AsyncIterable

from fastapi import APIRouter
from fastapi.sse import EventSourceResponse, ServerSentEvent
from pydantic import BaseModel

from app.agent import graph

router = APIRouter()


class ChatRequest(BaseModel):
    message: str


async def agent_generator(query: str) -> AsyncIterable[ServerSentEvent]:
    """Stream LangGraph node updates as SSE events."""
    try:
        async for chunk in graph.astream(
            {"user_input": query},
            stream_mode="updates",
        ):
            for node_name, node_output in chunk.items():
                yield ServerSentEvent(
                    data={
                        "node": node_name,
                        "output": node_output,
                    },
                    event="update",
                )

        # Send final state
        yield ServerSentEvent(data={"status": "done"}, event="done")

    except Exception as e:
        yield ServerSentEvent(data={"error": str(e)}, event="error")


@router.post("/stream", response_class=EventSourceResponse)
async def chat_stream(request: ChatRequest) -> AsyncIterable[ServerSentEvent]:
    async for event in agent_generator(request.message):
        yield event


@router.post("")
async def chat(request: ChatRequest):
    result = await graph.ainvoke({"user_input": request.message})
    return {
        "intent": result.get("intent"),
        "pokemon_names": result.get("pokemon_names", []),
        "tool_name": result.get("tool_name"),
        "response": result.get("response", ""),
    }


@router.get("/health")
async def health_check():
    return {"status": "ok"}
