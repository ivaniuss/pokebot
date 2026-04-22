from agent import agent
import asyncio
from collections.abc import AsyncIterable
from fastapi import FastAPI
from fastapi.sse import EventSourceResponse, ServerSentEvent
from pydantic import BaseModel
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from deepagents import create_deep_agent
import uvicorn

load_dotenv(find_dotenv())

app = FastAPI(title="PokeBot")

class ChatRequest(BaseModel):
    message: str

BANNER = """
Examples:
  > What items should I put on Gardevoir?
  > My team is Ralts, Kirlia, Pikachu, Jolteon. What synergies am I close to?
  > I have 30g. Shop shows Charizard, Blastoise, Venusaur, Pikachu. What should I buy?
"""

async def agent_generator(query: str) -> AsyncIterable[ServerSentEvent]:
    try:
        async for chunk in agent.astream(
            {"messages": [{"role": "user", "content": query}]},
            stream_mode="messages",
            version="v2",
        ):
            if chunk["type"] == "messages":
                token, metadata = chunk["data"]
                
                content = token.content_blocks if hasattr(token, 'content_blocks') else token.content
                
                # Normalize structured content (list of blocks) to string
                if isinstance(content, list):
                    text_parts = []
                    for b in content:
                        if isinstance(b, dict) and b.get("type") == "text":
                            text_parts.append(b.get("text", ""))
                        elif isinstance(b, str):
                            text_parts.append(b)
                    content = "".join(text_parts)

                data = {
                    "node": metadata.get("langgraph_node"),
                    "content": content
                }

                yield ServerSentEvent(
                    data=data, 
                    event="message"
                )
                
    except Exception as e:
        yield ServerSentEvent(data={"error": str(e)}, event="error")

@app.post("/chat/stream", response_class=EventSourceResponse)
async def chat_stream(request: ChatRequest) -> AsyncIterable[ServerSentEvent]:
    async for event in agent_generator(request.message):
        yield event

@app.post("/chat")
async def chat(request: ChatRequest):
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": request.message}]},
    )
    content = result["messages"][-1].content
    
    # Handle structured content (list of blocks) common with Gemini
    if isinstance(content, list):
        text_blocks = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text_blocks.append(block.get("text", ""))
            elif isinstance(block, str):
                text_blocks.append(block)
        content = "".join(text_blocks)
        
    return {"message": content}

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="[IP_ADDRESS]", port=8000)