import discord
import httpx
import asyncio
import os
import time
import logging
from dotenv import load_dotenv, find_dotenv
from agent import agent

# =========================
# Environment setup
# =========================
load_dotenv(find_dotenv())
TOKEN = os.getenv("DISCORD_BOT_TOKEN")

# =========================
# Logging configuration
# =========================
logging.basicConfig(
    level=logging.INFO,  # change to DEBUG for more verbosity
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger("discord-bot")

# Optional: reduce httpx noise
logging.getLogger("httpx").setLevel(logging.WARNING)

# =========================
# Discord configuration
# =========================
intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)

# =========================
# HTTP config (optional)
# =========================
timeout = httpx.Timeout(60.0, connect=10.0)
API_URL = "http://localhost:8000/chat"


# =========================
# Events
# =========================
@client.event
async def on_ready():
    logger.info(f"✅ Bot logged in as {client.user}")


@client.event
async def on_message(message):
    # Prevent bot from replying to itself
    if message.author == client.user:
        return

    logger.debug(f"📩 Message received from {message.author}: {message.content}")

    # Only respond if mentioned
    if client.user not in message.mentions:
        logger.debug("🤐 Bot was not mentioned, ignoring message")
        return

    # Clean mention from message
    clean_input = message.content.replace(f"<@{client.user.id}>", "").strip()
    logger.info(f"🧠 Processing cleaned input: {clean_input}")

    async with message.channel.typing():
        try:
            start_time = time.time()

            logger.debug("⚙️ Invoking agent...")
            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": clean_input}]}
            )

            response = result["messages"][-1].content
            
            # Handle structured content (list of blocks) common with Gemini
            if isinstance(response, list):
                text_blocks = []
                for block in response:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_blocks.append(block.get("text", ""))
                    elif isinstance(block, str):
                        text_blocks.append(block)
                response = "".join(text_blocks)

            elapsed = time.time() - start_time
            logger.info(f"⏱️ Agent response time: {elapsed:.2f}s")
            logger.info(f"💬 Generated response: {response}")

            await message.channel.send(response)

        except Exception:
            logger.exception("❌ Error while processing message")
            await message.channel.send("Error 😢")


# =========================
# Run bot
# =========================
if __name__ == "__main__":
    if not TOKEN:
        logger.error("❌ DISCORD_BOT_TOKEN is not set in environment")
    else:
        logger.info("🚀 Starting bot...")
        client.run(TOKEN)