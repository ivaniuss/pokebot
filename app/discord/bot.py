"""
app/discord/bot.py
Discord bot powered by the LangGraph state-driven agent.
Dispatches every message through the graph, then builds embeds from structured state.
"""

import logging
import os
import time

import discord
from dotenv import find_dotenv, load_dotenv

from app.agent import graph
from app.embeds import build_embed_from_state

# ── Setup ─────────────────────────────────────────────────────────────────────
load_dotenv(find_dotenv())
TOKEN = os.getenv("DISCORD_BOT_TOKEN")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("pokebot")
logging.getLogger("httpx").setLevel(logging.WARNING)

# ── Discord client ───────────────────────────────────────────────────────────
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

MAX_PLAIN_LEN = 1900


# ── Events ────────────────────────────────────────────────────────────────────

@client.event
async def on_ready():
    logger.info(f"✅ PokeBot online as {client.user}")


@client.event
async def on_message(message: discord.Message):
    if message.author == client.user:
        return

    is_mentioned = client.user in message.mentions
    is_dm = isinstance(message.channel, discord.DMChannel)
    if not is_mentioned and not is_dm:
        return

    clean_input = message.content.replace(f"<@{client.user.id}>", "").strip()
    if not clean_input:
        return

    logger.info(f"📩 [{message.author}] {clean_input}")

    async with message.channel.typing():
        try:
            start = time.time()

            # Run the LangGraph agent
            result_state = await graph.ainvoke({"user_input": clean_input})

            elapsed = time.time() - start
            response = result_state.get("response", "")
            intent = result_state.get("intent", "unknown")
            tool = result_state.get("tool_name", "none")

            logger.info(
                f"⏱ {elapsed:.2f}s | {len(response)} chars | "
                f"intent={intent} | tool={tool}"
            )
            logger.info(f"🔍 Response preview: {response[:200]}")

            # Build Discord embed from the final state
            embed, files = build_embed_from_state(result_state)
            logger.info(f"🎨 Embed built: {embed is not None} | Files: {len(files)}")

            if embed:
                await message.channel.send(embed=embed, files=files)
            else:
                await _send_text(message.channel, response)

        except Exception:
            logger.exception("❌ Error processing message")
            await message.channel.send("Something went wrong 😢")


async def _send_text(channel: discord.TextChannel, text: str):
    if not text:
        await channel.send("I couldn't process that request.")
        return
    if len(text) <= MAX_PLAIN_LEN:
        await channel.send(text)
        return
    chunks = [text[i : i + MAX_PLAIN_LEN] for i in range(0, len(text), MAX_PLAIN_LEN)]
    for chunk in chunks:
        await channel.send(chunk)


# ── Entry point ───────────────────────────────────────────────────────────────

def run_discord_bot():
    if not TOKEN:
        logger.error("❌ DISCORD_BOT_TOKEN not set in .env")
        return
    logger.info("🚀 Starting PokeBot...")
    client.run(TOKEN)


if __name__ == "__main__":
    run_discord_bot()