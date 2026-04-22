"""
agent.py
PokeBot agent definition using deepagents + LangChain OpenAI.
"""

from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from deepagents import create_deep_agent

from tools import (
    get_pokemon_info,
    recommend_items,
    synergy_advisor,
    team_optimizer,
)

load_dotenv(find_dotenv())

SYSTEM_PROMPT = """You are PokeBot, an expert assistant for the game Pokémon Auto Chess.

Your job is to help the player make better decisions during a match by:
- Recommending the best items for each Pokémon they have
- Suggesting which Pokémon to pick to complete or upgrade synergies
- Analyzing their current team composition
- Helping them decide what to buy from the shop given their gold budget

Rules:
- Always use the tools to get accurate game data. Do NOT invent stats or effects.
- When the player mentions Pokémon names, convert them to SCREAMING_SNAKE_CASE before calling tools.
  Example: "Gardevoir" → "GARDEVOIR", "Mr. Mime" → "MR_MIME"
- Be concise and practical. The player is mid-game and needs fast answers.
- When recommending items, always explain briefly WHY each item fits the Pokémon.
- When advising on synergies, prioritize synergies that need only 1-2 more Pokémon.
- Always consider the player's gold budget when making suggestions.
"""

# llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview"
)

agent = create_deep_agent(
    model=llm,
    tools=[
        get_pokemon_info,
        recommend_items,
        synergy_advisor,
        team_optimizer,
    ],
    system_prompt=SYSTEM_PROMPT,
)