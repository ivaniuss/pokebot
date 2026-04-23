"""
agent.py
PokeBot — State-driven LangGraph agent for Pokémon Auto Chess.
"""

from __future__ import annotations

import logging
import re
import difflib
from typing import Literal, Optional, Any

from dotenv import find_dotenv, load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

from app.tools import recommend_items, team_optimizer, POKEMON as VALID_POKEMON_DB

load_dotenv(find_dotenv())

logger = logging.getLogger("pokebot.agent")

# ── LLM ───────────────────────────────────────────────────────────────────────
# llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview", temperature=0, convert_system_message_to_human=True)

# ═════════════════════════════════════════════════════════════════════════════
# 1. STATE DEFINITION
# ═════════════════════════════════════════════════════════════════════════════

class AgentState(TypedDict):
    user_input: str
    intent: Optional[Literal["items", "team"]]
    pokemon_names: list[str]
    query_items: list[str]
    role: Optional[str]
    tool_name: Optional[str]
    tool_args: Optional[dict]
    tool_result: Optional[str]
    analysis: Optional[str]
    response: Optional[str]

class IntentClassification(BaseModel):
    intent: Literal["items", "team"] = Field(description="The user's primary intent.")

class EntityExtraction(BaseModel):
    pokemon_names: list[str] = Field(description="List of Pokémon names mentioned.")
    query_items: list[str] = Field(description="Specific item names mentioned (e.g. 'explosive band').", default_factory=list)
    role: Optional[Literal["tank", "carry_ap", "carry_atk", "support"]] = Field(description="Role mentioned.", default=None)

# ── Chains ────────────────────────────────────────────────────────────────────
_INTENT_SYSTEM = "Classify if the user wants item recommendations ('items') or team/synergy advice ('team')."
_intent_chain = llm.with_structured_output(IntentClassification)

_ENTITY_SYSTEM = (
    "Extract Pokémon names (normalized to SCREAMING_SNAKE_CASE) and any specific items mentioned. "
    "Also detect the role (tank, carry_ap, carry_atk, support)."
)
_entity_chain = llm.with_structured_output(EntityExtraction)

# ═════════════════════════════════════════════════════════════════════════════
# 2. HELPER UTILITIES
# ═════════════════════════════════════════════════════════════════════════════

def _normalize_name(name: str) -> str:
    return name.upper().strip().replace(" ", "_").replace("-", "_").replace(".", "")

def _parse_section(text: str, header: str) -> list[str]:
    idx = text.find(header)
    if idx == -1: return []
    
    # Find the next header starting with newlines and dashes
    # This ensures we stop at the next '── SECTION ──'
    next_sec = text.find("\n──", idx)
    section = text[idx : next_sec if next_sec != -1 else len(text)]
    
    # Capture the full line of each bullet point
    return re.findall(r"•\s+(.+)", section)

# ═════════════════════════════════════════════════════════════════════════════
# 3. NODE IMPLEMENTATIONS
# ═════════════════════════════════════════════════════════════════════════════

def classify_intent(state: AgentState) -> dict:
    result = _intent_chain.invoke([{"role": "user", "content": state["user_input"]}])
    return {"intent": result.intent}

def extract_entities(state: AgentState) -> dict:
    result = _entity_chain.invoke([{"role": "user", "content": state["user_input"]}])
    raw_names = result.pokemon_names
    validated_names = []
    all_valid = list(VALID_POKEMON_DB.keys())
    for name in raw_names:
        norm = _normalize_name(name)
        if norm in VALID_POKEMON_DB: validated_names.append(norm)
        else:
            matches = difflib.get_close_matches(norm, all_valid, n=1, cutoff=0.7)
            if matches: validated_names.append(matches[0])
    return {"pokemon_names": validated_names, "query_items": result.query_items, "role": result.role}

def router(state: AgentState):
    intent = state["intent"]
    names = state["pokemon_names"]
    if intent == "items":
        return "recommend_items"
    return "team_optimizer"

def items_tool(state: AgentState) -> dict:
    pokemon_name = state["pokemon_names"][0] if state["pokemon_names"] else None
    if not pokemon_name: return {"tool_result": "No Pokémon found."}
    
    role = state.get("role") or "auto"
    main_result = recommend_items(pokemon_name, role)
    
    from app.tools import get_item_details
    item_details = []
    for item in state.get("query_items", []):
        item_details.append(get_item_details(item))
    
    combined = main_result
    if item_details:
        combined += "\n\n── SPECIFIC ITEM DETAILS ──\n" + "\n".join(item_details)
    return {"tool_result": combined}

def team_tool(state: AgentState) -> dict:
    result = team_optimizer(", ".join(state["pokemon_names"]))
    return {"tool_result": result}

def analyst(state: AgentState) -> dict:
    from app.tools import get_item_details
    import re
    
    raw_data = state.get("tool_result", "")
    # Find all items mentioned in the tool data (UPPER_CASE names)
    item_names = set(re.findall(r"([A-Z][A-Z0-9_]{3,})", raw_data))
    
    # Fetch full specs for all these items
    specs = []
    for name in item_names:
        details = get_item_details(name)
        if "not found" not in details.lower():
            specs.append(details)
    
    specs_text = "\n".join(specs)
    
    # Determine the strategy hierarchy based on Data Quality
    if "Data quality: HIGH" in raw_data:
        instruction = (
            "The BOT META DATA contains real HIGH-ELO validated items. "
            "CONFIRM and EXPLAIN the top items from that data. "
            "Do NOT suggest alternatives or contradict them."
        )
    elif "Data quality: MEDIUM" in raw_data:
        instruction = (
            "Bot data exists but from MEDIUM ELO players. "
            "Use bot items as base, you may suggest 1 mathematical complement. "
            "Be transparent that data is not high-ELO validated."
        )
    else:
        instruction = (
            "No high-ELO data available. Give a theoretical build "
            "and EXPLICITLY WARN the user this is not meta-validated."
        )

    prompt = (
        "You are a master Pokémon Auto Chess strategist. Respond ALWAYS in the same language as the user.\n"
        f"INSTRUCTION: {instruction}\n"
        f"ITEM SPECS (Use to explain reasons): {specs_text}\n\n"
        "RULES:\n"
        "1. ALWAYS keep item names in ENGLISH.\n"
        "2. Use **BOLD** for all items.\n"
        "3. Max 2 short sentences.\n"
        "4. Format: [Build] + [Reason]. STOP."
        f"\n\nUser Question: {state['user_input']}\n"
        f"Bot Meta Data: {raw_data}"
    )
    response = llm.invoke([{"role": "user", "content": prompt}])
    
    # Extract content and handle list format (Gemini specific)
    content = response.content if hasattr(response, 'content') else str(response)
    if isinstance(content, list):
        # Join text parts if it's a list of blocks
        content = " ".join([block.get("text", "") if isinstance(block, dict) else str(block) for block in content])
    
    return {"analysis": content}

def formatter(state: AgentState) -> dict:
    raw = state["tool_result"] or ""
    pokemon_name = state["pokemon_names"][0] if state["pokemon_names"] else "Unknown"
    pdata = VALID_POKEMON_DB.get(pokemon_name, {})
    
    # Simple extraction for the display
    role_match = re.search(r"Role detected: ([\w_]+)", raw)
    role_display = role_match.group(1).replace("_", " ").upper() if role_match else "AUTO"
    
    quality_match = re.search(r"Data quality: (\w+)", raw)
    quality = quality_match.group(1) if quality_match else "LOW"
    quality_map = {"HIGH": "HIGH ✅", "MEDIUM": "MEDIUM ⚠️", "LOW": "LOW ❓"}
    quality_display = quality_map.get(quality, "LOW ❓")

    stats = f"HP: {pdata.get('hp', '?')} | DEF: {pdata.get('def', '?')} | Range: {pdata.get('range', '?')}"
    
    lines = []
    if state.get("analysis"):
        lines.append(state["analysis"])
        lines.append("")
        
    lines += [
        f"**{pokemon_name.replace('_', ' ').title()}**",
        f"Rol: {role_display} | {stats}",
        f"**Confidence: {quality_display}**",
        "",
        "**FROM BOTS (REAL DATA)**"
    ]
    
    # Extract items for formatting (only from BOTS section)
    bot_items = _parse_section(raw, "FROM BOTS")
    if bot_items:
        for it in bot_items[:3]: 
            lines.append(f"• {it.strip()}")
    else:
        lines.append("• No data available from bots.")
    
    return {"response": "\n".join(lines)}

# ═════════════════════════════════════════════════════════════════════════════
# 4. GRAPH CONSTRUCTION
# ═════════════════════════════════════════════════════════════════════════════

def build_graph() -> StateGraph:
    workflow = StateGraph(AgentState)
    workflow.add_node("classify_intent", classify_intent)
    workflow.add_node("extract_entities", extract_entities)
    workflow.add_node("recommend_items", items_tool)
    workflow.add_node("team_optimizer", team_tool)
    workflow.add_node("analyst", analyst)
    workflow.add_node("formatter", formatter)

    workflow.add_edge(START, "classify_intent")
    workflow.add_edge("classify_intent", "extract_entities")
    workflow.add_conditional_edges("extract_entities", router)
    workflow.add_edge("recommend_items", "analyst")
    workflow.add_edge("team_optimizer", "analyst")
    workflow.add_edge("analyst", "formatter")
    workflow.add_edge("formatter", END)
    return workflow.compile()

graph = build_graph()