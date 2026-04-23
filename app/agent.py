"""
agent.py
PokeBot — State-driven LangGraph agent for Pokémon Auto Chess.

Architecture:
    classify_intent → extract_entities → router → (tool) → formatter

All routing is deterministic via state. The LLM is only used for
intent classification and entity extraction as a fallback.
The formatter is a pure Python function — no LLM.
"""

from __future__ import annotations

import logging
import re
import difflib
from typing import Literal, Optional

from dotenv import find_dotenv, load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from app.tools import recommend_items, team_optimizer, POKEMON as VALID_POKEMON_DB

load_dotenv(find_dotenv())

logger = logging.getLogger("pokebot.agent")

# ── LLM (only used as fallback for intent / entity extraction) ───────────────

llm = ChatOpenAI(model_name="gpt-5.4-mini", temperature=0)

# ═════════════════════════════════════════════════════════════════════════════
# 1. STATE DEFINITION
# ═════════════════════════════════════════════════════════════════════════════

class AgentState(TypedDict):
    user_input: str
    intent: Optional[Literal["items", "team"]]
    pokemon_names: list[str]
    role: Optional[str]
    tool_name: Optional[str]
    tool_args: Optional[dict]
    tool_result: Optional[str]
    response: Optional[str]


# ═════════════════════════════════════════════════════════════════════════════
# 2. HELPER UTILITIES
# ═════════════════════════════════════════════════════════════════════════════

def _normalize_name(name: str) -> str:
    """Convert any Pokémon name to SCREAMING_SNAKE_CASE."""
    return name.strip().upper().replace(" ", "_").replace("-", "_").replace(".", "")


# ── Pydantic schemas for LLM structured output ──────────────────────────────

from pydantic import BaseModel, Field


class IntentClassification(BaseModel):
    """The classified intent of a user message about Pokémon Auto Chess."""
    intent: Literal["items", "team"] = Field(
        description=(
            "'items' if the user is asking about items, builds, or equipment "
            "for a specific Pokémon. "
            "'team' if the user is asking about team composition, synergies, "
            "or optimization across multiple Pokémon."
        )
    )


class EntityExtraction(BaseModel):
    """Pokémon names and optional role extracted from a user message."""
    pokemon_names: list[str] = Field(
        description=(
            "List of Pokémon names mentioned by the user, each in "
            "SCREAMING_SNAKE_CASE. Examples: GARDEVOIR, MR_MIME, "
            "HISUIAN_TYPHLOSION. Only include actual Pokémon names, "
            "not other words."
        )
    )
    role: Optional[Literal["tank", "carry_ap", "carry_atk", "support"]] = Field(
        default=None,
        description=(
            "The gameplay role the user wants for the Pokémon, if mentioned. "
            "tank = defensive/HP focus, carry_ap = ability power/magic damage, "
            "carry_atk = physical attack damage, support = utility/shields. "
            "null if the user didn't specify a role."
        ),
    )


# ── Bound LLM chains with structured output ─────────────────────────────────

_intent_chain = llm.with_structured_output(IntentClassification)
_entity_chain = llm.with_structured_output(EntityExtraction)

_INTENT_SYSTEM = (
    "You are a classifier for Pokémon Auto Chess queries. "
    "Classify the user's message into exactly one intent:\n"
    "- 'items': the user asks about items, builds, or equipment for a Pokémon\n"
    "- 'team': the user asks about team composition, synergies, or optimization\n"
    "The message can be in any language."
)

_ENTITY_SYSTEM = (
    "You are an expert entity extractor for Pokémon Auto Chess. "
    "Your task is to identify Pokémon names and the requested gameplay role from the user message. "
    "\n\nRULES:"
    "\n1. Extract ALL Pokémon names mentioned."
    "\n2. Convert names to SCREAMING_SNAKE_CASE (e.g., 'Mr. Mime' -> 'MR_MIME', 'Tapu Koko' -> 'TAPU_KOKO')."
    "\n3. Identify the role ONLY if explicitly mentioned or strongly implied (tank, carry_ap, carry_atk, support)."
    "\n4. The message can be in any language (English, Spanish, etc.)."
    "\n5. If no Pokémon are found, return an empty list."
)



# ═════════════════════════════════════════════════════════════════════════════
# 3. NODE IMPLEMENTATIONS
# ═════════════════════════════════════════════════════════════════════════════

def classify_intent(state: AgentState) -> dict:
    """Classify user intent using LLM with structured output."""
    logger.info(f"[classify_intent] Input: {state['user_input'][:100]}")

    result: IntentClassification = _intent_chain.invoke([
        {"role": "system", "content": _INTENT_SYSTEM},
        {"role": "user", "content": state["user_input"]},
    ])

    logger.info(f"[classify_intent] Intent: {result.intent}")
    return {"intent": result.intent}


def extract_entities(state: AgentState) -> dict:
    """Extract Pokémon names and role using LLM, then validate against DB."""
    logger.info(f"[extract_entities] Processing input: {state['user_input'][:50]}...")

    result: EntityExtraction = _entity_chain.invoke([
        {"role": "system", "content": _ENTITY_SYSTEM},
        {"role": "user", "content": state["user_input"]},
    ])

    # Normalize and Validate
    raw_names = result.pokemon_names
    validated_names = []
    all_valid_names = list(VALID_POKEMON_DB.keys())
    
    for name in raw_names:
        norm = _normalize_name(name)
        # 1. Direct match
        if norm in VALID_POKEMON_DB:
            validated_names.append(norm)
        else:
            # 2. Fuzzy match (tolerates typos like 'gudurr')
            matches = difflib.get_close_matches(norm, all_valid_names, n=1, cutoff=0.7)
            if matches:
                validated_names.append(matches[0])
                logger.info(f"[extract_entities] Fuzzy match: {norm} -> {matches[0]}")
            else:
                logger.warning(f"[extract_entities] Pokémon not found in DB: {norm}")

    logger.info(f"[extract_entities] Final Names={validated_names}, Role={result.role}")
    return {"pokemon_names": validated_names, "role": result.role}


def router(state: AgentState) -> dict:
    """
    Decide which tool to call and prepare its arguments.
    Pure state-driven — no LLM.
    """
    intent = state["intent"]
    names = state["pokemon_names"]

    if intent == "items":
        return {
            "tool_name": "recommend_items",
            "tool_args": {
                "pokemon_name": names[0] if names else None,
                "role": state.get("role") or "auto",
            },
        }

    if intent == "team":
        return {
            "tool_name": "team_optimizer",
            "tool_args": {
                "pokemon_names": names,
            },
        }

    # Should never reach here because classify_intent always picks one
    return {
        "tool_name": "recommend_items",
        "tool_args": {"pokemon_name": names[0] if names else "PIKACHU", "role": "auto"},
    }


def items_tool(state: AgentState) -> dict:
    """Call the recommend_items tool."""
    args = state["tool_args"] or {}
    pokemon_name = args.get("pokemon_name")
    role = args.get("role", "auto")

    if not pokemon_name:
        return {"tool_result": "Error: No Pokémon found in your query."}

    logger.info(f"[items_tool] Calling recommend_items({pokemon_name}, {role})")
    result = recommend_items(pokemon_name=pokemon_name, role=role)
    logger.info(f"[items_tool] Result length: {len(result)} chars")
    return {"tool_result": result}


def team_tool(state: AgentState) -> dict:
    """Call the team_optimizer tool."""
    args = state["tool_args"] or {}
    names = args.get("pokemon_names", [])
    csv_names = ", ".join(names)

    logger.info(f"[team_tool] Calling team_optimizer({csv_names[:80]})")
    result = team_optimizer(available=csv_names)
    logger.info(f"[team_tool] Result length: {len(result)} chars")
    return {"tool_result": result}


def formatter(state: AgentState) -> dict:
    """
    Format the final response.
    This is a PURE PYTHON function — no LLM. Output format is strictly
    determined by state['intent'].
    """
    intent = state["intent"]
    result = state["tool_result"] or ""
    names = state["pokemon_names"]

    logger.info(f"[formatter] Formatting for intent={intent}")

    if intent == "items":
        response = _format_items_response(result, names, state.get("role"))
    elif intent == "team":
        response = _format_team_response(result, names)
    else:
        response = result

    return {"response": response}


# ═════════════════════════════════════════════════════════════════════════════
# 4. FORMATTER LOGIC (pure Python, no LLM)
# ═════════════════════════════════════════════════════════════════════════════

def _format_items_response(raw: str, names: list[str], role: Optional[str]) -> str:
    """
    Parse the raw recommend_items output and reformat into a clean
    and detailed description.
    """
    pokemon_name = names[0] if names else "UNKNOWN"
    pdata = VALID_POKEMON_DB.get(pokemon_name, {})
    
    # Extract the actually used role from the tool output
    actual_role = "AUTO"
    role_match = re.search(r"Role detected: ([\w_]+)", raw)
    if role_match:
        actual_role = role_match.group(1).upper()
    
    # Show the specific role, sanitized
    role_display = actual_role.replace("_", " ")

    # Stats string
    stats = f"HP: {pdata.get('hp', '?')} | DEF: {pdata.get('def', '?')} | Range: {pdata.get('range', '?')}"
    
    cost = pdata.get('cost')
    cost_str = f" | Cost: {cost}g" if cost is not None else ""

    # Parse sections
    bot_items = _parse_section(raw, "FROM BOTS")
    rec_items = _parse_section(raw, "AI RECOMMENDED") or _parse_section(raw, "RECOMMENDED FOR THIS ROLE")

    lines = [
        f"**{pokemon_name.replace('_', ' ').title()}**",
        f"Rol: {role_display} | {stats}{cost_str}",
        "",
        "**FROM BOTS (REAL DATA)**"
    ]
    
    if bot_items:
        for item_line in bot_items[:5]:
            lines.append(f"• {item_line}")
    else:
        lines.append("• No data available from bots.")

    if rec_items:
        lines.append("")
        lines.append("**AI RECOMMENDED**")
        for item_line in rec_items[:5]:
            lines.append(f"• {item_line}")

    return "\n".join(lines)


def _format_team_response(raw: str, names: list[str]) -> str:
    """
    Parse the raw team_optimizer output and reformat into the strict
    team format.
    """
    # Parse synergy potential
    synergies = _parse_section(raw, "SYNERGY POTENTIAL IN YOUR POOL:")
    # Parse recommended core team
    core_team = _parse_section(raw, "RECOMMENDED CORE TEAM:")

    lines = ["[TEAM OPTIMIZATION]"]

    if synergies:
        lines.append("")
        lines.append("**CORE SYNERGY**")
        for syn_line in synergies:
            lines.append(f"• {syn_line}")

    if core_team:
        lines.append("")
        lines.append("**RECOMMENDED UNITS**")
        for unit_line in core_team:
            # Try to label ADD/KEEP/REMOVE based on context
            clean = unit_line.strip()
            if clean.startswith("Focus:"):
                lines.append(f"• {clean}")
            elif clean.startswith("Core:"):
                # Split core members and label KEEP
                core_members = clean.replace("Core:", "").strip()
                for member in core_members.split(","):
                    member = member.strip()
                    if member:
                        label = "KEEP" if member in names else "ADD"
                        lines.append(f"• {label}: {member}")
            elif clean.startswith("+"):
                # Secondary synergy additions
                lines.append(f"• ADD: {clean[1:].strip()}")
            else:
                lines.append(f"• {clean}")

    lines.append("")
    lines.append("**NOTES**")
    lines.append("• Focus on completing your strongest synergy first.")
    if len(names) >= 4:
        lines.append("• Consider bench space — prioritize units that activate two synergies.")

    return "\n".join(lines)


def _parse_section(raw: str, header: str) -> list[str]:
    """Extract bullet points / lines from a named section in tool output."""
    idx = raw.find(header)
    if idx == -1:
        # Try case-insensitive
        lower_raw = raw.lower()
        lower_header = header.lower()
        idx = lower_raw.find(lower_header)
        if idx == -1:
            return []

    # Find the start of content after the header
    content_start = raw.find("\n", idx)
    if content_start == -1:
        return []
    content_start += 1

    # Find the end: next section header (── or ===) or end of string
    next_section = len(raw)
    for marker in ["──", "===", "SYNERGY POTENTIAL", "RECOMMENDED CORE"]:
        pos = raw.find(marker, content_start)
        if pos != -1 and pos < next_section:
            next_section = pos

    section_text = raw[content_start:next_section].strip()
    if not section_text:
        return []

    lines = []
    for line in section_text.split("\n"):
        cleaned = line.strip().lstrip("•·-").strip()
        if cleaned:
            lines.append(cleaned)

    return lines


# ═════════════════════════════════════════════════════════════════════════════
# 5. ROUTING FUNCTION (for conditional edges)
# ═════════════════════════════════════════════════════════════════════════════

def route_to_tool(state: AgentState) -> str:
    """Conditional edge: route to the correct tool node based on state."""
    tool = state.get("tool_name")
    logger.info(f"[route_to_tool] Routing to: {tool}")
    if tool == "team_optimizer":
        return "team_tool"
    return "items_tool"


# ═════════════════════════════════════════════════════════════════════════════
# 6. GRAPH WIRING
# ═════════════════════════════════════════════════════════════════════════════

def build_graph() -> StateGraph:
    """Build and compile the PokeBot LangGraph."""
    builder = StateGraph(AgentState)

    # Add nodes
    builder.add_node("classify_intent", classify_intent)
    builder.add_node("extract_entities", extract_entities)
    builder.add_node("router", router)
    builder.add_node("items_tool", items_tool)
    builder.add_node("team_tool", team_tool)
    builder.add_node("formatter", formatter)

    # Edges: START → classify_intent → extract_entities → router
    builder.add_edge(START, "classify_intent")
    builder.add_edge("classify_intent", "extract_entities")
    builder.add_edge("extract_entities", "router")

    # Conditional edge: router → items_tool OR team_tool
    builder.add_conditional_edges(
        "router",
        route_to_tool,
        {"items_tool": "items_tool", "team_tool": "team_tool"},
    )

    # Tools → formatter → END
    builder.add_edge("items_tool", "formatter")
    builder.add_edge("team_tool", "formatter")
    builder.add_edge("formatter", END)

    return builder.compile()


# ── Compiled graph (singleton) ────────────────────────────────────────────────
graph = build_graph()