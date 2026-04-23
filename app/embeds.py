"""
app/embeds.py
Builds Discord Embeds from the LangGraph agent state.
Uses structured state data (intent, pokemon_names, tool_result) instead of
parsing raw text.
"""

import io
import re
from pathlib import Path
from typing import Optional

import discord
from PIL import Image

# ── Asset paths ─────────────────────────────────────────────────────────────
_REPO = Path(__file__).parent.parent / "pokemonAutoChess" / "app" / "public" / "src" / "assets"
ITEMS_DIR   = _REPO / "items"
POKEMON_DIR = _REPO / "pokemons"

# ── Knowledge base ────────────────────────────────────────────────────────────────
import json
_KB_PATH = Path(__file__).parent.parent / "knowledge_base" / "knowledge_base.json"
with open(_KB_PATH, encoding="utf-8") as _f:
    _KB = json.load(_f)
POKEMON   = _KB["pokemon"]
SYNERGIES = _KB["synergies"]

# ── Synergy colors ────────────────────────────────────────────────────────────────
SYNERGY_COLORS: dict[str, int] = {
    "NORMAL":     0xFEFEFE, "FLYING":    0xB2E9FF, "FIELD":     0xDE8A4E,
    "DARK":       0xA6A6A6, "GROUND":    0xC6964A, "PSYCHIC":   0xB955D2,
    "GRASS":      0x17B300, "BUG":       0xFFFE66, "WATER":     0x2DA2FD,
    "AQUATIC":    0x14C8C8, "POISON":    0x88D7A0, "FAIRY":     0xFFAFD1,
    "FIGHTING":   0xF33218, "FIRE":      0xFF9024, "GHOST":     0x876DAD,
    "ROCK":       0xE7E5AF, "MONSTER":   0x00B464, "AMORPHOUS": 0xE5B2F4,
    "WILD":       0xB22334, "SOUND":     0xFF6095, "FLORA":     0xFF60F1,
    "STEEL":      0xDBDBDB, "ELECTRIC":  0xFDFF4A, "ICE":       0xC3E4EE,
    "BABY":       0xFFD79A, "HUMAN":     0xFDBB8B, "DRAGON":    0xB87333,
    "LIGHT":      0xFFF896, "GOURMET":   0xFF8473, "FOSSIL":    0xD2D35B,
    "ARTIFICIAL": 0xEDEDED,
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def _item_path(name: str) -> Optional[Path]:
    p = ITEMS_DIR / f"{name}.png"
    return p if p.exists() else None

def _pokemon_path(name: str) -> Optional[Path]:
    p = POKEMON.get(name)
    if not p:
        return None
    idx = p["index"]
    path = POKEMON_DIR / f"{idx}.png"
    return path if path.exists() else None

def _make_item_strip(item_names: list[str], icon_size: int = 64) -> Optional[io.BytesIO]:
    """Stitch up to 3 item icons horizontally."""
    images = []
    for name in item_names[:3]:
        p = _item_path(name)
        if p:
            try:
                images.append(Image.open(p).convert("RGBA").resize((icon_size, icon_size)))
            except Exception:
                pass

    if not images:
        return None

    padding = 8
    total_w  = len(images) * icon_size + (len(images) - 1) * padding
    total_h  = icon_size
    strip    = Image.new("RGBA", (total_w, total_h), (0, 0, 0, 0))

    for i, img in enumerate(images):
        strip.paste(img, (i * (icon_size + padding), 0), img)

    buf = io.BytesIO()
    strip.save(buf, format="PNG")
    buf.seek(0)
    return buf


def _extract_names(text: str, lookup: dict) -> list[str]:
    """Find known SCREAMING_SNAKE_CASE names in text."""
    found = []
    for token in re.findall(r"[A-Z][A-Z0-9_]{2,}", text):
        if token in lookup and token not in found:
            found.append(token)
    return found


def _extract_section_items(text: str, section_header: str) -> list[str]:
    """Extract item names that appear after a specific section header."""
    idx = text.find(section_header)
    if idx == -1:
        return []
    next_section = text.find("──", idx + len(section_header))
    section_text = text[idx: next_section if next_section > idx else len(text)]
    return _extract_names(section_text, _KB["items"])


def _clean_text(text: str, max_len: int = 2048) -> str:
    """Trim text to Discord embed description limit."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


# ── State-driven embed builder ────────────────────────────────────────────────

def build_embed_from_state(
    state: dict,
) -> tuple[Optional[discord.Embed], list[discord.File]]:
    """
    Build the correct Discord embed from the agent's final state.

    Reads:
    - state["intent"]        → decides embed type
    - state["pokemon_names"] → Pokémon data lookup
    - state["tool_result"]   → raw tool output for extracting item names
    - state["response"]      → formatted display text
    - state["role"]          → role label (items only)
    """
    intent = state.get("intent")
    response = state.get("response", "")
    tool_result = state.get("tool_result", "")
    pokemon_names = state.get("pokemon_names", [])

    if intent == "items" and pokemon_names:
        return _build_item_embed(state, response, tool_result, pokemon_names)

    if intent == "team":
        return _build_team_embed(state, response, tool_result, pokemon_names)

    # Fallback: generic embed
    return _build_generic_embed(response), []


def _build_item_embed(
    state: dict,
    response: str,
    tool_result: str,
    pokemon_names: list[str],
) -> tuple[discord.Embed, list[discord.File]]:
    """Build an embed for item recommendations using structured state."""
    pokemon_name = pokemon_names[0]
    pdata = POKEMON.get(pokemon_name, {})
    types = pdata.get("types", [])
    color = SYNERGY_COLORS.get(types[0], 0x7289DA) if types else 0x7289DA

    embed = discord.Embed(
        title=f"Items for {pokemon_name.replace('_', ' ').title()}",
        description=_clean_text(response),
        color=color,
    )

    files: list[discord.File] = []

    # Pokémon thumbnail
    poke_path = _pokemon_path(pokemon_name)
    if poke_path:
        f = discord.File(poke_path, filename="pokemon.png")
        files.append(f)
        embed.set_thumbnail(url="attachment://pokemon.png")

    # Item strip — extract item names from the raw tool output
    # The RAW tool output uses "FROM BOTS" and "RECOMMENDED FOR THIS ROLE"
    recommended_items = _extract_section_items(tool_result, "RECOMMENDED FOR THIS ROLE")
    bot_items = _extract_section_items(tool_result, "FROM BOTS")

    strip_items: list[str] = []
    if bot_items:
        strip_items += bot_items[:1]
    if recommended_items:
        strip_items += recommended_items[:2]
    
    # Fallback to the formatted response if raw didn't work
    if not strip_items:
        strip_items = _extract_names(response, _KB["items"])[:3]

    strip_buf = _make_item_strip(strip_items)
    if strip_buf:
        f = discord.File(strip_buf, filename="items.png")
        files.append(f)
        embed.set_image(url="attachment://items.png")

    if strip_items:
        embed.set_footer(text="  ·  ".join(strip_items[:3]))

    return embed, files


def _build_team_embed(
    state: dict,
    response: str,
    tool_result: str,
    pokemon_names: list[str],
) -> tuple[discord.Embed, list[discord.File]]:
    """Build an embed for team optimization using structured state."""
    # Find the dominant synergy for color
    all_types: list[str] = []
    for name in pokemon_names:
        pdata = POKEMON.get(name, {})
        all_types.extend(pdata.get("types", []))

    # Count synergy frequency to pick dominant color
    type_counts: dict[str, int] = {}
    for t in all_types:
        type_counts[t] = type_counts.get(t, 0) + 1
    dominant = max(type_counts, key=type_counts.get) if type_counts else None
    color = SYNERGY_COLORS.get(dominant, 0x7289DA) if dominant else 0x7289DA

    embed = discord.Embed(
        title="Team Optimization",
        description=_clean_text(response),
        color=color,
    )

    if pokemon_names:
        display = ", ".join(n.replace("_", " ").title() for n in pokemon_names[:8])
        embed.add_field(name="Your Pool", value=display, inline=False)

    return embed, []


def _build_generic_embed(response_text: str) -> discord.Embed:
    """Fallback embed for responses that don't match a specific pattern."""
    return discord.Embed(
        description=_clean_text(response_text),
        color=0x7289DA,
    )


# ── Legacy: text-based response parser (kept for API/CLI compatibility) ──────

def parse_response(response_text: str) -> tuple[Optional[discord.Embed], list[discord.File]]:
    """Inspect the response text and build the best embed for it."""
    text = response_text.upper()

    item_names = _extract_names(response_text, _KB["items"])
    poke_names = _extract_names(response_text, POKEMON)

    is_item_response    = len(item_names) >= 2
    is_synergy_response = any(kw in text for kw in ["SYNERGY", "THRESHOLD", "ACTIVE", "SINERGIA"])

    if is_item_response and poke_names:
        recommended_items = _extract_section_items(response_text, "RECOMMENDED FOR THIS ROLE")
        bot_items        = _extract_section_items(response_text, "FROM BOTS")

        strip_items = []
        if bot_items:
            strip_items += bot_items[:1]
        if recommended_items:
            strip_items += recommended_items[:2]
        if not strip_items:
            strip_items = item_names[:3]

        embed, files = _build_item_embed_from_text(
            response_text,
            pokemon_name=poke_names[0],
            item_names=strip_items[:3],
        )
        return embed, files

    if is_synergy_response:
        all_syns   = set(SYNERGIES.keys())
        found_syns = [s for s in all_syns if s in text]
        split_idx  = text.find("CLOSE") if "CLOSE" in text else len(text)
        active = [s for s in found_syns if text.index(s) < split_idx]
        close  = [s for s in found_syns if text.index(s) >= split_idx]
        if found_syns:
            embed, files = _build_synergy_embed_from_text(response_text, active, close)
            return embed, files

    return None, []


def _build_item_embed_from_text(
    response_text: str,
    pokemon_name: str,
    item_names: list[str],
) -> tuple[discord.Embed, list[discord.File]]:
    """Build an embed for item recommendations from text."""
    pdata   = POKEMON.get(pokemon_name, {})
    types   = pdata.get("types", [])
    color   = SYNERGY_COLORS.get(types[0], 0x7289DA) if types else 0x7289DA

    embed = discord.Embed(
        title=f"Items for {pokemon_name.replace('_', ' ').title()}",
        description=_clean_text(response_text),
        color=color,
    )

    if pdata:
        embed.add_field(name="Types", value=" · ".join(f"`{t}`" for t in types), inline=True)
        embed.add_field(name="Cost", value=f"{pdata.get('cost', '?')}g", inline=True)
        embed.add_field(
            name="Role hint",
            value=f"Range {pdata.get('range','?')} | HP {pdata.get('hp','?')}",
            inline=True,
        )

    files: list[discord.File] = []

    poke_path = _pokemon_path(pokemon_name)
    if poke_path:
        f = discord.File(poke_path, filename="pokemon.png")
        files.append(f)
        embed.set_thumbnail(url="attachment://pokemon.png")

    strip_buf = _make_item_strip(item_names)
    if strip_buf:
        f = discord.File(strip_buf, filename="items.png")
        files.append(f)
        embed.set_image(url="attachment://items.png")

    if item_names:
        embed.set_footer(text="  ·  ".join(item_names[:3]))

    return embed, files


def _build_synergy_embed_from_text(
    response_text: str,
    active_synergies: list[str],
    close_synergies: list[str],
) -> tuple[discord.Embed, list[discord.File]]:
    """Build an embed for synergy analysis from text."""
    main_syn = active_synergies[0] if active_synergies else (close_synergies[0] if close_synergies else None)
    color    = SYNERGY_COLORS.get(main_syn, 0x7289DA) if main_syn else 0x7289DA

    embed = discord.Embed(
        title="Synergy Analysis",
        description=_clean_text(response_text),
        color=color,
    )

    if active_synergies:
        embed.add_field(
            name="Active",
            value="\n".join(f"`{s}`" for s in active_synergies[:6]),
            inline=True,
        )
    if close_synergies:
        embed.add_field(
            name="Almost there",
            value="\n".join(f"`{s}`" for s in close_synergies[:6]),
            inline=True,
        )

    return embed, []