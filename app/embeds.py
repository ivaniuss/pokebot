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
_REPO = Path(__file__).parent.parent.parent / "pokemonAutoChess" / "app" / "public" / "src" / "assets"
ITEMS_DIR   = _REPO / "item{tps}"
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

def _make_item_strip(
    bot_items: list[str], 
    ai_items: list[str], 
    icon_size: int = 64
) -> Optional[io.BytesIO]:
    """Stitch item icons with a gap between Bot and AI groups."""
    bot_imgs = []
    for name in bot_items[:3]:
        p = _item_path(name)
        if p:
            try:
                bot_imgs.append(Image.open(p).convert("RGBA").resize((icon_size, icon_size)))
            except Exception: pass

    ai_imgs = []
    for name in ai_items[:3]:
        p = _item_path(name)
        if p:
            try:
                ai_imgs.append(Image.open(p).convert("RGBA").resize((icon_size, icon_size)))
            except Exception: pass

    if not bot_imgs and not ai_imgs:
        return None

    padding = 8
    group_gap = 32  # Large gap between Bot and AI items
    
    total_w = (len(bot_imgs) + len(ai_imgs)) * icon_size
    total_w += (max(0, len(bot_imgs) - 1) + max(0, len(ai_imgs) - 1)) * padding
    if bot_imgs and ai_imgs:
        total_w += group_gap

    strip = Image.new("RGBA", (total_w, icon_size), (0, 0, 0, 0))
    
    current_x = 0
    # Paste bot items
    for img in bot_imgs:
        strip.paste(img, (current_x, 0), img)
        current_x += icon_size + padding
    
    # Add gap
    if bot_imgs and ai_imgs:
        current_x = current_x - padding + group_gap
    
    # Paste AI items
    for img in ai_imgs:
        strip.paste(img, (current_x, 0), img)
        current_x += icon_size + padding

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
    
    # Start looking for the NEXT section after the current header's line
    # We skip at least 10 chars to avoid finding the closing '──' of the current header
    search_start = idx + len(section_header) + 5
    next_section = text.find("──", search_start)
    
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

    # Item strip — only include top 3 bot items
    bot_items_raw = _extract_section_items(tool_result, "FROM BOTS")
    bot_strip = bot_items_raw[:3]
    
    # Empty list for AI items as we want to hide them from the strip
    ai_strip = []

    strip_buf = _make_item_strip(bot_strip, ai_strip)
    if strip_buf:
        f = discord.File(strip_buf, filename="items.png")
        files.append(f)
        embed.set_image(url="attachment://items.png")

    return embed, files


def _build_team_embed(
    state: dict,
    response: str,
    tool_result: str,
    pokemon_names: list[str],
) -> tuple[discord.Embed, list[discord.File]]:
    """Build an embed for team optimization using structured state."""
    all_types: list[str] = []
    for name in pokemon_names:
        pdata = POKEMON.get(name, {})
        all_types.extend(pdata.get("types", []))

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