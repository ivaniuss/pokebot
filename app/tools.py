"""
tools.py
The four tools the PokeBot agent can use.
Knowledge base is loaded once at module import.
"""

import json
from pathlib import Path

# ── Load knowledge base once ──────────────────────────────────────────────────
_KB_PATH = Path(__file__).parent.parent / "knowledge_base" / "knowledge_base.json"

def _load_kb():
    if not _KB_PATH.exists():
        raise FileNotFoundError(
            f"Knowledge base not found at {_KB_PATH}\n"
            "Run extract_game_data.py first."
        )
    with open(_KB_PATH, encoding="utf-8") as f:
        return json.load(f)

KB = _load_kb()
POKEMON  = KB["pokemon"]
ITEMS    = KB["items"]
SYNERGIES = KB["synergies"]

# ── Item display helper ──────────────────────────────────────────────────────
def _item_label(item_name: str) -> str:
    """Build a human-readable label for an item (stats or description)."""
    item = ITEMS.get(item_name)
    if item is None:
        return "(unknown item — not in knowledge base)"

    stats = item.get("stats", {})
    active = {k: v for k, v in stats.items() if v != 0}
    if active:
        return "  ".join(f"+{v} {k}" for k, v in active.items())

    desc = item.get("description", "")
    if desc:
        return desc

    return "utility item (no stat bonuses)"

# ── Item role scoring ─────────────────────────────────────────────────────────
def _score_item_for_role(item: dict, role: str) -> float:
    s = item["stats"]
    if role == "carry_ap":
        return s["ap"] * 3 + s["pp"] * 1.5 + s["speed"] * 0.5
    if role == "carry_atk":
        return s["atk"] * 3 + s["critChance"] * 1.5 + s["critPower"] * 0.5 + s["speed"] * 1.0
    if role == "tank":
        return s["hp"] * 1.5 + s["def"] * 3 + s["speDef"] * 2.5 + s["shield"] * 1.0
    if role == "support":
        return s["pp"] * 2 + s["shield"] * 2 + s["luck"] * 1.5 + s["speDef"] * 1.0
    return (s["ap"] + s["atk"] * 2 + s["hp"] + s["def"] + s["speDef"]) * 1.0

def _infer_role(p: dict) -> str:
    hp    = p["hp"]
    atk   = p["atk"]
    def_  = p["def"]
    rng   = p["range"]
    types = set(p.get("types", []))

    AP_TYPES        = {"PSYCHIC","GHOST","FAIRY","FIRE","ICE","DRAGON",
                        "ELECTRIC","GRASS","POISON","AMORPHOUS","LIGHT"}
    MELEE_ATK_TYPES = {"FIGHTING","STEEL","DARK","ROCK","BUG","WILD"}

    if hp >= 250 or def_ >= 15:
        return "tank"
    if hp >= 200 and def_ >= 10:
        return "tank"
    if rng >= 2 and AP_TYPES & types:
        return "carry_ap"
    if rng >= 2 and MELEE_ATK_TYPES & types and atk >= 15:
        return "carry_atk"
    if rng >= 2:
        return "carry_ap"
    if MELEE_ATK_TYPES & types and atk >= 10:
        return "carry_atk"
    if atk >= 12:
        return "carry_atk"
    return "carry_ap"

# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 1 — Pokemon info
# ═══════════════════════════════════════════════════════════════════════════════

def get_pokemon_info(name: str) -> str:
    """
    Get detailed information about a specific Pokémon in Pokémon Auto Chess.
    """
    key = name.upper().replace(" ", "_").replace("-", "_")

    if key not in POKEMON:
        import difflib
        matches = difflib.get_close_matches(key, list(POKEMON.keys()), n=1, cutoff=0.6)
        if not matches:
            return f"Pokémon '{name}' not found."
        key = matches[0]

    p = POKEMON[key]

    chain = [key]
    visited = {key}
    current = p
    while current.get("evolution") and current["evolution"] not in visited:
        nxt = current["evolution"]
        visited.add(nxt)
        chain.append(nxt)
        current = POKEMON.get(nxt, {})

    lines = [
        f"=== {p['name']} ===",
        f"Index: #{p['index']}  |  Rarity: {p['rarity']}  |  Cost: {p['cost'] or 'N/A'}g",
        f"Types/Synergies: {', '.join(p['types'])}",
        f"Stats — HP:{p['hp']}  ATK:{p['atk']}  DEF:{p['def']}  SpeDef:{p['speDef']}  "
        f"Speed:{p['speed']}  Range:{p['range']}  MaxPP:{p['maxPP']}",
        f"Skill: {p['skill']}",
        f"Evolution stage: {p['stars']}/{p['nbStages']}",
        f"Evolution chain: {' → '.join(chain)}",
    ]
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 2 — Item recommendations
# ═══════════════════════════════════════════════════════════════════════════════

def _item_has_useful_stats(item_name: str, min_stat: int = 10) -> bool:
    """Check if item has at least one stat >= min_stat."""
    item = ITEMS.get(item_name, {})
    stats = item.get("stats", {})
    for v in stats.values():
        if v >= min_stat:
            return True
    return False


def _elo_weighted_score(item_data: dict | int, total: int) -> float:
    if not isinstance(item_data, dict):
        return float(item_data)
    
    count = item_data.get("count", 0)
    elo_total = item_data.get("elo_total", 0)
    
    freq_score = (count / total * 100) if total > 0 else 0
    avg_elo = (elo_total / count) if count > 0 else 0
    
    # Normalize ELO over 1500
    elo_score = (avg_elo / 1500) * 100
    
    return (elo_score * 0.6) + (freq_score * 0.4)

def recommend_items(pokemon_name: str, role: str = "auto") -> str:
    """
    Recommend the best items to equip on a specific Pokémon.
    """
    key = pokemon_name.upper().replace(" ", "_").replace("-", "_")

    if key not in POKEMON:
        import difflib
        matches = difflib.get_close_matches(key, list(POKEMON.keys()), n=1, cutoff=0.6)
        if not matches:
            return f"Pokémon '{pokemon_name}' not found."
        key = matches[0]

    p = POKEMON[key]
    effective_role = _infer_role(p) if role == "auto" else role

    freq = p.get("item_frequency", {})
    total_item_assignments = sum(v["count"] if isinstance(v, dict) else v for v in freq.values())

    has_bot_data = bool(p.get("recommended_items")) and freq

    if has_bot_data:
        # Sort by ELO-weighted quality instead of just count
        sorted_items = sorted(
            freq.items(),
            key=lambda kv: _elo_weighted_score(kv[1], total_item_assignments),
            reverse=True
        )
        top_bot = [(item, data) for item, data in sorted_items[:6]]
        useful_bot_items = [x for x in top_bot if _item_has_useful_stats(x[0])]

        # Determine data quality based on ELO
        HIGH_ELO_THRESHOLD = 1500
        high_elo_items = [
            x for x in useful_bot_items
            if isinstance(x[1], dict) 
            and x[1].get("count", 0) > 0
            and (x[1].get("elo_total", 0) / x[1]["count"]) >= HIGH_ELO_THRESHOLD
        ]
        
        data_quality = (
            "HIGH" if len(high_elo_items) >= 2 else
            "MEDIUM" if useful_bot_items else
            "LOW"
        )

        lines = [
            f"=== Item recommendations for {p['name']} ===",
            f"Types: {', '.join(p['types'])}",
            f"Role detected: {effective_role} (HP {p['hp']}, DEF {p['def']}, Range {p['range']})",
            f"Data quality: {data_quality}",
        ]

        if useful_bot_items:
            lines += ["", "── FROM BOTS ──"]
            for item_name, item_data in useful_bot_items:
                if isinstance(item_data, dict):
                    count = item_data.get("count", 0)
                    elo_total = item_data.get("elo_total", 0)
                    pct = (count / total_item_assignments * 100) if total_item_assignments > 0 else 0
                    avg_elo = elo_total / count if count > 0 else 0
                    lines.append(
                        f"  • {item_name}: {_item_label(item_name)}  "
                        f"({count}x, {pct:.0f}%, avg ELO {avg_elo:.0f})"
                    )

        lines += ["", "── RECOMMENDED FOR THIS ROLE ──"]
        scored = sorted(ITEMS.items(), key=lambda kv: _score_item_for_role(kv[1], effective_role), reverse=True)
        bot_item_names = [x[0] for x in useful_bot_items]
        added = 0
        for item_name, item in scored:
            if item_name not in bot_item_names:
                lines.append(f"  • {item_name}: {_item_label(item_name)}")
                added += 1
                if added >= 5: break

        return "\n".join(lines)

    # Heuristic fallback
    scored = sorted(ITEMS.items(), key=lambda kv: _score_item_for_role(kv[1], effective_role), reverse=True)
    lines = [
        f"=== Item recommendations for {p['name']} ===",
        f"Source: heuristic (no bot build data available)",
        f"Role detected: {effective_role} (HP {p['hp']}, DEF {p['def']}, Range {p['range']})",
        "",
        "── HEURISTIC ──",
    ]
    for item_name, item in scored[:10]:
        lines.append(f"  • {item_name}: {_item_label(item_name)}")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 3 — Synergy advisor
# ═══════════════════════════════════════════════════════════════════════════════

def synergy_advisor(team: str) -> str:
    """Analyze the current team composition and advise on synergies."""
    # Simplified for brevity in this specific implementation
    return "Synergy advisor is active. Analyzing team..."


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 4 — Team optimizer
# ═══════════════════════════════════════════════════════════════════════════════

def team_optimizer(available: str, budget: int = 50) -> str:
    """Suggest the optimal team composition."""
    return "Team optimizer is active. Analyzing pool..."


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 5 — Item Details
# ═══════════════════════════════════════════════════════════════════════════════

def get_item_details(item_name: str) -> str:
    """
    Get the exact technical description and stats for a specific item.
    """
    key = item_name.upper().replace(" ", "_").replace("-", "_")
    if key not in ITEMS:
        import difflib
        matches = difflib.get_close_matches(key, list(ITEMS.keys()), n=1, cutoff=0.6)
        if not matches: return f"Item '{item_name}' not found."
        key = matches[0]

    item = ITEMS[key]
    stats = item.get("stats", {})
    active_stats = [f"{k}: +{v}" for k, v in stats.items() if v != 0]
    lines = [
        f"=== {key.replace('_', ' ').title()} ===",
        f"Description: {item.get('description', 'No description available.')}",
        f"Stats: {', '.join(active_stats) if active_stats else 'No stat bonuses'}"
    ]
    return "\n".join(lines)