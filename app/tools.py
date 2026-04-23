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
    Returns its types/synergies, stats, cost, skill (ability), and evolution chain.
    Use this when the player asks about a specific Pokémon.
    """
    key = name.upper().replace(" ", "_").replace("-", "_")

    if key not in POKEMON:
        matches = [k for k in POKEMON if key in k]
        if not matches:
            return f"Pokémon '{name}' not found. Try a name like PIKACHU, GARDEVOIR, CHARIZARD."
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
    if p.get("regional"):
        lines.append("Note: Regional variant")
    if p.get("additional"):
        lines.append("Note: Additional pick (not in main pool)")

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


def recommend_items(pokemon_name: str, role: str = "auto") -> str:
    """
    Recommend the best items to equip on a specific Pokémon in Pokémon Auto Chess.
    Each Pokémon can hold up to 3 items.
    """
    key = pokemon_name.upper().replace(" ", "_").replace("-", "_")

    if key not in POKEMON:
        matches = [k for k in POKEMON if key in k]
        if not matches:
            return f"Pokémon '{pokemon_name}' not found."
        key = matches[0]

    p = POKEMON[key]
    effective_role = _infer_role(p) if role == "auto" else role

    freq = p.get("item_frequency", {})
    total_item_assignments = 0
    for item_data in freq.values():
        if isinstance(item_data, dict):
            total_item_assignments += item_data.get("count", 0)
        else:
            total_item_assignments += item_data

    has_bot_data = bool(p.get("recommended_items")) and freq

    if has_bot_data:
        sorted_items = sorted(
            freq.items(),
            key=lambda kv: (kv[1]["count"] if isinstance(kv[1], dict) else kv[1]),
            reverse=True
        )
        top_bot = [(item, data) for item, data in sorted_items[:6]]
        useful_bot_items = [x for x in top_bot if _item_has_useful_stats(x[0])]

        lines = [
            f"=== Item recommendations for {p['name']} ===",
            f"Types: {', '.join(p['types'])}",
            f"Role detected: {effective_role} (HP {p['hp']}, DEF {p['def']}, Range {p['range']})",
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
        scored = sorted(
            ITEMS.items(),
            key=lambda kv: _score_item_for_role(kv[1], effective_role),
            reverse=True,
        )
        bot_item_names = [x[0] for x in useful_bot_items]
        for item_name, item in scored:
            if item_name not in bot_item_names:
                lines.append(f"  • {item_name}: {_item_label(item_name)}")
                if len(lines) - 8 >= 5:
                    break

        return "\n".join(lines)

    scored = sorted(
        ITEMS.items(),
        key=lambda kv: _score_item_for_role(kv[1], effective_role),
        reverse=True,
    )
    top3 = scored[:3]
    top10 = scored[:10]

    lines = [
        f"=== Item recommendations for {p['name']} ===",
        f"Source: heuristic (no bot build data available)",
        f"Role detected: {effective_role} (HP {p['hp']}, DEF {p['def']}, Range {p['range']})",
        f"Types: {', '.join(p['types'])}",
        "",
        "── HEURISTIC ──",
        "  Based on role: " + effective_role,
    ]
    for item_name, item in top3:
        lines.append(f"  • {item_name}: {_item_label(item_name)}")

    lines += ["", "── Honorable mentions ──"]
    for item_name, item in top10[3:]:
        lines.append(f"  • {item_name}: {_item_label(item_name)}")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 3 — Synergy advisor
# ═══════════════════════════════════════════════════════════════════════════════

def synergy_advisor(team: str) -> str:
    """
    Analyze the current team composition and advise on synergies.
    Shows which synergies are active, which are close to the next threshold,
    and which cheap Pokémon would complete or upgrade a synergy.
    """
    names = [n.strip().upper().replace(" ", "_").replace("-", "_") for n in team.split(",")]

    resolved = []
    not_found = []
    for n in names:
        if n in POKEMON:
            resolved.append(n)
        else:
            matches = [k for k in POKEMON if n in k]
            if matches:
                resolved.append(matches[0])
            else:
                not_found.append(n)

    if not resolved:
        return "No valid Pokémon found. Check the names and try again."

    synergy_count: dict[str, int] = {}
    for name in resolved:
        for t in POKEMON[name]["types"]:
            synergy_count[t] = synergy_count.get(t, 0) + 1

    lines = [f"=== Synergy analysis for: {', '.join(resolved)} ===", ""]

    if not_found:
        lines.append(f"Not found (skipped): {', '.join(not_found)}\n")

    active   = []
    close    = []
    inactive = []

    for syn_name, count in sorted(synergy_count.items(), key=lambda x: -x[1]):
        syn = SYNERGIES.get(syn_name)
        if not syn:
            continue
        thresholds = syn["thresholds"]
        effects    = syn["effects_per_level"]

        current_level = sum(1 for t in thresholds if count >= t)
        current_effect = effects[current_level - 1] if current_level > 0 else None
        next_threshold = next((t for t in thresholds if t > count), None)
        next_effect = effects[current_level] if current_level < len(effects) else None

        if current_level > 0:
            active.append((syn_name, count, current_level, current_effect, next_threshold, next_effect))
        elif next_threshold and next_threshold - count <= 2:
            close.append((syn_name, count, next_threshold, next_effect))
        else:
            inactive.append((syn_name, count, next_threshold))

    lines.append("ACTIVE SYNERGIES:")
    if active:
        for syn_name, count, level, effect, next_t, next_eff in active:
            upgrade = f"  ->  +{next_t - count} for {next_eff}" if next_t else "  (MAX)"
            lines.append(f"  {syn_name} [{count}/{SYNERGIES[syn_name]['thresholds'][-1]}]  "
                         f"Level {level} ({effect}){upgrade}")
    else:
        lines.append("  None active yet.")

    lines.append("\nCLOSE (1-2 away):")
    if close:
        for syn_name, count, next_t, next_eff in close:
            needed = next_t - count
            candidates = [
                pname for pname in SYNERGIES[syn_name]["pokemon"]
                if pname not in resolved and not POKEMON[pname]["additional"]
            ][:5]
            lines.append(f"  {syn_name}: {count}/{next_t}  (+{needed} -> {next_eff})")
            if candidates:
                lines.append(f"    Suggestions: {', '.join(candidates)}")
    else:
        lines.append("  None close.")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 4 — Team optimizer
# ═══════════════════════════════════════════════════════════════════════════════

def team_optimizer(available: str, budget: int = 50) -> str:
    """
    Given a pool of available Pokémon and a gold budget, suggest the optimal
    team composition that maximizes synergy activations.
    """
    names = [n.strip().upper().replace(" ", "_").replace("-", "_") for n in available.split(",")]

    pool = []
    for n in names:
        if n in POKEMON:
            pool.append(n)
        else:
            matches = [k for k in POKEMON if n in k]
            if matches:
                pool.append(matches[0])

    if not pool:
        return "No valid Pokémon in the pool. Check the names."

    affordable = [n for n in pool if (POKEMON[n]["cost"] or 99) <= budget]

    if not affordable:
        return f"No Pokémon affordable with {budget}g budget."

    synergy_candidates: dict[str, list] = {}
    for name in affordable:
        for t in POKEMON[name]["types"]:
            synergy_candidates.setdefault(t, []).append(name)

    lines = [f"=== Team optimizer  |  Pool: {len(affordable)} Pokemon  |  Budget: {budget}g ===", ""]

    lines.append("SYNERGY POTENTIAL IN YOUR POOL:")
    scored_syns = []
    for syn_name, members in synergy_candidates.items():
        syn = SYNERGIES.get(syn_name)
        if not syn:
            continue
        thresholds = syn["thresholds"]
        first_t = thresholds[0]
        coverage = len(members) / first_t
        scored_syns.append((syn_name, members, thresholds, coverage))

    scored_syns.sort(key=lambda x: -x[3])

    for syn_name, members, thresholds, coverage in scored_syns[:6]:
        first_t = thresholds[0]
        bar = "█" * min(len(members), first_t) + "░" * max(0, first_t - len(members))
        effect = SYNERGIES[syn_name]["effects_per_level"][0] if SYNERGIES[syn_name]["effects_per_level"] else "?"
        lines.append(
            f"  {syn_name:12}  [{bar}] {len(members)}/{first_t}  "
            f"-> {effect}"
        )
        lines.append(f"    Pokemon: {', '.join(members[:6])}")

    lines.append("\nRECOMMENDED CORE TEAM:")
    if scored_syns:
        best_syn, best_members, best_thresholds, _ = scored_syns[0]
        core = best_members[: best_thresholds[0]]
        total_cost = sum(POKEMON[n]["cost"] or 0 for n in core)
        lines.append(f"  Focus: {best_syn} synergy (needs {best_thresholds[0]})")
        lines.append(f"  Core:  {', '.join(core)}")
        lines.append(f"  Cost:  {total_cost}g")

        remaining_budget = budget - total_cost
        if scored_syns[1:]:
            second_syn, second_members, second_thresholds, _ = scored_syns[1]
            fillers = [n for n in second_members if n not in core][:3]
            filler_cost = sum(POKEMON[n]["cost"] or 0 for n in fillers)
            if filler_cost <= remaining_budget:
                lines.append(f"  +{second_syn}: {', '.join(fillers)} ({filler_cost}g)")
    else:
        lines.append("  Not enough data to form a recommendation.")

    return "\n".join(lines)