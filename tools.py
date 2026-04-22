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
    # "auto" — balanced
    return (s["ap"] + s["atk"] * 2 + s["hp"] + s["def"] + s["speDef"]) * 1.0

def _infer_role(p: dict) -> str:
    """Infer a Pokémon's likely role from its stats."""
    if p["range"] >= 3 and p["speDef"] > p["def"]:
        return "carry_ap"
    if p["range"] >= 2 and p["atk"] > 20:
        return "carry_atk"
    if p["hp"] >= 200 and p["def"] >= 10:
        return "tank"
    return "carry_ap"  # safe default for ranged

# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 1 — Pokemon info
# ═══════════════════════════════════════════════════════════════════════════════

def get_pokemon_info(name: str) -> str:
    """
    Get detailed information about a specific Pokémon in Pokémon Auto Chess.
    Returns its types/synergies, stats, cost, skill (ability), and evolution chain.
    Use this when the player asks about a specific Pokémon.

    Args:
        name: Pokémon name in SCREAMING_SNAKE_CASE (e.g. GARDEVOIR, PIKACHU, CHARIZARD).
    """
    key = name.upper().replace(" ", "_").replace("-", "_")

    # Fuzzy fallback: substring match
    if key not in POKEMON:
        matches = [k for k in POKEMON if key in k]
        if not matches:
            return f"Pokémon '{name}' not found. Try a name like PIKACHU, GARDEVOIR, CHARIZARD."
        key = matches[0]

    p = POKEMON[key]

    # Build evolution chain
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

def recommend_items(pokemon_name: str, role: str = "auto") -> str:
    """
    Recommend the best items to equip on a specific Pokémon in Pokémon Auto Chess.
    Each Pokémon can hold up to 3 items.

    Args:
        pokemon_name: Name of the Pokémon in SCREAMING_SNAKE_CASE (e.g. GARDEVOIR).
        role: The Pokémon's intended role. One of:
              - "auto"      → inferred from stats (default)
              - "carry_ap"  → ability-power damage dealer
              - "carry_atk" → physical attack damage dealer
              - "tank"      → frontline with high HP/DEF
              - "support"   → utility, mana, shields
    """
    key = pokemon_name.upper().replace(" ", "_").replace("-", "_")

    if key not in POKEMON:
        matches = [k for k in POKEMON if key in k]
        if not matches:
            return f"Pokémon '{pokemon_name}' not found."
        key = matches[0]

    p = POKEMON[key]
    effective_role = _infer_role(p) if role == "auto" else role

    # Score all items
    scored = sorted(
        ITEMS.items(),
        key=lambda kv: _score_item_for_role(kv[1], effective_role),
        reverse=True,
    )

    top3 = scored[:3]
    top10 = scored[:10]

    lines = [
        f"=== Item recommendations for {p['name']} ===",
        f"Detected role: {effective_role}",
        f"Types: {', '.join(p['types'])}",
        "",
        "── TOP 3 (equip these) ──",
    ]
    for item_name, item in top3:
        active = {k: v for k, v in item["stats"].items() if v != 0}
        stats_str = "  ".join(f"+{v} {k}" for k, v in active.items())
        lines.append(f"  • {item_name}: {stats_str or 'special effect'}")

    lines += ["", "── Honorable mentions ──"]
    for item_name, item in top10[3:]:
        active = {k: v for k, v in item["stats"].items() if v != 0}
        stats_str = "  ".join(f"+{v} {k}" for k, v in active.items() if v > 0)
        lines.append(f"  • {item_name}: {stats_str or 'special effect'}")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 3 — Synergy advisor
# ═══════════════════════════════════════════════════════════════════════════════

def synergy_advisor(team: str) -> str:
    """
    Analyze the current team composition and advise on synergies.
    Shows which synergies are active, which are close to the next threshold,
    and which cheap Pokémon would complete or upgrade a synergy.

    Args:
        team: Comma-separated list of Pokémon names currently on the board.
              Example: "RALTS, KIRLIA, ABRA, PIKACHU"
    """
    names = [n.strip().upper().replace(" ", "_").replace("-", "_") for n in team.split(",")]

    # Resolve names
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

    # Count synergy contributions
    synergy_count: dict[str, int] = {}
    for name in resolved:
        for t in POKEMON[name]["types"]:
            synergy_count[t] = synergy_count.get(t, 0) + 1

    lines = [f"=== Synergy analysis for: {', '.join(resolved)} ===", ""]

    if not_found:
        lines.append(f"⚠ Not found (skipped): {', '.join(not_found)}\n")

    # Classify synergies
    active   = []
    close    = []  # 1 away from next threshold
    inactive = []

    for syn_name, count in sorted(synergy_count.items(), key=lambda x: -x[1]):
        syn = SYNERGIES.get(syn_name)
        if not syn:
            continue
        thresholds = syn["thresholds"]
        effects    = syn["effects_per_level"]

        # Find current level and next threshold
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

    # Active synergies
    lines.append("✅ ACTIVE SYNERGIES:")
    if active:
        for syn_name, count, level, effect, next_t, next_eff in active:
            upgrade = f"  →  +{next_t - count} for {next_eff}" if next_t else "  (MAX)"
            lines.append(f"  {syn_name} [{count}/{SYNERGIES[syn_name]['thresholds'][-1]}]  "
                         f"Level {level} ({effect}){upgrade}")
    else:
        lines.append("  None active yet.")

    # Close to activation
    lines.append("\n🔶 CLOSE (1-2 Pokémon away):")
    if close:
        for syn_name, count, next_t, next_eff in close:
            needed = next_t - count
            # Suggest cheap Pokémon
            candidates = [
                pname for pname in SYNERGIES[syn_name]["pokemon"]
                if pname not in resolved and not POKEMON[pname]["additional"]
            ][:5]
            lines.append(f"  {syn_name}: {count}/{next_t}  (+{needed} → {next_eff})")
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
    Use this when the player wants to know which Pokémon to buy from the shop.

    Args:
        available: Comma-separated list of Pokémon available in the shop or on bench.
                   Example: "PIKACHU, RAICHU, JOLTEON, GARDEVOIR, RALTS"
        budget: Gold available to spend (default 50). Used to filter by cost.
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

    # Filter by budget
    affordable = [n for n in pool if (POKEMON[n]["cost"] or 99) <= budget]

    if not affordable:
        return f"No Pokémon affordable with {budget}g budget."

    # Find which synergy groups we can contribute to
    synergy_candidates: dict[str, list] = {}
    for name in affordable:
        for t in POKEMON[name]["types"]:
            synergy_candidates.setdefault(t, []).append(name)

    # Score each synergy by how many Pokémon we have towards first threshold
    lines = [f"=== Team optimizer  |  Pool: {len(affordable)} Pokémon  |  Budget: {budget}g ===", ""]

    lines.append("📊 SYNERGY POTENTIAL IN YOUR POOL:")
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
            f"→ {effect}"
        )
        lines.append(f"    Pokémon: {', '.join(members[:6])}")

    # Recommend a core team (greedy: pick the synergy with most coverage, add those Pokémon)
    lines.append("\n🏆 RECOMMENDED CORE TEAM:")
    if scored_syns:
        best_syn, best_members, best_thresholds, _ = scored_syns[0]
        core = best_members[: best_thresholds[0]]
        total_cost = sum(POKEMON[n]["cost"] or 0 for n in core)
        lines.append(f"  Focus: {best_syn} synergy (needs {best_thresholds[0]})")
        lines.append(f"  Core:  {', '.join(core)}")
        lines.append(f"  Cost:  {total_cost}g")

        # Fill remaining slots with next best synergy
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