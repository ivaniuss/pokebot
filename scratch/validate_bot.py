import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(os.getcwd())

from app.agent import _format_items_response, VALID_POKEMON_DB
from app.embeds import ITEMS_DIR, _item_path

def validate():
    print("--- 1. DATABASE VALIDATION ---")
    print(f"Total Pokémon in DB: {len(VALID_POKEMON_DB)}")
    for pkmn in ["GURDURR", "PIKACHU", "CHARIZARD"]:
        exists = pkmn in VALID_POKEMON_DB
        print(f"Checking {pkmn}: {'✅ FOUND' if exists else '❌ MISSING'}")
        if exists:
            pdata = VALID_POKEMON_DB[pkmn]
            print(f"   Stats: HP={pdata.get('hp')} DEF={pdata.get('def')} Range={pdata.get('range')}")

    print("\n--- 2. FORMATTER VALIDATION (GURDURR) ---")
    # Simulate tool output for Gurdurr
    raw_tool_output = """=== Item recommendations for GURDURR ===
Role detected: tank (HP 280, DEF 12, Range 1)
── FROM BOTS ──
• PROTECTIVE_PADS: +6 atk (2x)
• GREEN_ORB: +15 hp (2x)
── RECOMMENDED FOR THIS ROLE ──
• EVIOLITE: +50 ap
"""
    formatted = _format_items_response(raw_tool_output, ["GURDURR"], "auto")
    print("Formatted Result:")
    print("-" * 20)
    print(formatted)
    print("-" * 20)
    
    if "HP: 280" in formatted and "Rol: TANK" in formatted:
        print("Formatter: ✅ SUCCESS")
    else:
        print("Formatter: ❌ FAILURE (Stats or Role missing)")

    print("\n--- 3. IMAGE PATH VALIDATION ---")
    print(f"Items Directory: {ITEMS_DIR}")
    print(f"Directory exists: {ITEMS_DIR.exists()}")
    
    test_items = ["PROTECTIVE_PADS", "GREEN_ORB", "EVIOLITE"]
    for item in test_items:
        path = _item_path(item)
        print(f"Checking {item}: {'✅ OK' if path else '❌ NOT FOUND'} at {path}")

if __name__ == "__main__":
    validate()
