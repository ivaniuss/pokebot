"""
app/cli.py
CLI test mode for the PokeBot LangGraph agent.
"""

import sys
import os

# Add the project root to sys.path so we can import app
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.agent import graph
from dotenv import load_dotenv


def main():
    load_dotenv()

    print("\n" + "=" * 50)
    print("      🔥 POKEBOT CONSOLE TEST MODE 🔥")
    print("=" * 50)
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            user_input = input("👤 You: ").strip()

            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye! 👋")
                break

            if not user_input:
                continue

            print("⏳ PokeBot is thinking...")

            # Run the LangGraph agent
            result = graph.invoke({"user_input": user_input})

            intent = result.get("intent", "?")
            tool = result.get("tool_name", "?")
            names = result.get("pokemon_names", [])
            response = result.get("response", "")

            print(f"\n📊 Intent: {intent} | Tool: {tool} | Pokemon: {names}")
            print(f"\n🤖 PokeBot:\n{response}\n")
            print("-" * 30)

        except KeyboardInterrupt:
            print("\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
