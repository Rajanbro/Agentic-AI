import google.generativeai as genai
from functools import wraps
import re
import ast
import os
from dotenv import load_dotenv

load_dotenv()  

# ========== Config ==========
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

model = genai.GenerativeModel(model_name="models/gemini-2.5-pro")

# ========== Tool Decorator & Registry ==========
TOOL_REGISTRY = {}

def tool(name=None, description=None):
    def decorator(func):
        tool_name = name or func.__name__
        TOOL_REGISTRY[tool_name] = {
            "func": func,
            "description": description or func.__doc__
        }
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator

# ========== Sample Tool ==========
@tool(name="get_weather", description="Returns the current weather in a city.")
def get_weather(city: str) -> str:
    return f"The weather in {city} is sunny with 30°C."
# 
# Optional: another example tool
@tool(name="add_numbers", description="Adds two numbers.")
def add_numbers(a: str, b: str) -> str:
    return str(float(a) + float(b))

# ========== Agent Logic ==========
def run_agent(prompt: str):
    tools_description = "\n".join(
        f"{name}: {info['description']}" for name, info in TOOL_REGISTRY.items()
    )

    system_prompt = (
        "You are a smart assistant with tool-using capability.\n"
        "You can call a tool like this:\n"
        "TOOL_CALL: tool_name(\"arg1\", \"arg2\")\n"
        "Available tools:\n"
        f"{tools_description}\n"
        "Only call a tool if necessary. Otherwise, respond naturally.\n"
    )

    full_prompt = f"{system_prompt}\nUSER: {prompt}"

    try:
        response = model.generate_content([full_prompt])
        content = response.text.strip()
        # print(f"\n[Gemini Output] {content}")
    except Exception as e:
        print(f"[❌ Error] Gemini API failed: {e}")
        return

    # ========== Tool Call Handler ==========
    if content.startswith("TOOL_CALL:"):
        try:
            call_line = content[len("TOOL_CALL:"):].strip()
            tool_name, args_str = call_line.split("(", 1)
            tool_name = tool_name.strip()
            args_str = args_str.rstrip(")").strip()

            args = ast.literal_eval(f"[{args_str}]")  # Safe eval of args

            tool = TOOL_REGISTRY.get(tool_name)
            if tool:
                result = tool["func"](*args)
                print(f"[✅ Tool Result] {result}")
            else:
                print(f"[❌ Error] Tool '{tool_name}' not found.")
        except Exception as e:
            print(f"[❌ Error] Failed to parse or call tool: {e}")
    else:
        print(f"[🤖 Agent Response] {content}")

# ========== Main ==========
if __name__ == "__main__":
    print("=== 🔧 Agentic Gemini AI ===")
    print("Type 'help' to list tools. Type 'exit' or 'quit' to stop.")

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in {"exit", "quit"}:
            print("Agent: Goodbye 👋")
            break
        elif user_input.lower() == "help":
            print("\n📦 Available Tools:")
            for name, info in TOOL_REGISTRY.items():
                print(f" - {name}: {info['description']}")
            continue
        run_agent(user_input)
