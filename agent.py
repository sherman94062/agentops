"""Interactive AI agent with tools, memory, and observability via AgentOps."""

import json
import math
import os
from datetime import datetime
from urllib.error import URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import agentops
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

agentops.init(os.getenv("AGENTOPS_API_KEY"))
client = Anthropic()

LOG_FILE = "agent_calls.jsonl"

SYSTEM_PROMPT = """You are a helpful research assistant with access to these tools:
- calculator: evaluate math expressions (supports arithmetic and math functions like sqrt, sin, log)
- get_current_datetime: get the current date and time
- wikipedia_search: search Wikipedia for articles matching a query
- wikipedia_summary: fetch the full summary of a specific Wikipedia article by title

Use tools when they would help answer the question. For Wikipedia lookups, search first if you're unsure of the exact article title, then fetch the summary. Provide clear, concise answers."""

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

def calculator(expression: str) -> str:
    try:
        result = eval(expression, {"__builtins__": {}, "math": math})
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def get_current_datetime() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def wikipedia_search(query: str) -> str:
    try:
        params = f"action=query&list=search&srsearch={quote(query)}&srlimit=5&format=json"
        url = f"https://en.wikipedia.org/w/api.php?{params}"
        req = Request(url, headers={"User-Agent": "AgentOpsDemo/1.0"})
        with urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            results = data.get("query", {}).get("search", [])
            if not results:
                return "No results found."
            lines = []
            for r in results:
                snippet = r.get("snippet", "").replace('<span class="searchmatch">', "").replace("</span>", "")
                lines.append(f"- {r['title']}: {snippet}")
            return "\n".join(lines)
    except URLError as e:
        return f"Error searching Wikipedia: {e}"


def wikipedia_summary(title: str) -> str:
    try:
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(title)}"
        req = Request(url, headers={"User-Agent": "AgentOpsDemo/1.0"})
        with urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            return data.get("extract", "No summary available.")
    except URLError as e:
        return f"Error fetching Wikipedia: {e}"


TOOLS = [
    {
        "name": "calculator",
        "description": "Evaluate a mathematical expression. Supports arithmetic, powers, and math module functions (sqrt, sin, cos, log, pi, e).",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Math expression to evaluate, e.g. '2**10' or 'math.sqrt(144)'"
                }
            },
            "required": ["expression"]
        }
    },
    {
        "name": "get_current_datetime",
        "description": "Get the current date and time.",
        "input_schema": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "wikipedia_search",
        "description": "Search Wikipedia for articles matching a query. Returns up to 5 results with titles and snippets.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query, e.g. 'quantum computing'"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "wikipedia_summary",
        "description": "Fetch the full summary of a specific Wikipedia article by its exact title.",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "The Wikipedia article title, e.g. 'Albert Einstein'"
                }
            },
            "required": ["title"]
        }
    },
]

TOOL_DISPATCH = {
    "calculator": lambda args: calculator(args["expression"]),
    "get_current_datetime": lambda args: get_current_datetime(),
    "wikipedia_search": lambda args: wikipedia_search(args["query"]),
    "wikipedia_summary": lambda args: wikipedia_summary(args["title"]),
}

# ---------------------------------------------------------------------------
# Cost calculation (Claude Sonnet 4 pricing)
# ---------------------------------------------------------------------------

COST_PER_INPUT_TOKEN = 3.00 / 1_000_000   # $3.00 per 1M input tokens
COST_PER_OUTPUT_TOKEN = 15.00 / 1_000_000  # $15.00 per 1M output tokens


def calc_cost(input_tokens: int, output_tokens: int) -> float:
    return input_tokens * COST_PER_INPUT_TOKEN + output_tokens * COST_PER_OUTPUT_TOKEN

# ---------------------------------------------------------------------------
# Call logger
# ---------------------------------------------------------------------------

def log_call(request_kwargs: dict, response) -> None:
    """Append a request/response pair to the JSONL log file."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "request": {
            "model": request_kwargs.get("model"),
            "system": request_kwargs.get("system"),
            "messages": request_kwargs.get("messages"),
            "tools": [t["name"] for t in request_kwargs.get("tools", [])],
            "max_tokens": request_kwargs.get("max_tokens"),
        },
        "response": {
            "id": response.id,
            "model": response.model,
            "stop_reason": response.stop_reason,
            "content": [block.model_dump() for block in response.content],
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "cost_usd": round(calc_cost(response.usage.input_tokens, response.usage.output_tokens), 6),
            },
        },
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")

# ---------------------------------------------------------------------------
# Agentic loop (handles tool use)
# ---------------------------------------------------------------------------

MAX_TOOL_ROUNDS = 10


def chat_turn(messages: list) -> str:
    """Send messages to Claude, execute any tool calls, and return the final text."""
    for _ in range(MAX_TOOL_ROUNDS):
        kwargs = dict(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            messages=messages,
            tools=TOOLS,
        )
        response = client.messages.create(**kwargs)
        log_call(kwargs, response)

        # If no tool use, extract text and return
        if response.stop_reason != "tool_use":
            text_parts = [b.text for b in response.content if b.type == "text"]
            return "\n".join(text_parts) if text_parts else "(no response)"

        # Execute tools
        messages.append({"role": "assistant", "content": [b.model_dump() for b in response.content]})
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result = TOOL_DISPATCH[block.name](block.input)
                print(f"  [tool] {block.name}({block.input}) -> {result[:200]}")
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })
        messages.append({"role": "user", "content": tool_results})

    return "(max tool rounds reached)"

# ---------------------------------------------------------------------------
# Interactive loop
# ---------------------------------------------------------------------------

def main():
    messages = []

    print("\n=== AgentOps AI Agent ===")
    print("Tools: calculator, wikipedia_search, wikipedia_summary, get_current_datetime")
    print("Type 'quit' to exit.\n")

    try:
        while True:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                break

            messages.append({"role": "user", "content": user_input})
            answer = chat_turn(messages)
            messages.append({"role": "assistant", "content": answer})
            print(f"\nAgent: {answer}\n")

    except (KeyboardInterrupt, EOFError):
        print()

    agentops.end_session("Success")
    print(f"\n--- Session complete ---")
    print(f"Local call log: {os.path.abspath(LOG_FILE)}")
    print(f"AgentOps dashboard: https://app.agentops.ai")


if __name__ == "__main__":
    main()
