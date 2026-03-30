"""Interactive AI agent with tools, memory, and observability via AgentOps."""

import json
import math
import os
import re
import time
from datetime import datetime
from html.parser import HTMLParser
from urllib.error import URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

import agentops
# agentops.init() auto-instruments Anthropic calls; no decorators needed
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

SESSION_NAME = f"agent-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
_agentops_session = agentops.init(
    os.getenv("AGENTOPS_API_KEY"),
    trace_name=SESSION_NAME,
    default_tags=["research-agent", "claude-sonnet-4"],
)
time.sleep(2)  # Wait for async auth token before any LLM calls
client = Anthropic()

LOG_FILE = "agent_calls.jsonl"
TRACES_FILE = "traces.jsonl"

_trace_id_cache: str | None = None

SYSTEM_PROMPT = """You are a helpful research assistant with access to these tools:
- calculator: evaluate math expressions (supports arithmetic and math functions like sqrt, sin, log)
- get_current_datetime: get the current date and time
- web_search: search the web using DuckDuckGo (use this for general questions, current events, companies, products, etc.)
- fetch_url: fetch and read the text content of any web page
- wikipedia_search: search Wikipedia for articles matching a query
- wikipedia_summary: fetch the full summary of a specific Wikipedia article by title

Use web_search for general queries, especially about companies, products, news, or anything not likely on Wikipedia. Use fetch_url to read a specific page for more detail. Use Wikipedia tools for encyclopedic topics. Provide clear, concise answers."""

# ---------------------------------------------------------------------------
# Tools (decorated as operations for AgentOps tracking)
# ---------------------------------------------------------------------------

def calculator(expression: str) -> str:
    try:
        result = eval(expression, {"__builtins__": {}, "math": math})
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def get_current_datetime() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class _TextExtractor(HTMLParser):
    """Simple HTML-to-text extractor."""
    def __init__(self):
        super().__init__()
        self._pieces: list[str] = []
        self._skip = False
        self._skip_tags = {"script", "style", "noscript", "svg", "nav", "footer", "header"}

    def handle_starttag(self, tag, attrs):
        if tag in self._skip_tags:
            self._skip = True

    def handle_endtag(self, tag):
        if tag in self._skip_tags:
            self._skip = False

    def handle_data(self, data):
        if not self._skip:
            text = data.strip()
            if text:
                self._pieces.append(text)

    def get_text(self) -> str:
        return "\n".join(self._pieces)


def web_search(query: str) -> str:
    """Search the web using DuckDuckGo HTML."""
    try:
        url = "https://html.duckduckgo.com/html/?" + urlencode({"q": query})
        req = Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AgentOpsDemo/1.0",
        })
        with urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8", errors="replace")
        results = []
        for match in re.finditer(
            r'<a rel="nofollow" class="result__a" href="([^"]+)"[^>]*>(.*?)</a>.*?'
            r'<a class="result__snippet"[^>]*>(.*?)</a>',
            html, re.DOTALL
        ):
            href, title, snippet = match.groups()
            title = re.sub(r"<[^>]+>", "", title).strip()
            snippet = re.sub(r"<[^>]+>", "", snippet).strip()
            if title:
                results.append(f"- {title}\n  {href}\n  {snippet}")
            if len(results) >= 8:
                break
        return "\n\n".join(results) if results else "No results found."
    except Exception as e:
        return f"Error searching web: {e}"


def fetch_url(url: str) -> str:
    """Fetch a URL and return its text content (truncated to ~4000 chars)."""
    try:
        req = Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AgentOpsDemo/1.0",
        })
        with urlopen(req, timeout=15) as resp:
            content_type = resp.headers.get("Content-Type", "")
            raw = resp.read(200_000).decode("utf-8", errors="replace")
        if "html" in content_type:
            parser = _TextExtractor()
            parser.feed(raw)
            text = parser.get_text()
        else:
            text = raw
        if len(text) > 4000:
            text = text[:4000] + "\n\n[... truncated]"
        return text
    except Exception as e:
        return f"Error fetching URL: {e}"


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
        "name": "web_search",
        "description": "Search the web using DuckDuckGo. Returns titles, URLs, and snippets for the top results. Use this for general queries about companies, products, news, people, or anything not limited to Wikipedia.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query, e.g. 'AgentOps AI observability' or 'latest Python release'"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "fetch_url",
        "description": "Fetch and read the text content of a web page. Use this to get more detail from a URL found via web_search.",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The full URL to fetch, e.g. 'https://example.com/page'"
                }
            },
            "required": ["url"]
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
    "web_search": lambda args: web_search(args["query"]),
    "fetch_url": lambda args: fetch_url(args["url"]),
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
        "trace_id": get_trace_id(),
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
# Agent class (decorated for AgentOps span hierarchy)
# ---------------------------------------------------------------------------

MAX_TOOL_ROUNDS = 10


class ResearchAgent:
    """AI research agent with tools, memory, and AgentOps instrumentation."""

    def __init__(self):
        self.messages: list = []

    def chat_turn(self) -> str:
        """Send messages to Claude, execute any tool calls, and return the final text."""
        for _ in range(MAX_TOOL_ROUNDS):
            kwargs = dict(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                messages=self.messages,
                tools=TOOLS,
            )
            response = client.messages.create(**kwargs)
            log_call(kwargs, response)

            if response.stop_reason != "tool_use":
                text_parts = [b.text for b in response.content if b.type == "text"]
                return "\n".join(text_parts) if text_parts else "(no response)"

            self.messages.append({"role": "assistant", "content": [b.model_dump() for b in response.content]})
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
            self.messages.append({"role": "user", "content": tool_results})

        return "(max tool rounds reached)"

    def ask(self, user_input: str) -> str:
        """Add user message and get agent response."""
        self.messages.append({"role": "user", "content": user_input})
        answer = self.chat_turn()
        self.messages.append({"role": "assistant", "content": answer})
        return answer

    def reset(self):
        """Clear conversation memory."""
        self.messages.clear()


# Module-level agent instance for the web UI
research_agent = ResearchAgent()


def chat_turn(messages: list) -> str:
    """Compatibility wrapper for app.py — delegates to the agent instance."""
    research_agent.messages = messages
    return research_agent.chat_turn()


def get_trace_id() -> str:
    """Extract the trace ID hex string from the AgentOps session (cached)."""
    global _trace_id_cache
    if _trace_id_cache is None:
        try:
            ctx = _agentops_session.trace_context.span.get_span_context()
            _trace_id_cache = format(ctx.trace_id, "032x")
        except Exception:
            _trace_id_cache = "unknown"
        if _trace_id_cache != "unknown":
            entry = {
                "timestamp": datetime.now().isoformat(),
                "trace_id": _trace_id_cache,
                "dashboard_url": f"https://app.agentops.ai/sessions?trace_id={_trace_id_cache}",
            }
            with open(TRACES_FILE, "a") as f:
                f.write(json.dumps(entry) + "\n")
    return _trace_id_cache


# ---------------------------------------------------------------------------
# Interactive CLI loop
# ---------------------------------------------------------------------------

def main():
    agent = ResearchAgent()

    print("\n=== AgentOps AI Agent ===")
    print("Tools: web_search, fetch_url, calculator, wikipedia_search, wikipedia_summary, get_current_datetime")
    print("Type 'quit' to exit.\n")

    try:
        while True:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                break

            answer = agent.ask(user_input)
            print(f"\nAgent: {answer}\n")

    except (KeyboardInterrupt, EOFError):
        print()

    trace_id = get_trace_id()
    print(f"\n--- Session complete ---")
    print(f"Trace ID:      {trace_id}")
    print(f"Dashboard:     https://app.agentops.ai/sessions?trace_id={trace_id}")
    print(f"Local log:     {os.path.abspath(LOG_FILE)}")
    print(f"Trace history: {os.path.abspath(TRACES_FILE)}")


if __name__ == "__main__":
    main()
