# Walkthrough: Building an AI Agent with AgentOps

This guide walks through every part of `agent.py`, explaining what each section does and why.

## Step 1: Imports and Initialization

```python
import json, math, os
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
```

- `agentops.init()` starts a session and **auto-patches the Anthropic SDK**. From this point on, every call to `client.messages.create()` is recorded — no extra decorators needed.
- `Anthropic()` picks up `ANTHROPIC_API_KEY` from the environment automatically.
- Standard library imports (`json`, `math`, `urllib`) power the tools with no extra pip dependencies.

## Step 2: Define the Tools

The agent has three tools, each a plain Python function:

### Calculator

```python
def calculator(expression: str) -> str:
    result = eval(expression, {"__builtins__": {}, "math": math})
    return str(result)
```

Uses `eval()` with a restricted namespace — only the `math` module is available, so no file I/O or imports can be executed. Supports expressions like `2**16`, `math.sqrt(144)`, or `math.pi * 10**2`.

### Current Date/Time

```python
def get_current_datetime() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
```

Returns the system clock. Useful when the user asks time-sensitive questions.

### Wikipedia Summary

```python
def wikipedia_summary(title: str) -> str:
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(title)}"
    req = Request(url, headers={"User-Agent": "AgentOpsDemo/1.0"})
    with urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read())
        return data.get("extract", "No summary available.")
```

Fetches the summary paragraph from Wikipedia's REST API. The `User-Agent` header is required by Wikipedia's API policy. No third-party Wikipedia library needed.

## Step 3: Tool Schemas for Claude

Claude's tool-use API needs JSON Schema definitions so it knows what tools are available and what parameters they accept:

```python
TOOLS = [
    {
        "name": "calculator",
        "description": "Evaluate a mathematical expression...",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "..."}
            },
            "required": ["expression"]
        }
    },
    # ... similarly for get_current_datetime and wikipedia_summary
]
```

A dispatch dict maps tool names to their Python implementations:

```python
TOOL_DISPATCH = {
    "calculator": lambda args: calculator(args["expression"]),
    "get_current_datetime": lambda args: get_current_datetime(),
    "wikipedia_summary": lambda args: wikipedia_summary(args["title"]),
}
```

## Step 4: Local Call Logger

Every LLM call is logged to `agent_calls.jsonl` (one JSON object per line):

```python
def log_call(request_kwargs, response):
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
            },
        },
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
```

JSONL format is used instead of a JSON array because each line can be appended independently — no need to read/rewrite the file. You can inspect it with `jq`, `tail`, or any JSON parser.

This local log complements the AgentOps dashboard. AgentOps gives you a visual timeline and cost tracking; the JSONL file gives you raw data you can grep, filter, and script against.

## Step 5: The Agentic Tool-Use Loop

This is the core of the agent. When Claude decides to use a tool, it returns a `tool_use` stop reason instead of finishing with text. The agent must execute the tool and send the result back:

```python
def chat_turn(messages: list) -> str:
    for _ in range(MAX_TOOL_ROUNDS):
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            messages=messages,
            tools=TOOLS,
        )
        log_call(kwargs, response)

        # If Claude is done talking, return the text
        if response.stop_reason != "tool_use":
            return "\n".join(b.text for b in response.content if b.type == "text")

        # Otherwise, execute tools and loop
        messages.append({"role": "assistant", "content": [b.model_dump() for b in response.content]})
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result = TOOL_DISPATCH[block.name](block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })
        messages.append({"role": "user", "content": tool_results})
```

The flow:

1. Send all messages (including conversation history) to Claude
2. If Claude responds with text → return it
3. If Claude responds with `tool_use` → execute each tool, append results, and call Claude again
4. Repeat up to 10 rounds (prevents runaway loops)

The key detail is message sequencing: after an assistant message with `tool_use` blocks, the next message must be a `user` message containing the corresponding `tool_result` blocks. Getting this wrong causes API errors.

## Step 6: Conversation Memory

Memory is simply the `messages` list that persists across turns:

```python
messages = []

while True:
    user_input = input("You: ")
    messages.append({"role": "user", "content": user_input})
    answer = chat_turn(messages)
    messages.append({"role": "assistant", "content": answer})
```

Each turn appends the user's input and the agent's response. Claude sees the full history on every call, so it can reference earlier parts of the conversation. Tool-use exchanges (the intermediate `tool_use`/`tool_result` messages) are also in the history, so Claude remembers what tools it called and what they returned.

For very long conversations, the message list could exceed the model's context window. A production agent would handle this by trimming older messages or summarizing them.

## Step 7: Session Lifecycle

```python
agentops.init(os.getenv("AGENTOPS_API_KEY"))  # Start session
# ... chat loop ...
agentops.end_session("Success")               # End session
```

The session brackets the entire conversation. When it ends, AgentOps uploads the full trace. You can view it at the URL printed when the session started, or browse all sessions at [app.agentops.ai](https://app.agentops.ai).

## What Gets Tracked

| Where | What you see |
|-------|-------------|
| **AgentOps dashboard** | Visual timeline, token counts, cost estimates, latency per call, full request/response content |
| **agent_calls.jsonl** | Raw JSON of every request and response — model, messages, tool calls, token usage, timestamps |
| **Terminal output** | Tool invocations as they happen (`[tool] calculator(...) -> 42.0`) |
