# AgentOps AI Agent

An interactive AI agent with tool use, conversation memory, and full observability via [AgentOps](https://www.agentops.ai/) and [Anthropic Claude](https://www.anthropic.com/).

## Features

- **Interactive chat** — conversational loop with persistent memory across turns
- **Tool use** — calculator, Wikipedia lookup, and current date/time via Claude's native tool-use API
- **Dual observability** — remote dashboard via AgentOps + local JSONL call log
- **Auto-instrumentation** — AgentOps patches the Anthropic SDK so every LLM call is tracked automatically

## Prerequisites

- Python 3.9+
- An [Anthropic API key](https://console.anthropic.com/)
- An [AgentOps API key](https://app.agentops.ai/) (free tier available)

## Setup

```bash
git clone https://github.com/sherman94062/agentops.git
cd agentops

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
ANTHROPIC_API_KEY=your-anthropic-key
AGENTOPS_API_KEY=your-agentops-key
```

## Usage

### Web UI

```bash
python app.py
```

Open [http://localhost:5001](http://localhost:5001). The UI has two panels:

- **Chat** (left) — type messages, see tool calls and responses in real time
- **LLM Call Log** (right) — summary cards (total calls, tool calls, token counts) and a clickable list of every LLM call with full request/response JSON

Buttons: **New Chat** resets conversation memory, **Toggle Logs** shows/hides the log panel, **Refresh Logs** reloads the log data.

### CLI

```bash
python agent.py
```

Start chatting. The agent remembers your conversation and can use tools:

```
You: What's the square root of 1764?
  [tool] calculator({'expression': 'math.sqrt(1764)'}) -> 42.0
Agent: The square root of 1764 is 42.

You: Tell me about Marie Curie
  [tool] wikipedia_summary({'title': 'Marie Curie'}) -> Maria Salomea Skłodowska Curie...
Agent: Marie Curie was a groundbreaking Polish-French physicist and chemist...

You: What time is it?
  [tool] get_current_datetime({}) -> 2026-03-30 14:44:57
Agent: The current date and time is March 30, 2026 at 2:44:57 PM.
```

Type `quit` to exit.

## Viewing LLM Calls

You have two ways to inspect every call made to the LLM:

### AgentOps Dashboard (remote)

When the agent starts, it prints a session replay URL:

```
🖇 AgentOps: Session Replay: https://app.agentops.ai/sessions?trace_id=abc123
```

Open it to see a timeline of every LLM call with token counts, latency, cost, and full request/response content.

### Local Call Log

Every call is also appended to `agent_calls.jsonl` in the project directory. Each line is a JSON object with:

- `timestamp` — when the call was made
- `request` — model, system prompt, messages, tools, max_tokens
- `response` — model response ID, stop reason, content blocks, token usage

Inspect it with:

```bash
# Pretty-print the last call
tail -1 agent_calls.jsonl | python -m json.tool

# Count total calls
wc -l agent_calls.jsonl

# Extract all token usage
cat agent_calls.jsonl | python -c "
import json, sys
for line in sys.stdin:
    entry = json.loads(line)
    ts = entry['timestamp'][:19]
    usage = entry['response']['usage']
    print(f\"{ts}  in={usage['input_tokens']}  out={usage['output_tokens']}\")
"
```

## Project Structure

```
.
├── app.py               # Flask web UI
├── templates/
│   └── index.html       # Chat + log viewer frontend
├── agent.py             # Agent core (tools, memory, agentic loop) + CLI
├── run.sh               # Shell script to activate venv and start CLI
├── view_log.sh          # Shell script to inspect the call log
├── requirements.txt     # Python dependencies
├── agent_calls.jsonl    # Local LLM call log (generated at runtime)
├── .env                 # API keys (not committed)
└── .gitignore
```

## License

MIT
