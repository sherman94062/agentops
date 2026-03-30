# AgentOps Simple AI Agent

A Python AI agent that uses [AgentOps](https://www.agentops.ai/) for observability and [Anthropic Claude](https://www.anthropic.com/) as the LLM backend. AgentOps automatically instruments Anthropic SDK calls, giving you session replays, token usage tracking, and cost monitoring with zero extra code.

## Prerequisites

- Python 3.9+
- An [Anthropic API key](https://console.anthropic.com/)
- An [AgentOps API key](https://app.agentops.ai/) (free tier available)

## Setup

```bash
# Clone the repo
git clone https://github.com/sherman94062/agentops.git
cd agentops

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
ANTHROPIC_API_KEY=your-anthropic-key
AGENTOPS_API_KEY=your-agentops-key
```

## Usage

```bash
python agent.py
```

The agent runs two sample queries against Claude Sonnet 4 and prints the responses. When it finishes, AgentOps prints a link to your session replay dashboard where you can inspect every LLM call, token count, and cost.

## Project Structure

```
.
├── agent.py           # Main agent script
├── requirements.txt   # Python dependencies
├── .env               # API keys (not committed)
└── .gitignore
```

## How It Works

1. `agentops.init()` starts a session and auto-instruments the Anthropic SDK
2. Each `client.messages.create()` call is tracked automatically
3. `agentops.end_session("Success")` closes the session and uploads telemetry
4. Visit the printed AgentOps URL to see the full session replay

## License

MIT
