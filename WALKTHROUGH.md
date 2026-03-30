# Walkthrough: Building an AI Agent with AgentOps

This guide walks through every part of `agent.py` step by step, explaining what each section does and why.

## Step 1: Import Dependencies

```python
import os
import agentops
from anthropic import Anthropic
from dotenv import load_dotenv
```

- **agentops** provides observability — session tracking, LLM call recording, cost monitoring
- **anthropic** is the official Python SDK for Claude
- **python-dotenv** loads API keys from a `.env` file so they stay out of source control

## Step 2: Load Environment and Initialize

```python
load_dotenv()

agentops.init(os.getenv("AGENTOPS_API_KEY"))

client = Anthropic()
```

`load_dotenv()` reads `.env` into environment variables. Then:

- `agentops.init()` starts an AgentOps session and automatically patches the Anthropic SDK. From this point on, every LLM call is recorded — no decorators or manual event logging needed.
- `Anthropic()` picks up `ANTHROPIC_API_KEY` from the environment automatically.

When `agentops.init()` runs, it prints a session replay URL like:

```
🖇 AgentOps: Session Replay: https://app.agentops.ai/sessions?trace_id=abc123
```

## Step 3: Define the System Prompt

```python
SYSTEM_PROMPT = """You are a helpful research assistant. When asked a question, provide a clear,
concise answer. If the question requires multiple steps, break it down and work through each step."""
```

The system prompt shapes the agent's behavior. It's sent with every request to Claude so the model maintains a consistent persona.

## Step 4: Build the Agent Function

```python
def run_agent(user_query: str) -> str:
    print(f"\n>>> User: {user_query}")

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_query}],
    )

    answer = response.content[0].text
    print(f"\n<<< Agent: {answer}")
    return answer
```

This is the core of the agent. It:

1. Prints the user's query
2. Sends it to Claude Sonnet 4 via the Anthropic Messages API
3. Extracts the text response from `response.content[0].text`
4. Prints and returns the answer

Because AgentOps patched the SDK in Step 2, this call is automatically recorded with:
- Input/output tokens
- Latency
- Model name
- Cost estimate

## Step 5: Run Queries and End the Session

```python
def main():
    queries = [
        "What are the three laws of thermodynamics? Explain each in one sentence.",
        "Write a Python function that checks if a string is a palindrome.",
    ]

    for query in queries:
        run_agent(query)

    agentops.end_session("Success")
    print("\n--- Session complete. Check your AgentOps dashboard for the replay. ---")
```

The main function loops through sample queries, runs each one, then calls `agentops.end_session("Success")` to finalize the session. The end state can be `"Success"`, `"Fail"`, or `"Indeterminate"`.

## Step 6: View Results in AgentOps

After running `python agent.py`, open the session replay URL printed in the terminal. The AgentOps dashboard shows:

- **Timeline**: A step-by-step view of every LLM call in order
- **Token usage**: Input and output tokens per call
- **Cost**: Estimated spend per call and for the session
- **Latency**: How long each call took

## Extending the Agent

Some ideas for building on this foundation:

- **Add tools**: Give the agent functions it can call (web search, calculations, file I/O) and use Claude's tool-use API
- **Add memory**: Maintain a conversation history by appending messages to the `messages` list
- **Use AgentOps decorators**: Add `@agentops.track_agent()` to class-based agents or `@agentops.record_action()` to track custom operations beyond LLM calls
- **Multi-agent setup**: Create multiple agent classes that collaborate, each tracked separately in the AgentOps dashboard
