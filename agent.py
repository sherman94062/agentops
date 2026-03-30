"""Simple AI agent using AgentOps for observability and Anthropic for LLM calls."""

import os
import agentops
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

# Initialize AgentOps - automatically tracks LLM calls
agentops.init(os.getenv("AGENTOPS_API_KEY"))

client = Anthropic()

SYSTEM_PROMPT = """You are a helpful research assistant. When asked a question, provide a clear,
concise answer. If the question requires multiple steps, break it down and work through each step."""


def run_agent(user_query: str) -> str:
    """Run the agent on a single query and return the response."""
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


def main():
    queries = [
        "What are the three laws of thermodynamics? Explain each in one sentence.",
        "Write a Python function that checks if a string is a palindrome.",
    ]

    for query in queries:
        run_agent(query)

    agentops.end_session("Success")
    print("\n--- Session complete. Check your AgentOps dashboard for the replay. ---")


if __name__ == "__main__":
    main()
