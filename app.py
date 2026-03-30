"""Flask web UI for the AgentOps AI agent."""

import json
import os

from flask import Flask, jsonify, render_template, request

from agent import LOG_FILE, TRACES_FILE, SESSION_NAME, get_trace_id, research_agent

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "").strip()
    if not user_input:
        return jsonify({"error": "Empty message"}), 400

    # Collect tool calls made during this turn
    tool_events = []
    _original_print = __builtins__.__dict__.get("print") if isinstance(__builtins__, dict) else getattr(__builtins__, "print")
    def _capture_print(*args, **kwargs):
        text = " ".join(str(a) for a in args)
        if text.strip().startswith("[tool]"):
            tool_events.append(text.strip())
        _original_print(*args, **kwargs)

    import builtins
    builtins.print = _capture_print
    try:
        answer = research_agent.ask(user_input)
    finally:
        builtins.print = _original_print

    return jsonify({"response": answer, "tools": tool_events})


@app.route("/api/logs")
def logs():
    if not os.path.exists(LOG_FILE):
        return jsonify([])
    entries = []
    with open(LOG_FILE) as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    return jsonify(entries)


@app.route("/api/logs/summary")
def logs_summary():
    if not os.path.exists(LOG_FILE):
        return jsonify({"total_calls": 0})
    calls = []
    with open(LOG_FILE) as f:
        for line in f:
            if line.strip():
                calls.append(json.loads(line))
    if not calls:
        return jsonify({"total_calls": 0})
    total_in = sum(c["response"]["usage"]["input_tokens"] for c in calls)
    total_out = sum(c["response"]["usage"]["output_tokens"] for c in calls)
    total_cost = sum(c["response"]["usage"].get("cost_usd", 0) for c in calls)
    tool_calls = sum(1 for c in calls if c["response"]["stop_reason"] == "tool_use")
    return jsonify({
        "total_calls": len(calls),
        "tool_use_calls": tool_calls,
        "input_tokens": total_in,
        "output_tokens": total_out,
        "total_cost_usd": round(total_cost, 4),
        "first_call": calls[0]["timestamp"][:19],
        "last_call": calls[-1]["timestamp"][:19],
    })


@app.route("/api/trace")
def trace():
    return jsonify({
        "session_name": SESSION_NAME,
        "trace_id": get_trace_id(),
        "dashboard_url": f"https://app.agentops.ai/sessions?trace_id={get_trace_id()}",
    })


@app.route("/api/traces")
def traces():
    if not os.path.exists(TRACES_FILE):
        return jsonify([])
    entries = []
    with open(TRACES_FILE) as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    return jsonify(entries)


@app.route("/api/reset", methods=["POST"])
def reset():
    research_agent.reset()
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(port=5001)
