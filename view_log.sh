#!/usr/bin/env bash
# View and summarize the local LLM call log.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_FILE="$SCRIPT_DIR/agent_calls.jsonl"

if [ ! -f "$LOG_FILE" ]; then
    echo "No call log found. Run the agent first: ./run.sh"
    exit 1
fi

TOTAL=$(wc -l < "$LOG_FILE" | tr -d ' ')

usage() {
    echo "Usage: ./view_log.sh [command]"
    echo ""
    echo "Commands:"
    echo "  summary    Show call count and token totals (default)"
    echo "  last       Pretty-print the last call"
    echo "  last N     Pretty-print the last N calls"
    echo "  all        Pretty-print all calls"
    echo "  tokens     Show timestamp and token usage per call"
    echo "  clear      Delete the log file"
}

summary() {
    python3 -c "
import json, sys
calls = [json.loads(line) for line in open('$LOG_FILE')]
total_in = sum(c['response']['usage']['input_tokens'] for c in calls)
total_out = sum(c['response']['usage']['output_tokens'] for c in calls)
tool_calls = sum(1 for c in calls if c['response']['stop_reason'] == 'tool_use')
first = calls[0]['timestamp'][:19]
last = calls[-1]['timestamp'][:19]
print(f'Total LLM calls:  {len(calls)}')
print(f'Tool-use calls:   {tool_calls}')
print(f'Input tokens:     {total_in:,}')
print(f'Output tokens:    {total_out:,}')
print(f'First call:       {first}')
print(f'Last call:        {last}')
"
}

tokens() {
    python3 -c "
import json
for i, line in enumerate(open('$LOG_FILE'), 1):
    c = json.loads(line)
    ts = c['timestamp'][:19]
    u = c['response']['usage']
    stop = c['response']['stop_reason']
    print(f'{i:3d}  {ts}  in={u[\"input_tokens\"]:5d}  out={u[\"output_tokens\"]:5d}  {stop}')
"
}

last_n() {
    local n="${1:-1}"
    tail -n "$n" "$LOG_FILE" | python3 -m json.tool
}

case "${1:-summary}" in
    summary) summary ;;
    tokens)  tokens ;;
    last)    last_n "${2:-1}" ;;
    all)     last_n "$TOTAL" ;;
    clear)
        rm "$LOG_FILE"
        echo "Log file deleted."
        ;;
    -h|--help|help) usage ;;
    *) usage; exit 1 ;;
esac
