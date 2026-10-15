#!/bin/bash
# Adds the sleeper MCP server to the Claude desktop app config.
# The app rewrites this file while running, so it waits for Claude to quit first.
trap '' HUP TERM
: "${SLEEPER_USERNAME:?set SLEEPER_USERNAME to your Sleeper username}"
DIR="$(cd "$(dirname "$0")" && pwd)"
UV="$(command -v uv)"
CFG="$HOME/Library/Application Support/Claude/claude_desktop_config.json"
while ps -ax -o command= | grep -q "^/Applications/Claude.app/Contents/MacOS/Claude$"; do sleep 1; done
sleep 1
/usr/bin/python3 - "$CFG" "$UV" "$DIR" "$SLEEPER_USERNAME" <<'PY'
import json, sys
p, uv, d, user = sys.argv[1:5]; c = json.load(open(p))
c.setdefault("mcpServers", {})["sleeper"] = {
    "command": uv,
    "args": ["run", "--directory", d, "python", "server.py"],
    "env": {"SLEEPER_USERNAME": user},
}
json.dump(c, open(p, "w"), indent=2)
PY
echo "patched $(date)"
open -a Claude
