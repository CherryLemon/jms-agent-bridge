---
name: jumpserver-access
description: Run scripts, remote Docker commands, jobs, and file transfers on JumpServer assets through the compact jms-agent-bridge MCP service.
---

# JumpServer access

Use the configured bridge's four tools: `run`, `read_result`, `discover`, `call`.
Client-specific prefixes may differ.

- Find authorized targets with `discover(topic="assets", query="<name or address>")`.
  Use an exact returned name or UUID. REST refreshes hourly; YAML does not grant
  target access. Do not fan out to `all` unless the task calls for it.
- Pass literal Bash as `run(targets=[...], script="...")`, or use an absolute local
  `script_path`. Use `cwd`, `env`, `container` or `image` instead of nested quoting.
  `docker_args` is an argv list. Do not encode the script or add shell quote layers.
- Output is saved completely. Start with the compact preview, then use
  `read_result` with `stream`, byte `offset`/`limit`, `tail`, or a literal `pattern`.
  Continue from `next_offset`; clipped previews are not complete logs.
- For important runs and job submission, generate a unique `request_id` before
  calling. After a lost reply, discover/call `request_status` with that original
  ID; do not invent another ID or submit again while completion is unknown.
  Repeated IDs never replay work. Saved output expires after the configured TTL
  (default 7 days); receipts remain and reads do not renew retention.
- `allow_dangerous` acknowledges only the local pattern guard for a specific
  action already authorized by the user. It does not grant approval or override
  client review or JumpServer policy. `script_guard_check` diagnoses that guard
  without executing anything. Do not change channels to retry an external denial.
- Discover a category such as `jobs` or `files`, then request one operation's
  schema with `discover(operation="...")` before invoking `call`. Large operation
  responses are read using `read_result(stream="result")`; `execution_results`
  references retain the underlying command logs.
- Default executions have isolated shell state and reuse the connection.
  Use a task-specific `session_id` only when `cd`, exports, or functions must
  survive. Close task-owned shells through the discovered `session_close` operation.
- The service reconnects idle connections and may retry opening a channel before
  script transmission. A failure after transmission is never replayed automatically.
  If `completion_unknown=true`, inspect saved output and remote state before
  deciding whether to repeat a command. `executed=null` means even its start is uncertain.
- A lost named shell is reported explicitly, including after a service restart.
  Use a new ID and reapply setup, or explicitly close/reset the previous ID. Do not
  assume reconnecting restores shell state. Avoid changing the shell's standard
  streams, `exit`/`exec`, and unmanaged background commands in retained shells.
- For background or long work, discover job operations. `run` is bounded to 600
  seconds. A timeout does not prove that a remote process stopped.
- `discover(topic="status")` reports connection and inventory state. A stale REST
  snapshot is not current permission or hardware-health evidence. The gateway
  enforces permissions; a rejected operation should not be retried through another transport.

Apply the user's authorized scope to remote changes and transfers. Output may
contain secrets: retrieve relevant excerpts instead of copying entire logs into chat.
