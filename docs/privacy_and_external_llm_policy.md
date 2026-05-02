# Privacy and External LLM Policy

DysonSpherain defaults to local-only memory governance.

Defaults:

- `local_only: true`
- `allow_raw_memory_external: false`
- `allow_canonical_memory_external: true`
- `redact_secrets: true`
- `require_external_call_confirmation: true`
- `show_external_call_preview: true`

Compaction preserves raw/source memories locally. External LLM calls must be explicitly enabled and should not receive raw memory unless the user changes the privacy setting.

API keys are not printed by the UI and are redacted in runtime configuration audit events.
