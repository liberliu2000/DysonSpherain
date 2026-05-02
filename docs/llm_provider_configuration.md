# LLM Provider Configuration

External LLM compaction is optional and disabled by default.

Default behavior:

```json
{
  "provider": "auto",
  "mode": "use_existing_agent_if_available",
  "external_llm_enabled": false,
  "fallback_to_local": true,
  "local_only": true,
  "require_user_config_for_external_api": true
}
```

Provider settings are exposed through `GET/POST /api/settings` and the Web UI settings panel. API keys are accepted as configuration input but are redacted from runtime configuration change events.

Provider discovery/test endpoints:

- `GET /api/llm/providers`
- `POST /api/llm/test-provider`

If no provider is configured, deterministic and local semantic compaction continue to work.
