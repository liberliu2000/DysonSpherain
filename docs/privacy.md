# Privacy

The product memory layer is local-first. It uses local SQLite and filesystem
storage by default and does not require cloud services.

Before durable writes, text and metadata pass through the existing redaction
helpers in `sphere_cli.security`. Raw traces are still preserved after redaction,
so users should avoid capturing secrets in commands or files unless the local
machine is trusted.

Deletion modes:

- Default `forget` archives a capsule and marks it deprecated.
- `forget --hard` removes the capsule row and FTS row.

Raw trace blobs are local files under `.memory/raw/`.

Encryption-at-rest is reported by `dyson doctor` and the product privacy API.
By default the status is `not_configured`; use FileVault, an encrypted external
volume, or another OS-managed encrypted location for `.memory/`. If encryption
is managed outside DysonSpherain, write `.memory/encryption.json` to document the
external control, for example:

```json
{
  "provider": "FileVault",
  "scope": "project_volume"
}
```

SQLCipher-backed product SQLite is supported when the optional encrypted extra
and key environment variable are available:

```bash
python -m pip install -e ".[encrypted]"
export DYSON_MEMORY_SQLCIPHER_KEY="..."
dyson index configure-encryption sqlcipher --key-env DYSON_MEMORY_SQLCIPHER_KEY
```

If SQLCipher is configured without the driver or key, `dyson doctor` reports
`configured_unavailable` and the product store refuses to open the encrypted
database until the dependency and key are present.

Existing plaintext product databases can be converted with:

```bash
dyson index migrate-sqlcipher --key-env DYSON_MEMORY_SQLCIPHER_KEY
dyson index migrate-sqlcipher --key-env DYSON_MEMORY_SQLCIPHER_KEY --replace
```

Without `--replace`, the command writes a separate encrypted copy. With
`--replace`, it keeps a `.plaintext.backup.sqlite3` backup and moves the
encrypted DB into the active product DB path.

Ignore controls:

- Default excluded patterns include `.env`, private keys, credentials files,
  `node_modules/`, virtualenv folders, `__pycache__/`, and `.git/`.
- Project-specific rules can be added to `.dysonignore`.
- `dyson record` also accepts repeated `--allow` and `--deny` glob patterns.

Retention controls:

```bash
dyson forget --project DysonSpherain --before 2026-05-01T00:00:00+00:00
dyson forget --project DysonSpherain --keep-last 200
dyson forget --capsule-id cap_xxx
dyson forget --capsule-id cap_xxx --hard
```

Default forget tombstones the capsule, marks it deprecated, and records
tombstoned capsule references in affected context-pack payloads. Hard forget
also removes auxiliary entity, artifact, embedding, relation, FTS, and unused raw
trace rows.

Exports write a sidecar manifest with export time, capsule count, local-only
status, and redaction policy.
