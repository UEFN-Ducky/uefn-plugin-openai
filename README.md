# OpenAI

OpenAI API key + Codex coding agent for Settings → LLMs. Install from Store → Gateways, then enable to show OpenAI under Providers & Keys and Codex under Coding Agents.

Desktop plugin for [UEFN-Ducky](https://github.com/UEFN-Ducky/UEFN-Ducky) (`openai`).
Install or update from **Settings → Store** in the app — do not install from a zip by hand.

## Build

```bash
py scripts/build_zip.py
```

Writes `deploy/openai-1.0.12.ducky-plugin.zip` (scripts/ and deploy/ are not packed).

## Secrets

Never commit tokens or keys. The app stores `openai` locally (DPAPI), not in this package.

## License

MIT. Copyright (c) 2026 Mindful Path Company, LLC. See [LICENSE](LICENSE).
