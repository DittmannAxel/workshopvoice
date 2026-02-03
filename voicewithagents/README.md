# VoiceLive With Agents

Minimal Python sample that connects to **Azure VoiceLive** and uses an **existing** agent stored in **AI Foundry** (no agent creation).

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create `voicewithagents/.env` from the template:
```bash
cp sample_env .env
```

3. Fill in at least:
- `AZURE_VOICELIVE_ENDPOINT`
- `AZURE_VOICELIVE_PROJECT_NAME`
- `AZURE_VOICELIVE_AGENT_ID` (format: `agentName:version`, e.g. `agentFeborder:8`)
- `AZURE_VOICELIVE_VOICE`

Auth options:
- Default: uses `AzureCliCredential` (run `az login` first).
- API key: set `AZURE_VOICELIVE_API_KEY` and run with `--use-api-key`.

## Run

```bash
python voice_live_agent.py
```

Useful flags:
- `--use-api-key`
- `--verbose`

Logs are written to `voicewithagents/logs/`.
