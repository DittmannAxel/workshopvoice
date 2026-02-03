# workshopvoice

This repo contains small experiments and samples around Azure VoiceLive and Foundry agents.

## VoiceLive + Existing Foundry Agent

See `voicewithagents/` for a minimal Python script that connects to Azure VoiceLive and uses an **existing** agent stored in AI Foundry (no agent creation).

- Code: `voicewithagents/voice_live_agent.py`
- Config: `voicewithagents/sample_env` (copy to `.env` and fill values)
- Deps: `voicewithagents/requirements.txt`

Run:
```bash
pip install -r voicewithagents/requirements.txt
python voicewithagents/voice_live_agent.py
```

## Notes

- `credentials.md`, `instructions_agent.md`, etc. are local setup notes and are not meant to be committed.
