# Microdrama Promo Finder

Finds the 20–60 second clips in a microdrama screenplay that are worth posting as promos:
each one opens on a hook, cuts on a cliffhanger, and makes sense to someone who has never
seen the show.

**Live app:** _add your streamlit.app link here_

## What it does

| Step | Who does it | Why |
|---|---|---|
| Parse screenplays into timestamped beats | Code | Timestamps must be exact |
| Find hooks and cliffhangers | AI | Judging what grips a stranger is a judgment task |
| Validate the AI's picks | Code | Invented beat IDs and scene transitions get dropped |
| Build 20–60 s promos, snapped to shot cuts | Code | Length and clean cuts are rules, not opinions |
| Rank and de-duplicate across the show | Code | Repeatable, explainable, free |

The model never reads or writes a timestamp. It returns beat IDs, and code maps those back to
times, so a timestamp cannot be hallucinated.

## Swapping the model

Two settings, no code changes:

```python
LLM_PROVIDER = "groq"              # groq | gemini | openai | anthropic | mock
LLM_MODEL    = "openai/gpt-oss-120b"
```

Set them in the app sidebar, or as environment variables. API keys are read from
`GROQ_API_KEY`, `GEMINI_API_KEY`, `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`, and are never
written into the code.

## Running it yourself

```bash
pip install -r requirements.txt

export LLM_PROVIDER=groq
export LLM_MODEL=openai/gpt-oss-120b
export GROQ_API_KEY=...

python parser.py path/to/screenplays   # -> parsed_show.json
python candidates.py                   # -> outputs/candidates.json  (the only AI step)
python promos.py                       # -> outputs/ranked_promos.csv / .md / all_promos.json

streamlit run streamlit_app.py         # or use the UI for all of the above
```

Expected runtime for a 50-episode show: a few minutes on a paid key, longer on a free tier,
where rate limits dominate. Each episode's answer is cached and saved, so an interrupted run
resumes instead of restarting.

## Input format

One `.txt` per episode, with timestamped blocks:

```
**LADY JULIANA (Confused):**
Start: (00:11.30) | End: (00:13.10)
"Why are you holding her hand?"
```

Blocks may be `CAMERA`, `ACTION`, `SOUND FX`, or a speaker name for dialogue.

## Note

Screenplays are not included in this repository.
