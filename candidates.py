"""
Step 2b: find hook and cliffhanger moments in every episode.

AI does the judgment (which moments grip a stranger); code does the checking (every beat ID the
model returns must exist in that episode, otherwise it is dropped and logged).

Usage (after parsing):
    python candidates.py parsed_show.json
Outputs:
    outputs/raw/candidates/E01.json ... E50.json   (unedited model answers + validation notes)
    outputs/candidates.json                          (all validated moments, used by step 3)
"""
import re
import sys
import json
import time
from pathlib import Path

from llm import call_llm_json, current_config, CACHE_DIR
from prompts import EDITOR_SYSTEM, FIND_MOMENTS

OUT = Path("outputs")
SECONDS_BETWEEN_CALLS = 6   # free Gemini tier allows roughly 10 requests/minute
COMPACT = True              # drop camera notes already covered by dialogue/action (~40% fewer tokens)


def fmt_time(s):
    return f"{int(s // 60)}:{s % 60:05.2f}"


def beat_line(b, compact=False, extra_short=False):
    text = b["text"]
    if compact and b["type"] == "CAMERA":
        text = text[:70] + ("..." if len(text) > 70 else "")   # camera notes are the longest lines
    if extra_short and len(text) > 140:
        text = text[:140] + "..."
    who = f" {b['speaker']} ({b['tone']})" if b["type"] == "DIALOGUE" and b["tone"] else \
          f" {b['speaker']}" if b["type"] == "DIALOGUE" else ""
    return f"{b['id']} [{fmt_time(b['start'])}-{fmt_time(b['end'])}] {b['type']}{who}: {text}"


def compact_beats(beats):
    """Drop CAMERA beats that overlap in time with a dialogue/action/sound beat: those moments are
    already described, so the camera note is mostly redundant. Beat IDs stay unchanged."""
    story = [b for b in beats if b["type"] != "CAMERA"]
    def covered(cam):
        return any(s["start"] < cam["end"] and s["end"] > cam["start"] for s in story)
    return [b for b in beats if b["type"] != "CAMERA" or not covered(b)]


def build_prompt(ep, compact=True, extra_short=False):
    beats = compact_beats(ep["beats"]) if compact else ep["beats"]
    return FIND_MOMENTS.format(
        episode=ep["episode"], logline=ep["logline"], continuity=ep["continuity"],
        beats="\n".join(beat_line(b, compact, extra_short) for b in beats))


def is_too_large(err):
    resp = getattr(err, "response", None)
    return resp is not None and resp.status_code == 413


def validate_moments(ep, answer):
    """Keep only moments whose beat_id really exists; clamp scores to 1-10."""
    by_id = {b["id"]: b for b in ep["beats"]}
    kept, dropped = {"hooks": [], "cliffhangers": []}, []
    for kind in ("hooks", "cliffhangers"):
        for m in answer.get(kind, []) or []:
            bid = str(m.get("beat_id", "")).strip()
            if bid not in by_id:
                dropped.append({"kind": kind, "beat_id": bid, "reason": "ID not in episode"})
                continue
            b = by_id[bid]
            # Code-level safety net: a transition like "Cut to black" is not a story moment.
            if kind == "cliffhangers" and b["type"] == "CAMERA" and \
                    re.search(r"cut to black|fade (out|to)|black screen", b["text"], re.I):
                dropped.append({"kind": kind, "beat_id": bid, "reason": "transition, not a moment"})
                continue
            try:
                m["score"] = max(1, min(10, int(m.get("score", 5))))
            except (TypeError, ValueError):
                m["score"] = 5
            m.update(episode=ep["episode"], start=b["start"], end=b["end"],
                     beat_type=b["type"], speaker=b["speaker"], text=b["text"])
            kept[kind].append(m)
    return kept, dropped


def run(show):
    provider, model = current_config()
    print(f"Finding moments with {provider} / {model}")
    (OUT / "raw" / "candidates").mkdir(parents=True, exist_ok=True)
    all_hooks, all_cliffs, characters, stats = [], [], {}, {"dropped": 0, "failed_episodes": []}

    for ep in show:
        # Resume: if this episode was already answered by the same model, reuse the saved answer
        saved = OUT / "raw" / "candidates" / f"E{ep['episode']:02d}.json"
        if saved.exists():
            prev = json.loads(saved.read_text())
            if prev.get("provider") == provider and prev.get("model") == model \
                    and prev.get("model_answer"):
                kept, dropped = validate_moments(ep, prev["model_answer"])
                stats["dropped"] += len(dropped)
                all_hooks += kept["hooks"]
                all_cliffs += kept["cliffhangers"]
                characters[ep["episode"]] = prev["model_answer"].get("characters", {})
                stats.setdefault("reused_episodes", []).append(ep["episode"])
                continue
        n_cached = len(list(CACHE_DIR.rglob("*.json"))) if CACHE_DIR.exists() else 0
        try:
            try:
                answer = call_llm_json(build_prompt(ep, compact=COMPACT), system=EDITOR_SYSTEM)
            except Exception as e:
                if not is_too_large(e):
                    raise
                # Request too big for this model's limit: resend with shortened camera notes
                print(f"  Ep {ep['episode']:02d}: request too large, retrying with compact prompt")
                stats.setdefault("compact_episodes", []).append(ep["episode"])
                answer = call_llm_json(build_prompt(ep, compact=True, extra_short=True),
                                       system=EDITOR_SYSTEM)
        except Exception as e:                       # one bad episode must not kill the run
            detail = ""
            if getattr(e, "response", None) is not None:
                detail = " | " + e.response.text[:300].replace("\n", " ")
            print(f"  Ep {ep['episode']:02d}: FAILED ({e}){detail}")
            stats["failed_episodes"].append(ep["episode"])
            continue
        kept, dropped = validate_moments(ep, answer)
        stats["dropped"] += len(dropped)
        all_hooks += kept["hooks"]
        all_cliffs += kept["cliffhangers"]
        characters[ep["episode"]] = answer.get("characters", {})

        (OUT / "raw" / "candidates" / f"E{ep['episode']:02d}.json").write_text(json.dumps(
            {"provider": provider, "model": model, "model_answer": answer,
             "dropped_by_validation": dropped}, indent=1))
        print(f"  Ep {ep['episode']:02d}: {len(kept['hooks'])} hooks, "
              f"{len(kept['cliffhangers'])} cliffhangers, {len(dropped)} dropped")

        made_new_call = len(list(CACHE_DIR.rglob("*.json"))) > n_cached
        if made_new_call and provider != "mock":
            time.sleep(SECONDS_BETWEEN_CALLS)

    result = {"provider": provider, "model": model, "hooks": all_hooks,
              "cliffhangers": all_cliffs, "characters": characters, "stats": stats}
    (OUT / "candidates.json").write_text(json.dumps(result, indent=1))
    if stats.get("reused_episodes"):
        print(f"Reused saved answers for {len(stats['reused_episodes'])} episodes")
    print(f"\nTotal: {len(all_hooks)} hooks, {len(all_cliffs)} cliffhangers, "
          f"{stats['dropped']} invalid IDs dropped, failed episodes: {stats['failed_episodes']}")
    return result


if __name__ == "__main__":
    show = json.loads(Path(sys.argv[1] if len(sys.argv) > 1 else "parsed_show.json").read_text())
    run(show)
