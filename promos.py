"""
Step 3: turn hooks + cliffhangers into actual promos, then rank them for the whole show.

Pure code, no AI calls (fast, free, repeatable). Uses the AI's scores from step 2.

  1. Single-segment promos: a hook and a later cliffhanger in the same episode, 20-60 s apart.
  2. Stitched promos: a short hook moment joined to a later cliffhanger moment (same or next
     episode), total 20-60 s.
  3. Cut points are snapped to camera-shot boundaries so the editor gets clean cuts, while the
     hook stays inside the first 3 seconds.
  4. Ranking: weighted score, then de-duplicated and spread across the show.

Usage:  python promos.py      (needs parsed_show.json and outputs/candidates.json)
Writes: outputs/all_promos.json, outputs/ranked_promos.csv, outputs/ranked_promos.md
"""
import re
import csv
import json
from pathlib import Path

OUT = Path("outputs")
MIN_LEN, MAX_LEN = 20.0, 60.0
SNAP = 1.5            # max seconds we move a cut to land on a shot boundary
HOOK_SEG = (6, 15)    # length range of the hook segment in a stitched promo
TOP_N = 20
MAX_PER_EPISODE = 2   # in the shortlist, so the top picks cover the whole show

# Ranking weights (explained in the doc)
W_HOOK, W_CLIFF, W_STANDALONE, W_EARLY = 0.40, 0.35, 0.15, 0.10
STITCH_PENALTY = 1.0  # stitched promos risk feeling disjointed, so they must earn their place
STITCH_MIN_SCORE = 9  # both the hook and the cliffhanger must be this strong to justify a stitch
MAX_STITCH_SHARE = 0.25   # the brief says a promo is "usually one continuous stretch"


def ts(s):
    return f"{int(s // 60):02d}:{s % 60:05.2f}"


def shot_starts(ep):
    return sorted({b["start"] for b in ep["beats"] if b["type"] == "CAMERA"})


def shot_ends(ep):
    return sorted({b["end"] for b in ep["beats"] if b["type"] == "CAMERA"})


def snap_start(t, ep):
    """Move a start earlier to the shot cut just before it (at most SNAP s), so the clip opens
    on a clean cut and the hook still lands within the first 3 seconds."""
    before = [s for s in shot_starts(ep) if t - SNAP <= s <= t]
    return max(before) if before else t


def snap_end(t, ep):
    """Move an end later to the next shot cut (at most SNAP s), without running into the next
    line of dialogue, which would start answering the cliffhanger."""
    next_line = min((b["start"] for b in ep["beats"]
                     if b["type"] == "DIALOGUE" and b["start"] >= t), default=ep["duration"])
    after = [e for e in shot_ends(ep) if t <= e <= min(t + SNAP, next_line + 0.3)]
    return min(after) if after else t


UNRESOLVED_END = (
    "?",      # a question nobody has answered yet
    "...",    # a sentence cut off mid-threat ("I said...")
    "\u2026",
)
RESOLVED_WORDS = re.compile(
    r"\b(i am offering|here it is|i accept|i do|yes|thank you|i forgive|it is done|"
    r"we are safe|at last|finally)\b", re.I)
TRANSITION = re.compile(r"\b(wide (tracking|establishing) shot|they begin walking|walk(s|ing) away|"
                        r"fade|cut to|credits|title card)\b", re.I)


def cliff_quality(cliff):
    """How unresolved does this ending feel? 0.3 (resolved) to 1.2 (cut mid-threat).
    Judged in code from the text itself, not from the model's own score."""
    text = (cliff.get("text") or "").strip()
    q = 1.0
    if text.endswith(UNRESOLVED_END):
        q += 0.2                      # ends on a question or mid-sentence
    if RESOLVED_WORDS.search(text):
        q -= 0.5                      # the thought is closed: tension has landed
    if cliff.get("beat_type") == "CAMERA" and TRANSITION.search(text):
        q -= 0.4                      # a scene transition is not a story cliffhanger
    if cliff.get("beat_type") == "CAMERA" and not text.endswith(UNRESOLVED_END):
        q -= 0.1                      # dialogue/action endings read better than shot notes
    return max(0.3, min(1.2, q))


CATEGORIES = {
    "Romance": {"romance", "love", "seduction", "kiss"},
    "Action & danger": {"slap", "threat", "violence", "fight", "action", "danger", "attack"},
    "Betrayal & revenge": {"betrayal", "humiliation", "accusation", "malicious", "revenge", "insult"},
    "Mystery": {"reveal", "question", "mystery", "secret"},
}
TOP_FIXED = 5   # the first five are ranked purely by score; variety applies after them


def category(hook, cliff):
    """Viewer-facing theme for a promo, derived from what the AI already returned:
    the hook's type, and whether the cliffhanger leaves a who/what/why question."""
    t = str(hook.get("type", "")).lower().strip()
    for name, types in CATEGORIES.items():
        if t in types:
            return name
    q = str(cliff.get("open_question", "")).lower()
    if q.startswith(("who", "what", "why", "where", "which")):
        return "Mystery"
    return "Other drama"


def standalone_score(hook):
    """1.0 if the AI said a new viewer needs no context, lower the more backstory is needed."""
    ctx = str(hook.get("context_needed", "")).strip().lower()
    if ctx in ("", "none", "n/a", "nothing"):
        return 1.0
    return 0.6 if len(ctx) < 60 else 0.3


def early_score(ep_num, n_eps):
    """New viewers start at episode 1, and early clips spoil less."""
    return max(0.0, 1 - (ep_num - 1) / max(n_eps - 1, 1))


def hook_clarity(hook):
    """A two- or three-word line ("Not anymore.") only grips someone who knows the story.
    Very short dialogue hooks are discounted; actions and visuals are judged by the AI alone."""
    if hook.get("beat_type") == "DIALOGUE" and len(str(hook.get("text", "")).split()) <= 3:
        return 0.8
    return 1.0


def score(hook, cliff, n_eps, stitched):
    s = (W_HOOK * hook["score"] * hook_clarity(hook)
         + W_CLIFF * cliff["score"] * cliff_quality(cliff)
         + W_STANDALONE * 10 * standalone_score(hook)
         + W_EARLY * 10 * early_score(hook["episode"], n_eps))
    return round(s - (STITCH_PENALTY if stitched else 0), 2)


def make_promo(hook, cliff, segments, n_eps, stitched):
    length = round(sum(e - s for _, s, e in segments), 2)
    return {
        "segments": [{"episode": ep, "start": ts(s), "end": ts(e), "start_s": s, "end_s": e}
                     for ep, s, e in segments],
        "episodes": sorted({ep for ep, _, _ in segments}),
        "length_s": length,
        "type": "stitched" if stitched else "continuous",
        "hook": hook["text"], "hook_type": hook.get("type", ""),
        "hook_beat": f"E{hook['episode']:02d}:{hook['start']}",
        "cliffhanger": cliff["text"], "open_question": cliff.get("open_question", ""),
        "cliff_beat": f"E{cliff['episode']:02d}:{cliff['end']}",
        "hook_score": hook["score"], "cliff_score": cliff["score"],
        "cliff_quality": round(cliff_quality(cliff), 2),
        "context_needed": hook.get("context_needed", ""),
        "why": f"Opens on {hook.get('type', 'a hook')} ({hook.get('why', '').rstrip('.')}); "
               f"cuts before the answer to: {cliff.get('open_question', '').rstrip('?')}?",
        "score": score(hook, cliff, n_eps, stitched),
        "category": category(hook, cliff),
    }


def prune_stitches(promos, rejected):
    """Keep stitched promos as the exception: drop any whose hook already has a continuous
    promo (that hook works fine uncut), then cap the stitched share of the pool."""
    continuous = [p for p in promos if p["type"] == "continuous"]
    covered = {p["hook_beat"] for p in continuous}
    stitches = [p for p in promos if p["type"] == "stitched"]
    dropped = [p for p in stitches if p["hook_beat"] in covered]
    kept = [p for p in stitches if p["hook_beat"] not in covered]
    kept.sort(key=lambda p: -p["score"])
    cap = int(len(continuous) * MAX_STITCH_SHARE / (1 - MAX_STITCH_SHARE)) if continuous else len(kept)
    rejected["stitch_hook_already_continuous"] = len(dropped)
    rejected["stitch_over_cap"] = max(0, len(kept) - cap)
    return continuous + kept[:cap]


def build(show, cands):
    eps = {e["episode"]: e for e in show}
    n_eps = len(show)
    promos, rejected = [], {"too_short": 0, "too_long": 0, "cliff_before_hook": 0,
                            "weak_cliffhanger": 0}

    for h in cands["hooks"]:
        ep = eps[h["episode"]]
        start = snap_start(h["start"], ep)
        for c in cands["cliffhangers"]:
            # 1) continuous promo, same episode
            if c["episode"] == h["episode"]:
                end = snap_end(c["end"], ep)
                if end <= h["end"]:
                    rejected["cliff_before_hook"] += 1
                    continue
                length = end - start
                if length < MIN_LEN:
                    rejected["too_short"] += 1
                elif length > MAX_LEN:
                    rejected["too_long"] += 1
                    # 2) too long to be continuous: try stitching hook moment + cliff moment
                    promos += stitched(h, c, ep, ep, start, end, n_eps)
                elif cliff_quality(c) < 0.7:
                    rejected["weak_cliffhanger"] += 1
                else:
                    promos.append(make_promo(h, c, [(ep["episode"], start, end)], n_eps, False))
            # 2b) stitched across into the next episode
            elif c["episode"] == h["episode"] + 1:
                ep2 = eps[c["episode"]]
                promos += stitched(h, c, ep, ep2, start, snap_end(c["end"], ep2), n_eps)
    promos = prune_stitches(promos, rejected)
    return promos, rejected


def stitched(h, c, ep1, ep2, start, end, n_eps):
    """Hook segment (6-15 s from the hook) + cliffhanger segment (the lead-up to the cliff).
    Only built for strong pairs: the brief says a promo is usually one continuous stretch, and
    joined moments from text alone may not actually cut together."""
    if h["score"] < STITCH_MIN_SCORE or c["score"] < STITCH_MIN_SCORE:
        return []
    if cliff_quality(c) < 1.0:
        return []
    hook_end = snap_end(min(max(h["end"] + 3, start + HOOK_SEG[0]), start + HOOK_SEG[1]), ep1)
    hook_len = hook_end - start
    out = []
    for cliff_len in (15, 20, 30, 40):
        seg2_start = snap_start(max(end - cliff_len, 0), ep2)
        if ep1["episode"] == ep2["episode"] and seg2_start <= hook_end:
            continue                     # segments must not overlap and must stay in order
        total = hook_len + (end - seg2_start)
        if MIN_LEN <= total <= MAX_LEN:
            out.append(make_promo(h, c, [(ep1["episode"], start, hook_end),
                                         (ep2["episode"], seg2_start, end)], n_eps, True))
            break
    return out


def overlaps(a, b):
    """True if two promos share a hook or cliff beat, or their first segments mostly overlap."""
    if a["hook_beat"] == b["hook_beat"] or a["cliff_beat"] == b["cliff_beat"]:
        return True
    sa, sb = a["segments"][0], b["segments"][0]
    if sa["episode"] != sb["episode"]:
        return False
    inter = min(sa["end_s"], sb["end_s"]) - max(sa["start_s"], sb["start_s"])
    return inter > 0.5 * min(sa["end_s"] - sa["start_s"], sb["end_s"] - sb["start_s"])


def rank(promos):
    """Top 5 purely by score. Then make sure every category in the show is represented,
    then fill the rest by score. No duplicates, at most MAX_PER_EPISODE per episode."""
    pool = sorted(promos, key=lambda p: -p["score"])
    chosen, per_ep = [], {}
    n_eps = len({p["episodes"][0] for p in pool}) or 1
    cap = max(MAX_PER_EPISODE, -(-TOP_N // n_eps))   # small runs: allow more per episode

    def can_add(p):
        return per_ep.get(p["episodes"][0], 0) < cap and \
            not any(overlaps(p, q) for q in chosen)

    def add(p):
        chosen.append(p)
        per_ep[p["episodes"][0]] = per_ep.get(p["episodes"][0], 0) + 1

    for p in pool:                                   # 1) the fixed top five
        if len(chosen) == TOP_FIXED:
            break
        if can_add(p):
            add(p)
    present = {p["category"] for p in pool}          # 2) one of each missing category
    for cat in sorted(present - {p["category"] for p in chosen}):
        best = next((p for p in pool if p["category"] == cat and p not in chosen and can_add(p)), None)
        if best and len(chosen) < TOP_N:
            add(best)
    for p in pool:                                   # 3) fill the rest by score
        if len(chosen) == TOP_N:
            break
        if p not in chosen and can_add(p):
            add(p)
    chosen = chosen[:TOP_FIXED] + sorted(chosen[TOP_FIXED:], key=lambda p: -p["score"])
    for i, p in enumerate(chosen, 1):
        p["rank"] = i
    return chosen


def write_outputs(all_promos, ranked, rejected, cands):
    OUT.mkdir(exist_ok=True)
    (OUT / "all_promos.json").write_text(json.dumps(
        {"provider": cands.get("provider"), "model": cands.get("model"),
         "episodes": len({h["episode"] for h in cands["hooks"]} | {c["episode"] for c in cands["cliffhangers"]}),
         "count": len(all_promos), "rejected_pairs": rejected,
         "categories": {c: sum(p["category"] == c for p in all_promos)
                        for c in sorted({p["category"] for p in all_promos})},
         "model_by_episode": cands.get("model_by_episode", {}),
         "promos": all_promos}, indent=1))

    with open(OUT / "ranked_promos.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "score", "category", "type", "length_s", "segments", "hook", "cliffhanger", "why"])
        for p in ranked:
            segs = " + ".join(f"Ep{s['episode']} {s['start']}-{s['end']}" for s in p["segments"])
            w.writerow([p["rank"], p["score"], p["category"], p["type"], p["length_s"], segs,
                        p["hook"], p["cliffhanger"], p["why"]])

    lines = [f"# Ranked promos ({cands.get('provider')}/{cands.get('model')})", "",
             f"{len(all_promos)} valid promos built; top {len(ranked)} shown.", ""]
    for p in ranked:
        segs = " + ".join(f"Ep {s['episode']} {s['start']}–{s['end']}" for s in p["segments"])
        lines += [f"## #{p['rank']} · score {p['score']} · {p['length_s']} s · {p['type']}",
                  f"- **Segments:** {segs}",
                  f"- **Hook:** {p['hook']}",
                  f"- **Cliffhanger:** {p['cliffhanger']}",
                  f"- **Why it works:** {p['why']}", ""]
    (OUT / "ranked_promos.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    show = json.loads(Path("parsed_show.json").read_text())
    cands = json.loads((OUT / "candidates.json").read_text())
    all_promos, rejected = build(show, cands)
    ranked = rank(all_promos)
    write_outputs(all_promos, ranked, rejected, cands)
    n_stitch = sum(p["type"] == "stitched" for p in all_promos)
    print(f"Built {len(all_promos)} valid promos ({n_stitch} stitched). Rejected pairs: {rejected}")
    print(f"Top {len(ranked)} cover episodes: {sorted({p['episodes'][0] for p in ranked})}")
    from collections import Counter
    print("Categories (all promos):", dict(Counter(p["category"] for p in all_promos)))
    print("Categories (shortlist): ", dict(Counter(p["category"] for p in ranked)))
    for p in ranked[:5]:
        segs = " + ".join(f"Ep{s['episode']} {s['start']}-{s['end']}" for s in p["segments"])
        print(f"#{p['rank']} [{p['score']}] {segs} ({p['length_s']}s)\n"
              f"    HOOK: {p['hook'][:70]}\n    CLIFF: {p['cliffhanger'][:70]}")
