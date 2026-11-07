"""
Step 1 of the Kuku Promo Finder: turn raw screenplay .txt files into clean, structured beats.

Pure code, no AI. This is deliberate: timestamps must be exact, so the model is never allowed
to read or write a timestamp directly. Later steps refer to beats by their ID (e.g. "E03-B041"),
and code maps IDs back to timestamps.
"""
import re
import json
from pathlib import Path

# Matches a block header like **CAMERA:** or **LADY JULIANA (Confused, Breathless):**
HEADER_RE = re.compile(r"^\*\*(?P<label>[^*(]+?)\s*(?:\((?P<note>[^)]*)\))?\s*(?:\((?P<note2>[^)]*)\))?\s*:\*\*\s*$")
# Matches   Start: (00:11.30) | End: (00:13.10)
TIME_RE = re.compile(r"^Start:\s*\((\d+):(\d+(?:\.\d+)?)\)\s*\|\s*End:\s*\((\d+):(\d+(?:\.\d+)?)\)")
SCENE_RE = re.compile(r"^SCENE\s+(\d+)")
HEADING_RE = re.compile(r"^Scene heading:\s*(.+)")

NON_SPEAKER = {"CAMERA", "ACTION", "SOUND FX"}


def to_seconds(minutes, seconds):
    return round(int(minutes) * 60 + float(seconds), 2)


def parse_episode(path):
    text = Path(path).read_text(encoding="utf-8")
    lines = text.splitlines()

    # Metadata lines at the top of every file
    meta = {}
    for key, field in [("EPISODE NUMBER", "title"), ("EPISODE LOGLINE", "logline"),
                       ("CONTINUITY NOTES", "continuity")]:
        m = re.search(rf"^{key}:\s*(.+)$", text, re.MULTILINE)
        meta[field] = m.group(1).strip() if m else ""

    # Episode number comes from the file name (episode-003-...), which is reliable
    num = re.search(r"episode-(\d+)", Path(path).name)
    ep = int(num.group(1)) if num else int(re.search(r"\d+", meta["title"]).group())

    beats, scene, heading = [], 0, ""
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if m := SCENE_RE.match(line):
            scene = int(m.group(1))
        elif m := HEADING_RE.match(line):
            heading = m.group(1).strip()
        elif m := HEADER_RE.match(line):
            label = m.group("label").strip()
            kind = label if label in NON_SPEAKER else "DIALOGUE"
            tone = ", ".join(x for x in [m.group("note"), m.group("note2")] if x) or ""
            # Next non-empty line should be the timestamp, then the content
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            t = TIME_RE.match(lines[j].strip()) if j < len(lines) else None
            if t:
                content = []
                k = j + 1
                while k < len(lines) and lines[k].strip() and not lines[k].startswith("**") \
                        and not lines[k].startswith("---"):
                    content.append(lines[k].strip())
                    k += 1
                beats.append({
                    "id": f"E{ep:02d}-B{len(beats) + 1:03d}",
                    "episode": ep,
                    "scene": scene,
                    "location": heading,
                    "type": kind,                       # CAMERA / ACTION / SOUND FX / DIALOGUE
                    "speaker": label if kind == "DIALOGUE" else None,
                    "tone": tone if kind == "DIALOGUE" else None,
                    "start": to_seconds(t.group(1), t.group(2)),
                    "end": to_seconds(t.group(3), t.group(4)),
                    "text": " ".join(content).strip('"'),
                })
                i = k - 1
        i += 1

    beats.sort(key=lambda b: (b["start"], b["end"]))
    duration = max((b["end"] for b in beats), default=0)
    return {"episode": ep, **meta, "duration": duration, "beats": beats}


def validate(ep):
    """Code-level sanity checks. Returns a list of human-readable problems (empty = clean)."""
    problems = []
    for b in ep["beats"]:
        if b["end"] < b["start"]:
            problems.append(f"{b['id']}: end before start")
        if not b["text"]:
            problems.append(f"{b['id']}: empty text")
    if not ep["beats"]:
        problems.append("no beats parsed")
    return problems


def load_show(folder):
    files = sorted(Path(folder).glob("*.txt"))
    return [parse_episode(f) for f in files]


if __name__ == "__main__":
    import sys
    folder = sys.argv[1] if len(sys.argv) > 1 else "."
    show = load_show(folder)
    total_beats = sum(len(e["beats"]) for e in show)
    issues = {e["episode"]: validate(e) for e in show if validate(e)}
    print(f"Episodes parsed: {len(show)} | Beats: {total_beats} | "
          f"Episodes with issues: {len(issues)}")
    for ep, probs in issues.items():
        print(f"  Ep {ep}: {probs[:3]}")
    Path("parsed_show.json").write_text(json.dumps(show, indent=1))
