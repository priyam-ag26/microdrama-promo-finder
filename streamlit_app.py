"""
Promo Finder for microdrama screenplays — Streamlit UI.

Two ways to use it:
  * Sample results: the full ranked output from a completed run, shown instantly, no API key.
  * Run a new show: upload screenplays (.zip or .txt) and run the pipeline live.

The pipeline itself lives in parser.py / llm.py / prompts.py / candidates.py / promos.py.
This file is only the interface, so the tool can be used from a notebook, the command line or here.
"""
import io
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st

import candidates
import llm
import parser as screenplay_parser
import promos as promo_builder

st.set_page_config(page_title="Microdrama Promo Finder", page_icon="🎬", layout="wide")

SAMPLE = Path("sample_results")
PROVIDERS = {
    "groq": ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"],
    "gemini": ["gemini-flash-latest", "gemini-3.5-flash", "gemini-flash-lite-latest"],
    "openai": ["gpt-4o-mini", "gpt-4o"],
    "anthropic": ["claude-sonnet-4-5", "claude-haiku-4-5"],
}
KEY_ENV = {"groq": "GROQ_API_KEY", "gemini": "GEMINI_API_KEY",
           "openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}


# ----------------------------------------------------------------- helpers
def load_sample():
    try:
        return json.loads((SAMPLE / "all_promos.json").read_text())
    except Exception:
        return None


def promos_to_frame(promos):
    rows = []
    for p in promos:
        rows.append({
            "Rank": p.get("rank", ""),
            "Score": p["score"],
            "Episode(s)": ", ".join(str(e) for e in p["episodes"]),
            "Segments": " + ".join(f"{s['start']}–{s['end']}" for s in p["segments"]),
            "Length (s)": p["length_s"],
            "Type": p["type"],
            "Hook": p["hook"],
            "Cliffhanger": p["cliffhanger"],
            "Why it works": p["why"],
        })
    return pd.DataFrame(rows)


def extract_uploads(files, folder):
    """Save uploaded .zip / .txt files into folder; return the directory holding the scripts."""
    folder.mkdir(parents=True, exist_ok=True)
    for f in files:
        if f.name.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(f.getvalue())) as z:
                z.extractall(folder)
        else:
            (folder / f.name).write_bytes(f.getvalue())
    # a zip often contains one show folder; use the deepest folder that holds .txt files
    txt_dirs = {p.parent for p in folder.rglob("*.txt")}
    return max(txt_dirs, key=lambda d: len(list(d.glob("*.txt")))) if txt_dirs else folder


def run_pipeline(script_dir, limit, seconds_between):
    show = screenplay_parser.load_show(script_dir)
    if limit:
        show = show[:limit]
    if not show:
        st.error("No screenplay .txt files found in the upload.")
        return None, None

    st.success(f"Parsed {len(show)} episodes, "
               f"{sum(len(e['beats']) for e in show)} beats. Finding promo moments…")

    candidates.SECONDS_BETWEEN_CALLS = seconds_between
    bar = st.progress(0.0, text="Starting…")
    status = st.empty()
    hooks, cliffs, failed = [], [], []

    for i, ep in enumerate(show, 1):
        status.write(f"Episode {ep['episode']} ({i}/{len(show)})…")
        try:
            answer = candidates.call_llm_json(
                candidates.build_prompt(ep, compact=candidates.COMPACT),
                system=candidates.EDITOR_SYSTEM)
            kept, _ = candidates.validate_moments(ep, answer)
            hooks += kept["hooks"]
            cliffs += kept["cliffhangers"]
        except Exception as e:                      # one bad episode must not stop the run
            failed.append(ep["episode"])
            status.warning(f"Episode {ep['episode']} failed: {str(e)[:120]}")
        bar.progress(i / len(show), text=f"{i}/{len(show)} episodes")

    bar.empty()
    status.empty()
    if failed:
        st.warning(f"{len(failed)} episode(s) failed and were skipped: {failed}")
    if not hooks or not cliffs:
        st.error("No usable moments were found. Check the API key and model in the sidebar.")
        return None, None

    cands = {"hooks": hooks, "cliffhangers": cliffs,
             "provider": llm.current_config()[0], "model": llm.current_config()[1]}
    built, rejected = promo_builder.build(show, cands)
    ranked = promo_builder.rank(built)
    return {"promos": built, "ranked": ranked, "rejected": rejected,
            "episodes": len(show), "failed": failed}, show


def show_results(data, show=None):
    ranked, built = data["ranked"], data["promos"]
    stitched = sum(p["type"] == "stitched" for p in built)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Episodes", data["episodes"])
    c2.metric("Valid promos", len(built))
    c3.metric("Stitched", f"{stitched} ({stitched * 100 // max(len(built), 1)}%)")
    c4.metric("Shortlist", len(ranked))

    st.subheader("Post these first")
    for p in ranked[:3]:
        with st.container(border=True):
            segs = " + ".join(f"Ep {s['episode']} {s['start']}–{s['end']}" for s in p["segments"])
            st.markdown(f"**#{p['rank']} · {segs} · {p['length_s']}s · {p['type']}**")
            st.markdown(f"🎣 **Hook:** {p['hook']}")
            st.markdown(f"✂️ **Cliffhanger:** {p['cliffhanger']}")
            st.caption(p["why"])

    st.subheader("Full shortlist")
    eps = sorted({e for p in ranked for e in p["episodes"]})
    f1, f2 = st.columns([2, 1])
    pick = f1.multiselect("Filter by episode", eps, default=[])
    kind = f2.selectbox("Type", ["all", "continuous", "stitched"])
    rows = [p for p in ranked
            if (not pick or any(e in pick for e in p["episodes"]))
            and (kind == "all" or p["type"] == kind)]
    st.dataframe(promos_to_frame(rows), use_container_width=True, hide_index=True)

    st.subheader("Download")
    d1, d2 = st.columns(2)
    d1.download_button("Shortlist (CSV)", promos_to_frame(ranked).to_csv(index=False),
                       "ranked_promos.csv", "text/csv", use_container_width=True)
    d2.download_button("All promos (JSON)", json.dumps(
        {"count": len(built), "rejected_pairs": data["rejected"], "promos": built}, indent=1),
        "all_promos.json", "application/json", use_container_width=True)

    with st.expander("What the code rejected, and why"):
        st.write(data["rejected"])
        st.caption("The model proposes moments; code enforces length, order, clean cut points "
                   "and how unresolved the ending is. These counts are those rules at work.")


# ----------------------------------------------------------------- sidebar
st.sidebar.title("⚙️ Model")
provider = st.sidebar.selectbox("Provider", list(PROVIDERS), index=0)
model = st.sidebar.selectbox("Model", PROVIDERS[provider], index=0)
api_key = st.sidebar.text_input(f"{KEY_ENV[provider]}", type="password",
                                help="Your key is used only for this session and never stored.")
st.sidebar.caption("Swapping models is two settings — no code changes.")

st.sidebar.divider()
st.sidebar.subheader("Run settings")
limit = st.sidebar.number_input("Episodes to process (0 = all)", 0, 500, 5,
                                help="Free API tiers are slow. Start with 5 to see it work.")
seconds_between = st.sidebar.slider("Seconds between calls", 0, 20, 7,
                                    help="Raise this if you hit rate limits on a free key.")

os.environ["LLM_PROVIDER"] = provider
os.environ["LLM_MODEL"] = model
if api_key:
    os.environ[KEY_ENV[provider]] = api_key.strip()


# ----------------------------------------------------------------- main
st.title("🎬 Microdrama Promo Finder")
st.caption("Finds the 20–60 second clips worth posting: opens on a hook, cuts on a cliffhanger, "
           "makes sense to someone who has never seen the show.")

tab_sample, tab_run, tab_how = st.tabs(["📊 Sample results", "🚀 Run a new show", "ℹ️ How it works"])

with tab_sample:
    sample = load_sample()
    if not sample:
        st.info("No sample results bundled with this deployment. Use the 'Run a new show' tab.")
    else:
        st.caption(f"A completed run on one 50-episode show "
                   f"({sample.get('provider', '')} {sample.get('model', '')}). No API key needed.")
        ranked = [p for p in sample["promos"] if p.get("rank")]
        ranked.sort(key=lambda p: p["rank"])
        show_results({"promos": sample["promos"], "ranked": ranked,
                      "rejected": sample.get("rejected_pairs", {}),
                      "episodes": sample.get("episodes", "—")})

with tab_run:
    st.markdown("Upload screenplays as a **.zip** or individual **.txt** files. "
                "Each file needs timestamped beats (`Start: (00:11.30) | End: (00:13.10)`).")
    files = st.file_uploader("Screenplays", type=["zip", "txt"], accept_multiple_files=True)
    go = st.button("Find promos", type="primary", disabled=not files)

    if go:
        if not api_key:
            st.error(f"Add your {KEY_ENV[provider]} in the sidebar first.")
        else:
            work = Path(tempfile.mkdtemp())
            try:
                script_dir = extract_uploads(files, work / "scripts")
                llm.CACHE_DIR = work / "cache"
                with st.spinner("Running…"):
                    data, show = run_pipeline(script_dir, int(limit), int(seconds_between))
                if data:
                    st.session_state["result"] = data
            finally:
                shutil.rmtree(work, ignore_errors=True)

    if "result" in st.session_state:
        st.divider()
        show_results(st.session_state["result"])

with tab_how:
    st.markdown("""
**The split: AI judges, code checks.**

1. **Parse (code).** Each screenplay becomes timestamped beats with IDs. The model never reads
   or writes a timestamp, so it cannot invent one.
2. **Find moments (AI).** Per episode, the model returns beat IDs for *hooks* (a clip could
   start here) and *cliffhangers* (a clip could end here), with scores and what context a new
   viewer would need.
3. **Validate (code).** Beat IDs that don't exist are dropped. Scene transitions like
   "cut to black" are rejected as cliffhangers.
4. **Build (code).** Hooks pair with later cliffhangers into 20–60 s clips, snapped to camera-shot
   boundaries so cuts are clean and the hook still lands in the first 3 seconds.
5. **Rank (code).** Hook strength, how unresolved the ending is, whether a stranger would follow
   it, and a preference for early episodes. Near-duplicates are removed and the shortlist is
   spread across the show.

**A human stays in the loop:** the output is a shortlist for an editor to approve, not an
auto-publisher.
""")
