"""
Microdrama Promo Finder — Streamlit interface.

Design: a dark "screening room" look. Deep plum-black surfaces, a warm red ember glow taken from
the brand red, a soft serif for titles (like a film title card) and a clean sans for the UI.
The one recurring visual idea is the clip timeline: every promo shows where its hook lands and
where it cuts, on a 0–60 second scale.

The pipeline lives in parser.py / llm.py / prompts.py / candidates.py / promos.py.
This file is only the interface.
"""
import base64
import html
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

st.set_page_config(page_title="Microdrama Promo Finder", page_icon="🎬", layout="wide",
                   initial_sidebar_state="expanded")

HERE = Path(__file__).parent
SAMPLE = HERE / "sample_results" / "all_promos.json"
LOGO = next((p for p in [HERE / "assets" / "logo.png", HERE / "assets" / "logo.svg"]
             if p.exists()), None)

PROVIDERS = {
    "groq": ("Groq", ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b"]),
    "gemini": ("Google Gemini", ["gemini-flash-latest", "gemini-3.5-flash", "gemini-flash-lite-latest"]),
    "openai": ("OpenAI", ["gpt-4o-mini", "gpt-4o"]),
    "anthropic": ("Anthropic", ["claude-sonnet-4-5", "claude-haiku-4-5"]),
}
KEY_ENV = {"groq": "GROQ_API_KEY", "gemini": "GEMINI_API_KEY",
           "openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}
DEMO_MAX_EPISODES = 5   # protects the shared demo key's free quota


def demo_key(provider):
    """A key the app owner stored as a Streamlit secret, so visitors can try the tool
    without their own. It is never shown on screen or written to the repo."""
    try:
        return str(st.secrets.get(KEY_ENV[provider], "")).strip()
    except Exception:
        return ""


KEY_HELP = {"groq": "console.groq.com", "gemini": "aistudio.google.com/apikey",
            "openai": "platform.openai.com", "anthropic": "console.anthropic.com"}

# ---------------------------------------------------------------- design tokens + CSS
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,700&family=Manrope:wght@400;500;600;700&display=swap');

:root{
  --ink:#070509;        /* page */
  --surface:#110D13;    /* cards */
  --raised:#1A141D;     /* inputs, hovered */
  --line:rgba(255,236,240,.08);
  --text:#F5EFF1;
  --muted:#A99BA6;
  --red:#E3222C;        /* brand red */
  --ember:#7E1420;      /* deep glow */
  --good:#3FB37F;
  --r-lg:20px; --r-md:14px; --r-sm:10px;
}

/* page + streamlit chrome */
html, body, [data-testid="stAppViewContainer"]{
  background:
    radial-gradient(1100px 520px at 50% -180px, rgba(227,34,44,.30), transparent 70%),
    radial-gradient(700px 400px at 100% 0%, rgba(126,20,32,.25), transparent 70%),
    var(--ink) !important;
  color:var(--text);
  font-family:'Manrope', system-ui, sans-serif;
}
[data-testid="stHeader"]{background:transparent;}
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"]{visibility:hidden;}
.block-container{padding-top:2.2rem; max-width:1180px;}
h1,h2,h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
[data-testid="stMarkdownContainer"] h1, [data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3, .hero h1{
  font-family:'Fraunces', Georgia, serif !important; letter-spacing:-.01em; color:var(--text) !important;}
p, li, label, span, div{font-family:'Manrope', system-ui, sans-serif;}
h1 *, h2 *, h3 *{font-family:'Fraunces', Georgia, serif !important;}

/* sidebar */
[data-testid="stSidebar"]{
  background:linear-gradient(180deg, #0D0A0F 0%, #08060A 100%) !important;
  border-right:1px solid var(--line);
}
[data-testid="stSidebar"] .block-container{padding-top:1.4rem;}
.side-brand{display:flex; align-items:center; gap:.6rem; margin-bottom:1.2rem;}
.side-brand img{height:34px; mix-blend-mode:screen;}
.side-brand span{font-family:'Fraunces',serif; font-weight:700; font-size:1.05rem;}
.side-h{display:flex; align-items:center; gap:.5rem; color:var(--text); font-weight:700;
        font-size:.92rem; margin:1.1rem 0 .35rem;}
.side-h .ic{width:22px; height:22px; border-radius:7px; display:grid; place-items:center;
            background:rgba(227,34,44,.14); color:var(--red); font-size:.8rem;}
.side-note{color:var(--muted); font-size:.78rem; line-height:1.45; margin-top:.2rem;}
.side-status{margin-top:1rem; padding:.7rem .8rem; border-radius:var(--r-sm);
             border:1px solid var(--line); background:rgba(255,255,255,.02); font-size:.8rem;}
.dot{display:inline-block; width:7px; height:7px; border-radius:50%; margin-right:.45rem;}

/* inputs */
[data-baseweb="select"] > div, [data-baseweb="input"] > div, .stNumberInput input{
  background:var(--raised) !important; border:1px solid var(--line) !important;
  border-radius:var(--r-sm) !important; color:var(--text) !important;
}
.stSlider [role="slider"]{background:var(--red) !important;}

/* buttons */
.stButton > button, .stDownloadButton > button{
  border-radius:999px; border:1px solid var(--line); background:var(--raised);
  color:var(--text); font-weight:600; padding:.55rem 1.2rem; transition:border-color .15s;
}
.stButton > button:hover, .stDownloadButton > button:hover{border-color:rgba(227,34,44,.6); color:var(--text);}
.stButton > button[kind="primary"]{
  background:linear-gradient(135deg, #F0313A 0%, #B3161F 100%); border:none;
  box-shadow:0 8px 28px -8px rgba(227,34,44,.65);
}
.stButton > button[kind="primary"]:disabled{background:var(--raised); color:var(--muted);
  box-shadow:none; border:1px solid var(--line); opacity:1;}
.stButton > button:focus-visible, .stDownloadButton > button:focus-visible{outline:2px solid var(--red); outline-offset:2px;}

/* tabs */
.stTabs [data-baseweb="tab-list"]{gap:.4rem; border-bottom:1px solid var(--line);}
.stTabs [data-baseweb="tab"]{background:transparent; color:var(--muted); font-weight:600;
  padding:.6rem 1rem; border-radius:10px 10px 0 0;}
.stTabs [aria-selected="true"]{color:var(--text) !important;}
.stTabs [data-baseweb="tab-highlight"]{background:var(--red) !important;}

/* hero: centred title on the page itself */
.hero{text-align:center; padding:1.6rem 1rem 1.8rem; margin:0 auto;}
/* the logo file has a solid black background: "screen" blending makes black disappear
   into the dark page while white and red stay exact, so no box shows around it */
.hero-logo img{height:72px; margin-bottom:.6rem; mix-blend-mode:screen;}
.hero h1{font-size:clamp(2.1rem, 4.6vw, 3.4rem); line-height:1.05; margin:0 auto .8rem; font-weight:700;}
.hero p{color:var(--muted); font-size:1.06rem; max-width:56ch; margin:0 auto;}

/* upload panel: the main action, first thing below the title */
.panel-head{display:flex; align-items:center; justify-content:space-between; gap:1rem; flex-wrap:wrap; margin-bottom:.9rem;}
.panel-head h3{margin:0; font-size:1.35rem;}
.checks{display:flex; gap:.5rem; flex-wrap:wrap;}
.check{display:inline-flex; align-items:center; gap:.45rem; font-size:.8rem; padding:.32rem .7rem;
  border-radius:999px; border:1px solid var(--line); background:rgba(255,255,255,.02); color:var(--muted);}
.check.ok{color:var(--text); border-color:rgba(63,179,127,.45);}
.check.todo{color:var(--text); border-color:rgba(227,34,44,.55);}
.check i{width:7px; height:7px; border-radius:50%; background:var(--muted); display:inline-block;}
.check.ok i{background:var(--good);} .check.todo i{background:var(--red);}
.st-key-upload_panel, [data-testid="stVerticalBlockBorderWrapper"]:has(.upload-anchor){
  border:1px solid rgba(227,34,44,.30) !important; border-radius:24px !important; padding:1.4rem 1.5rem !important;
  background:linear-gradient(180deg, rgba(227,34,44,.07), rgba(17,13,19,.6) 60%) !important;}
.hint{color:var(--muted); font-size:.85rem; margin-top:.2rem;}

/* setup guide */
.guide{border-radius:var(--r-lg); border:1px solid rgba(227,34,44,.28);
  background:linear-gradient(180deg, rgba(227,34,44,.08), rgba(227,34,44,.02));
  padding:1.2rem 1.4rem; margin-bottom:1.4rem;}
.guide h3{font-size:1.15rem; margin:0 0 .8rem;}
.steps{display:grid; grid-template-columns:repeat(3, minmax(0,1fr)); gap:.8rem;}
.step{display:flex; gap:.7rem; align-items:flex-start; padding:.8rem; border-radius:var(--r-md);
  background:rgba(14,10,16,.55); border:1px solid var(--line);}
.step .n{flex:0 0 28px; height:28px; border-radius:50%; display:grid; place-items:center;
  font-weight:700; font-size:.85rem; background:var(--raised); color:var(--muted);}
.step.done .n{background:var(--good); color:#06140D;}
.step b{display:block; font-size:.92rem; margin-bottom:.15rem;}
.step span{color:var(--muted); font-size:.82rem; line-height:1.4;}
@media (max-width:820px){.steps{grid-template-columns:1fr;}}

/* stats */
.stats{display:grid; grid-template-columns:repeat(4, minmax(0,1fr)); gap:.8rem; margin:.4rem 0 1.4rem;}
.stat{padding:1rem 1.1rem; border-radius:var(--r-md); background:var(--surface); border:1px solid var(--line);}
.stat .v{font-family:'Fraunces',serif; font-size:1.9rem; font-weight:700; line-height:1;}
.stat .l{color:var(--muted); font-size:.82rem; margin-top:.35rem;}
@media (max-width:820px){.stats{grid-template-columns:repeat(2, minmax(0,1fr));}}

/* promo cards */
.promo{position:relative; padding:1.2rem 1.3rem; border-radius:var(--r-lg); background:var(--surface);
  border:1px solid var(--line); margin-bottom:.9rem;}
.promo.top{border-color:rgba(227,34,44,.45);
  background:linear-gradient(160deg, rgba(227,34,44,.10), var(--surface) 55%);}
.promo-head{display:flex; align-items:center; gap:.7rem; flex-wrap:wrap; margin-bottom:.8rem;}
.rank:empty{display:none;}
.rank{font-family:'Fraunces',serif; font-weight:700; font-size:1.6rem; color:var(--red); min-width:2.2rem;}
.chip{font-size:.75rem; padding:.22rem .6rem; border-radius:999px; background:var(--raised);
  color:var(--muted); border:1px solid var(--line);}
.chip.score{color:var(--text); border-color:rgba(227,34,44,.4);}
.chip.cat{color:#FFD9DC; background:rgba(227,34,44,.16); border-color:rgba(227,34,44,.35);}
.cats{display:grid; grid-template-columns:repeat(auto-fit, minmax(170px, 1fr)); gap:.7rem; margin:.3rem 0 1rem;}
.cat-tile{padding:.9rem 1rem; border-radius:var(--r-md); background:var(--surface); border:1px solid var(--line);}
.cat-tile .n{font-family:'Fraunces',serif; font-size:1.6rem; font-weight:700; line-height:1;}
.cat-tile .t{font-weight:600; font-size:.9rem; margin-top:.3rem;}
.cat-tile .bar{height:4px; border-radius:999px; background:rgba(255,255,255,.06); margin-top:.6rem; overflow:hidden;}
.cat-tile .bar i{display:block; height:100%; background:linear-gradient(90deg, var(--red), var(--ember));}
.where{font-weight:600; font-size:.92rem;}
.beats{display:grid; grid-template-columns:auto 1fr; gap:.35rem .8rem; font-size:.95rem; line-height:1.45;}
.beats .k{color:var(--muted); font-size:.8rem; padding-top:.15rem;}
.why{color:var(--muted); font-size:.85rem; margin-top:.75rem; padding-top:.7rem; border-top:1px solid var(--line);}
.clip{position:relative; height:8px; border-radius:999px; background:rgba(255,255,255,.06); margin:.2rem 0 .9rem;}
.clip .fill{position:absolute; left:0; top:0; bottom:0; border-radius:999px;
  background:linear-gradient(90deg, var(--red), var(--ember));}
.clip .min{position:absolute; top:-3px; bottom:-3px; width:1px; background:rgba(255,255,255,.35);}
.clip-scale{display:flex; justify-content:space-between; color:var(--muted); font-size:.7rem; margin-top:-.6rem; margin-bottom:.8rem;}

/* alerts restyled to the theme */
[data-testid="stAlert"], [data-testid="stAlertContainer"]{background:rgba(227,34,44,.08) !important;
  border:1px solid rgba(227,34,44,.3) !important; border-radius:var(--r-md) !important; color:var(--text) !important;}
[data-testid="stAlertContentSuccess"], [data-testid="stAlertContainer"]:has([data-testid="stAlertContentSuccess"]){
  background:rgba(63,179,127,.08) !important; border-color:rgba(63,179,127,.35) !important;}
[data-testid="stAlert"] *{color:var(--text) !important;}

/* category selector */
[data-testid="stButtonGroup"] button{background:var(--surface) !important; border:1px solid var(--line) !important;
  color:var(--muted) !important; border-radius:999px !important; padding:.45rem .95rem !important; font-weight:600;}
[data-testid="stButtonGroup"] button[aria-checked="true"], [data-testid="stButtonGroup"] button[kind$="Active"]{
  background:rgba(227,34,44,.16) !important; border-color:rgba(227,34,44,.6) !important; color:var(--text) !important;}
[data-testid="stButtonGroup"]{gap:.5rem; flex-wrap:wrap;}

/* empty + info states */
.empty{padding:2rem; text-align:center; border-radius:var(--r-lg); border:1px dashed var(--line); color:var(--muted);}
.empty b{color:var(--text); display:block; font-size:1.05rem; margin-bottom:.3rem;}
[data-testid="stFileUploader"] section{background:var(--surface); border:1px dashed rgba(227,34,44,.35);
  border-radius:var(--r-lg);}
[data-testid="stExpander"]{background:var(--surface); border:1px solid var(--line); border-radius:var(--r-md);}
[data-testid="stDataFrame"]{border:1px solid var(--line); border-radius:var(--r-md); overflow:hidden;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)



ICON = {
    "model": '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="5" y="5" width="14" height="14" rx="2"/><rect x="9" y="9" width="6" height="6"/><path d="M9 2v3M15 2v3M9 19v3M15 19v3M2 9h3M2 15h3M19 9h3M19 15h3"/></svg>',
    "key": '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="7.5" cy="15.5" r="4.5"/><path d="M10.7 12.3 21 2M17 6l3 3M14 9l2 2"/></svg>',
    "run": '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 21v-7M4 10V3M12 21v-9M12 8V3M20 21v-5M20 12V3M1 14h6M9 8h6M17 16h6"/></svg>',
}

# ---------------------------------------------------------------- small helpers
def esc(x):
    return html.escape(str(x or ""))


def logo_tag(height_class=""):
    if not LOGO:
        return ""
    mime = "image/svg+xml" if LOGO.suffix == ".svg" else "image/png"
    b64 = base64.b64encode(LOGO.read_bytes()).decode()
    return f'<img src="data:{mime};base64,{b64}" alt="Kuku">'


def _drive_download_url(link):
    """Turn a Google Drive share link into a direct download link."""
    import re
    m = re.search(r"/d/([\w-]+)", link) or re.search(r"[?&]id=([\w-]+)", link)
    return f"https://drive.google.com/uc?export=download&id={m.group(1)}" if m else link


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_json(url):
    import requests
    r = requests.get(_drive_download_url(url), timeout=30)
    r.raise_for_status()
    return r.json()


def load_sample():
    """Example results. Loaded from a link set as a Streamlit secret (SAMPLE_URL), so the
    show's content never sits in the public repo. Falls back to a local file if present."""
    try:
        url = st.secrets.get("SAMPLE_URL", "")
    except Exception:
        url = ""
    if url:
        try:
            return _fetch_json(url)
        except Exception:
            pass
    try:
        return json.loads(SAMPLE.read_text())
    except Exception:
        return None


def promos_frame(rows):
    return pd.DataFrame([{
        "Rank": p.get("rank", ""), "Score": p["score"],
        "Episode": ", ".join(str(e) for e in p["episodes"]),
        "Cut": " + ".join(f"{s['start']}–{s['end']}" for s in p["segments"]),
        "Category": p.get("category", ""), "Length (s)": p["length_s"], "Type": p["type"].capitalize(),
        "Hook": p["hook"], "Cliffhanger": p["cliffhanger"], "Why it works": p["why"],
    } for p in rows])


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown(f'<div class="side-brand">{logo_tag()}<span>Promo Finder</span></div>',
                unsafe_allow_html=True)

    st.markdown(f'<div class="side-h"><span class="ic">{ICON["model"]}</span>AI model</div>', unsafe_allow_html=True)
    provider = st.selectbox("Provider", list(PROVIDERS), format_func=lambda k: PROVIDERS[k][0],
                            label_visibility="collapsed")
    model = st.selectbox("Model", PROVIDERS[provider][1], label_visibility="collapsed")
    st.markdown('<div class="side-note">Switching models takes two settings and no code changes.</div>',
                unsafe_allow_html=True)

    st.markdown(f'<div class="side-h"><span class="ic">{ICON["key"]}</span>API key</div>', unsafe_allow_html=True)
    has_demo = bool(demo_key(provider))
    own_key = st.text_input("API key", type="password",
                            placeholder="Optional: a demo key is built in" if has_demo
                            else f"Paste your {PROVIDERS[provider][0]} key",
                            label_visibility="collapsed")
    if has_demo and not own_key:
        st.markdown(f'<div class="side-note">A demo key is built in, so you can run it right away '
                    f'(up to {DEMO_MAX_EPISODES} episodes per run). Paste your own key to use your '
                    f'quota and run more.</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="side-note">Get one free at {KEY_HELP[provider]}. Used for this session '
                    f'only, never stored.</div>', unsafe_allow_html=True)
    using_demo = has_demo and not own_key
    api_key = own_key or (demo_key(provider) if has_demo else "")

    st.markdown(f'<div class="side-h"><span class="ic">{ICON["run"]}</span>Run settings</div>', unsafe_allow_html=True)
    limit = st.number_input("Episodes to process", 0, 500, 5,
                            help="0 processes every episode. Start with 5: free keys are rate-limited.")
    pause = st.slider("Pause between calls (s)", 0, 20, 7,
                      help="Raise this if a free key hits rate limits.")

    ready = bool(api_key)
    st.markdown(
        f'<div class="side-status"><span class="dot" style="background:{"var(--good)" if ready else "var(--red)"}"></span>'
        f'{"Ready to run with the demo key" if (ready and using_demo) else "Ready to run" if ready else "Add an API key to run a new show"}<br>'
        f'<span style="color:var(--muted)">{esc(PROVIDERS[provider][0])}, {esc(model)}</span></div>',
        unsafe_allow_html=True)

os.environ["LLM_PROVIDER"] = provider
os.environ["LLM_MODEL"] = model
if api_key:
    os.environ[KEY_ENV[provider]] = api_key.strip()


# ---------------------------------------------------------------- hero
st.markdown(f"""
<div class="hero">
  <div class="hero-logo">{logo_tag()}</div>
  <h1>Microdrama Promo Finder</h1>
  <p>Upload a show and get the 20 to 60 second clips worth posting first. Each one opens on a
  hook that stops the scroll and cuts on a cliffhanger, ready for an editor to approve.</p>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- results rendering
def clip_bar(length):
    pct = min(length / 60, 1) * 100
    return (f'<div class="clip"><div class="fill" style="width:{pct:.0f}%"></div>'
            f'<div class="min" style="left:33.3%"></div></div>'
            f'<div class="clip-scale"><span>0s</span><span>20s min</span><span>60s max</span></div>')


def promo_card(p, top=False):
    where = " + ".join(f"Episode {s['episode']}, {s['start']} to {s['end']}" for s in p["segments"])
    kind = "Stitched from two moments" if p["type"] == "stitched" else "One continuous clip"
    return f"""
<div class="promo {'top' if top else ''}">
  <div class="promo-head">
    <span class="rank">{'#' + str(p['rank']) if p.get('rank') else ''}</span>
    <span class="where">{esc(where)}</span>
    <span class="chip cat">{esc(p.get('category', 'Other drama'))}</span>
    <span class="chip">{p['length_s']:.0f}s</span>
    <span class="chip">{kind}</span>
    <span class="chip score">Score {p['score']}</span>
  </div>
  {clip_bar(p['length_s'])}
  <div class="beats">
    <span class="k">Opens on</span><span>{esc(p['hook'])}</span>
    <span class="k">Cuts on</span><span>{esc(p['cliffhanger'])}</span>
  </div>
  <div class="why">{esc(p['why'])}</div>
</div>"""


def show_results(data, key="example"):
    ranked, built = data["ranked"], data["promos"]
    stitched = sum(p["type"] == "stitched" for p in built)
    st.markdown(f"""
<div class="stats">
  <div class="stat"><div class="v">{data.get('episodes', '—')}{f'<span style="font-size:1rem;color:var(--muted)"> of {data["uploaded"]}</span>' if data.get('uploaded') else ''}</div>
    <div class="l">Episodes processed</div></div>
  <div class="stat"><div class="v">{len(built)}</div><div class="l">Valid promos built</div></div>
  <div class="stat"><div class="v">{len(ranked)}</div><div class="l">On the shortlist</div></div>
  <div class="stat"><div class="v">{stitched}</div><div class="l">Stitched from two moments</div></div>
</div>""", unsafe_allow_html=True)

    # ---- one view at a time, switched with buttons right under the stats
    from collections import Counter
    counts = Counter(p.get("category", "Other drama") for p in built)
    view = st.segmented_control("View", ["Top 5", "By category", "Download all"],
                                default="Top 5", label_visibility="collapsed",
                                key=f"view_{key}") or "Top 5"

    if view == "Top 5":
        st.markdown("### Top 5 to post first")
        for i, p in enumerate(ranked[:5]):
            st.markdown(promo_card(p, top=(i == 0)), unsafe_allow_html=True)

    elif view == "By category":
        st.markdown("### Best promos by category")
        st.caption("Themes come from the type of each hook and the question each cliffhanger "
                   "leaves open. Pick one to see its three best promos.")
        options = [c for c, _ in counts.most_common()]
        choice = st.segmented_control("Category", options, default=options[0] if options else None,
                                      label_visibility="collapsed", key=f"cat_{key}")
        if choice:
            for p in sorted((p for p in built if p.get("category") == choice),
                            key=lambda p: -p["score"])[:3]:
                st.markdown(promo_card(p), unsafe_allow_html=True)

    else:
        st.markdown("### Download all promos")
        st.caption(f"The ranked shortlist and all {len(built)} valid promos, with episode, "
                   f"timestamps, category, hook, cliffhanger and why it works.")
        all_sorted = sorted(built, key=lambda p: -p["score"])
        d1, d2, d3 = st.columns(3)
        d1.download_button(f"Shortlist, top {len(ranked)} (CSV)", promos_frame(ranked).to_csv(index=False),
                           "ranked_shortlist.csv", "text/csv", use_container_width=True,
                           key=f"dl1_{key}")
        d2.download_button(f"All {len(built)} promos (CSV)", promos_frame(all_sorted).to_csv(index=False),
                           "all_promos.csv", "text/csv", use_container_width=True, key=f"dl2_{key}")
        d3.download_button("All promos (JSON)",
                           json.dumps({"count": len(built), "rejected_pairs": data.get("rejected", {}),
                                       "promos": built}, indent=1),
                           "all_promos.json", "application/json", use_container_width=True,
                           key=f"dl3_{key}")

    rej = data.get("rejected") or {}
    if rej:
        with st.expander("What the code rejected, and why"):
            labels = {"cliff_before_hook": "Cliffhanger came before the hook",
                      "too_long": "Longer than 60 seconds", "too_short": "Shorter than 20 seconds",
                      "weak_cliffhanger": "Ending already resolved",
                      "stitch_hook_already_continuous": "Stitch not needed, a continuous cut exists",
                      "stitch_over_cap": "Over the stitched-clip limit"}
            for k, v in rej.items():
                st.markdown(f"- {labels.get(k, k)}: **{v}**")
            st.caption("The model suggests moments. These rules decide which become promos.")


# ---------------------------------------------------------------- pipeline run
def extract_uploads(files, folder):
    folder.mkdir(parents=True, exist_ok=True)
    for f in files:
        if f.name.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(f.getvalue())) as z:
                z.extractall(folder)
        else:
            (folder / f.name).write_bytes(f.getvalue())
    txt_dirs = {p.parent for p in folder.rglob("*.txt")}
    return max(txt_dirs, key=lambda d: len(list(d.glob("*.txt")))) if txt_dirs else folder


def run_pipeline(script_dir, limit, pause):
    show = screenplay_parser.load_show(script_dir)
    uploaded = len(show)
    if limit:
        show = show[:limit]
    if not show:
        st.error("No screenplay files found. Upload .txt episodes or a .zip that contains them.")
        return None
    candidates.SECONDS_BETWEEN_CALLS = pause
    bar = st.progress(0.0, text=f"Read {len(show)} episodes. Finding hooks and cliffhangers…")
    note = st.empty()
    hooks, cliffs, failed = [], [], []
    for i, ep in enumerate(show, 1):
        note.caption(f"Episode {ep['episode']}")
        try:
            ans = candidates.call_llm_json(candidates.build_prompt(ep, compact=candidates.COMPACT),
                                           system=candidates.EDITOR_SYSTEM)
            kept, _ = candidates.validate_moments(ep, ans)
            hooks += kept["hooks"]
            cliffs += kept["cliffhangers"]
        except Exception as e:
            failed.append(ep["episode"])
            note.warning(f"Episode {ep['episode']} skipped: {str(e)[:140]}")
        bar.progress(i / len(show), text=f"{i} of {len(show)} episodes")
    bar.empty(); note.empty()
    if failed:
        st.warning(f"Episode(s) {', '.join(map(str, failed))} could not be processed, usually a free-key "
                   f"rate limit. Raise the pause in the left menu or run again later.")
    if not hooks or not cliffs:
        st.error("No usable moments came back. Check the API key and model in the sidebar.")
        return None
    cands = {"hooks": hooks, "cliffhangers": cliffs, "provider": provider, "model": model}
    built, rejected = promo_builder.build(show, cands)
    return {"promos": built, "ranked": promo_builder.rank(built), "rejected": rejected,
            "episodes": len(show) - len(failed), "uploaded": uploaded, "failed": failed}


# ---------------------------------------------------------------- upload panel (main action)
key_done = bool(api_key)
with st.container(border=True, key="upload_panel"):
    st.markdown(f"""
<span class="upload-anchor"></span>
<div class="panel-head">
  <h3>Find promos in your show</h3>
  <div class="checks">
    <span class="check ok"><i></i>Model: {esc(PROVIDERS[provider][0])}</span>
    <span class="check {'ok' if key_done else 'todo'}"><i></i>{'Demo key active' if using_demo else 'API key added' if key_done else 'API key needed'}</span>
  </div>
</div>""", unsafe_allow_html=True)
    files = st.file_uploader("Episode scripts (.zip or .txt)", type=["zip", "txt"],
                             accept_multiple_files=True, label_visibility="collapsed",
                             help="One .txt per episode with timestamped beats, e.g. Start: (00:11.30) | End: (00:13.10)")
    c1, c2 = st.columns([1, 2.4], vertical_alignment="center")
    go = c1.button("Find promos", type="primary", use_container_width=True,
                   disabled=not (files and api_key))
    if not key_done:
        c2.markdown('<div class="hint">Choose a model and paste your API key in the left menu, '
                    'then upload a .zip of episodes. Free keys work.</div>', unsafe_allow_html=True)
    elif not files:
        c2.markdown('<div class="hint">Upload a .zip of episodes, or individual .txt files. '
                    f'The first {int(limit) or "all"} episodes will be processed '
                    '(change this in the left menu).</div>', unsafe_allow_html=True)
    else:
        c2.markdown(f'<div class="hint">Ready. {len(files)} file(s) added.</div>', unsafe_allow_html=True)

if go:
    work = Path(tempfile.mkdtemp())
    try:
        llm.CACHE_DIR = work / "cache"
        run_limit = int(limit)
        if using_demo and (run_limit == 0 or run_limit > DEMO_MAX_EPISODES):
            run_limit = DEMO_MAX_EPISODES
            st.info(f"Using the built-in demo key, so this run covers the first {DEMO_MAX_EPISODES} "
                    f"episodes. Paste your own key in the left menu to process more.")
        result = run_pipeline(extract_uploads(files, work / "scripts"), run_limit, int(pause))
        if result:
            st.session_state["result"] = result
    finally:
        shutil.rmtree(work, ignore_errors=True)

# ---------------------------------------------------------------- results + explanation
mine = st.session_state.get("result")
tab_results, tab_how = st.tabs(["Your promos" if mine else "Example results", "How it works"])

with tab_results:
    if mine:
        done_n, up_n = mine.get("episodes", 0), mine.get("uploaded", 0)
        if up_n and done_n < up_n:
            st.success(f"Done. Processed {done_n} of {up_n} uploaded episodes. To process more, raise "
                       f"“Episodes to process” in the left menu (0 = all) and run again.")
        else:
            st.success("Done. Your promos are ranked below.")
        show_results(mine, key="mine")
    else:
        sample = load_sample()
        if not sample:
            st.markdown('<div class="empty"><b>Your ranked promos will appear here</b>'
                        'Upload a show above to get started.</div>', unsafe_allow_html=True)
        else:
            st.caption(f"Example: a completed run on {sample.get('episodes', 'all')} episodes of one show "
                       f"using {sample.get('model', 'the default model')}. Upload your own show above to "
                       f"replace it.")
            ranked = sorted((p for p in sample["promos"] if p.get("rank")), key=lambda p: p["rank"])
            show_results({"promos": sample["promos"], "ranked": ranked,
                          "rejected": sample.get("rejected_pairs", {}),
                          "episodes": sample.get("episodes", "—")})

with tab_how:
    st.markdown("""
### AI judges, code checks

**Read the scripts.** Every line, action and camera note becomes a numbered beat with exact
timestamps. The AI never reads or writes a timestamp, so it cannot invent one.

**Find the moments.** For each episode the AI marks where a promo could start (a slap, a reveal,
a cruel line) and where it could cut (the beat right before an answer lands).

**Check the picks.** Code drops any beat the AI made up and any "cut to black" transition.

**Build the clips.** Code pairs hooks with later cliffhangers into 20 to 60 second clips,
snapped to camera cuts so the hook still lands in the first 3 seconds.

**Rank them.** Hook strength, how unresolved the ending feels, whether a new viewer would follow
it, and a preference for early episodes, where new viewers start. Near-duplicates are removed
and the shortlist is spread across the show.

An editor approves the final list. This tool replaces watching every episode, not the editor.
""")
