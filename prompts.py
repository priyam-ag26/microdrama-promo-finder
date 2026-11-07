"""
All prompts live here, separate from the code, so they are identical whichever model runs them
and easy to improve without touching the pipeline.
"""

EDITOR_SYSTEM = """You are a senior promo editor for vertical-video microdramas (the style of \
ReelShort and DramaBox ads on TikTok, Instagram Reels and YouTube Shorts). You pick moments that \
make a scrolling viewer stop in the first 3 seconds and need to know what happens next. You are \
strict: most moments in a script are NOT promo-worthy. You answer only in valid JSON."""

FIND_MOMENTS = """Below is one episode of a microdrama screenplay, broken into numbered beats.
Each beat line is: BEAT_ID [start-end seconds] TYPE SPEAKER (tone): text
TYPE is CAMERA (shot description), ACTION, SOUND FX or DIALOGUE.

Episode {episode} of the show.
Logline: {logline}
What happened before: {continuity}

BEATS:
{beats}

TASK
A promo is a 20-60 second clip. It STARTS on a hook and ENDS on a cliffhanger, and the
cliffhanger must come 20-60 seconds after the hook. So you must find cut points spread across
the whole episode, not only at its end.

1. HOOKS: find up to 5 beats where a promo clip could START. A hook must grab a stranger who
   has never seen the show within 3 seconds, with no setup: a slap or physical violence, a
   shocking reveal, a cruel humiliating line, a public accusation, a threat, a betrayal, or a
   charged romantic moment. A calm line that only matters if you know the plot is NOT a hook.
2. CLIFFHANGERS: find up to 5 beats where a promo clip could END. The best promo cut point is
   the beat RIGHT BEFORE an answer, reveal, decision or blow lands: a question is asked but not
   yet answered, a hand is raised but has not struck, a secret is about to come out. The viewer
   must be left with a specific unresolved question.
   - At least 3 of your cliffhangers must be in the middle of the episode, not in its last
     15 seconds.
   - Never pick a "cut to black", "fade out" or music-swell beat; pick the story moment itself.
   - Do not rely on words like "cliffhanger" in the script; judge the moment yourself.
   - Do not pick a beat after the tension is already resolved.
Scores are 1-10 and must be honest: 9-10 only for moments that would clearly stop a scroller.
For "why", explain what the viewer FEELS or WANTS to know, do not just repeat the line.

RULES
- Use beat IDs exactly as written above. Never invent an ID or a timestamp.
- Prefer DIALOGUE, ACTION or SOUND FX beats; use a CAMERA beat only if the key moment is visual.

Keep every text field under 15 words.

Return JSON exactly in this shape and nothing else:
{{
  "hooks": [{{"beat_id": "E01-B012", "type": "slap|reveal|humiliation|accusation|threat|betrayal|romance|other",
             "score": 8, "why": "what makes a stranger stop scrolling",
             "context_needed": "what a new viewer must already know, or 'none'"}}],
  "cliffhangers": [{{"beat_id": "E01-B090", "open_question": "the question the viewer is left with",
                    "score": 7, "why": "why it leaves them hanging"}}]
}}"""
