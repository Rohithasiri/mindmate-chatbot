import streamlit as st
import datetime as dt
import re
import json
from typing import Tuple, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import random

# ---- Ensure VADER resources ----
try:
    SIA = SentimentIntensityAnalyzer()
except:
    nltk.download('vader_lexicon')
    SIA = SentimentIntensityAnalyzer()

# ---- Model Setup (LLM) ----
MODEL_NAME = "EleutherAI/gpt-neo-125M"  # small, CPU-friendly
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# ---- App Config ----
st.set_page_config(
    page_title="MindMate – Mental Health Chatbot (LLM Demo)",
    page_icon="💬",
    layout="centered",
)

# ---- Utilities ----
RISK_KEYWORDS = {
    "immediate": [
        r"suicid(e|al)", r"kill myself", r"end my life", r"i don't want to live",
        r"self[-\s]?harm(ing)?", r"overdose", r"jump off", r"hang myself", r"cut myself",
    ],
    "high": [
        r"i'm a danger to myself", r"i might hurt myself", r"i want to disappear",
        r"no reason to live", r"can't go on",
    ],
}

EMERGENCY_MESSAGE = (
    "I’m not a substitute for a professional. If you’re in immediate danger, "
    "please contact local emergency services or a trusted person nearby. "
    "If available in your region, consider reaching out to a crisis helpline."
)

COPING_LIBRARY = {
    "breathing_478": {
        "title": "4–7–8 Breathing",
        "steps": [
            "Inhale quietly through your nose for 4 seconds.",
            "Hold the breath for 7 seconds.",
            "Exhale through pursed lips for 8 seconds.",
            "Repeat 4 times, focusing on the pace.",
        ],
    },
    "box_breathing": {
        "title": "Box Breathing (4×4)",
        "steps": [
            "Inhale for 4 seconds.",
            "Hold for 4 seconds.",
            "Exhale for 4 seconds.",
            "Hold for 4 seconds, repeat 1–3 minutes.",
        ],
    },
    "grounding_54321": {
        "title": "Grounding (5–4–3–2–1)",
        "steps": [
            "Name 5 things you can see.",
            "Name 4 things you can feel.",
            "Name 3 things you can hear.",
            "Name 2 things you can smell.",
            "Name 1 thing you can taste.",
        ],
    },
    "journaling": {
        "title": "2-Minute Journal Prompt",
        "steps": [
            "What am I feeling right now?",
            "What triggered it today?",
            "What is one thing I can control in the next hour?",
        ],
    },
}

SAFE_TIPS = [
    "Drink a glass of water and take 3 slow breaths.",
    "Step outside for 2 minutes if possible.",
    "Put on a calming song or sound for 3 minutes.",
    "Message a supportive friend or family member.",
]

def assess_risk(text: str) -> Tuple[str, str]:
    t = text.lower()
    for pat in RISK_KEYWORDS["immediate"]:
        if re.search(pat, t):
            return "immediate", pat
    for pat in RISK_KEYWORDS["high"]:
        if re.search(pat, t):
            return "high", pat
    return "low", ""

def analyze_sentiment(text: str) -> Dict[str, float]:
    return SIA.polarity_scores(text)

def generate_llm_reply(user_input: str) -> str:
    """
    Generate an empathetic response with repetition penalty and trimmed output.
    """
    prompt = f"You are an empathetic mental health support bot. Respond kindly and helpfully.\nuser: {user_input}\nassistant:"

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.2,
        top_p=0.9
    )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    text = text.split("assistant:")[-1].strip()
    text = text.split("\n\n")[0].strip()  # take first paragraph only
    return text

# ---- Session State ----
if "messages" not in st.session_state:
    st.session_state.messages = []
if "mood_log" not in st.session_state:
    st.session_state.mood_log = []

# ---- Sidebar: Mood Check-in ----
st.sidebar.header("Daily Check‑in 🌤️")
with st.sidebar:
    mood = st.slider("How are you feeling right now?", 0, 10, 5)
    note = st.text_input("Add a short note (optional)")
    if st.button("Save Check‑in"):
        st.session_state.mood_log.append({
            "ts": dt.datetime.now().isoformat(timespec='seconds'),
            "mood": mood,
            "note": note.strip(),
        })
        st.success("Saved your check‑in.")
    if st.session_state.mood_log:
        st.subheader("Your Recent Mood")
        last = st.session_state.mood_log[-10:]
        st.line_chart({"mood": [entry["mood"] for entry in last]})
        st.download_button(
            "Download mood log (JSON)",
            data=json.dumps(st.session_state.mood_log, indent=2).encode("utf-8"),
            file_name="mood_log.json",
            mime="application/json",
        )

# ---- Header ----
st.title("💬 MindMate – Mental Health Chatbot (LLM Demo)")
st.caption("Educational demo. Not a medical device.")

# ---- Chat History ----
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---- Chat Input ----
user_input = st.chat_input("Share what's on your mind…")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    risk_level, _ = assess_risk(user_input)
    sentiment = analyze_sentiment(user_input)

    llm_reply = generate_llm_reply(user_input)

    # Prepend emergency message if needed
    if risk_level in ("immediate", "high"):
        llm_reply = f"**I’m really glad you reached out. Your safety matters a lot.**\n\n{EMERGENCY_MESSAGE}\n\n{llm_reply}"

    st.session_state.messages.append({"role": "assistant", "content": llm_reply})

    with st.chat_message("user"):
        st.markdown(user_input)
    with st.chat_message("assistant"):
        st.markdown(llm_reply)

    # --- Context-aware coping exercise suggestion ---
    suggestion_key = None
    compound = sentiment["compound"]
    lower_input = user_input.lower()

    if compound < -0.3:  # strong negative sentiment
        suggestion_key = random.choice(["breathing_478", "box_breathing"])
    elif any(word in lower_input for word in ["overwhelm", "anxious", "panic", "stressed"]):
        suggestion_key = "grounding_54321"
    else:
        suggestion_key = "journaling"

    if not suggestion_key:
        suggestion_key = random.choice(list(COPING_LIBRARY.keys()))

    suggestion = COPING_LIBRARY[suggestion_key]

    st.session_state.messages.append({
        "role": "assistant",
        "content": f"💡 Based on what you shared, you might try: **{suggestion['title']}**\n" +
                   "\n".join([f"{i+1}. {step}" for i, step in enumerate(suggestion['steps'])])
    })

    with st.chat_message("assistant"):
        st.markdown(f"💡 Based on what you shared, you might try: **{suggestion['title']}**\n" +
                    "\n".join([f"{i+1}. {step}" for i, step in enumerate(suggestion['steps'])]))

# ---- Footer ----
st.divider()
st.markdown(
    "**Privacy note:** Messages stay in-browser for this demo. Refreshing resets chat.\n\n"
    "**If in crisis or thinking about harming yourself or others,** contact local emergency services immediately."
)
