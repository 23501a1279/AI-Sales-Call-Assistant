import streamlit as st
from transformers import pipeline
import tempfile
import os
import time
import json
import re
import whisper
from typing import Dict, Any, List, Tuple
from audiorecorder import audiorecorder


# =========================================
# Configuration
# =========================================

INTENT_LABELS = [
    "product_query",
    "pricing_request",
    "issue_report",
    "positive_feedback",
    "negative_feedback",
    "scheduling_request",
    "accessory_inquiry",
    "service_booking",
    "referral_inquiry",
    "loyalty_program_inquiry",
]

PRODUCT_KEYWORDS = [
    "pro-max package", "gold subscription", "premium tier", "billing cycle",
    "dashboard", "api integration", "support center", "trial period",
    "account manager", "renewal date", "gemini", "google", "microsoft",
    "apple", "sony", "x-series", "pro-edition", "plan a", "plan b",
    "lease agreement", "financing option", "test drive", "vehicle inspection",
    "extended warranty", "down payment", "monthly payment", "trade-in value",
    "sedan", "suv", "truck", "hybrid", "electric vehicle",
    "ford", "toyota", "honda", "tesla", "bmw", "mercedes", "audi", "chevy", "nissan",
    "insurance", "protection plan", "first service", "service appointment",
    "service benefits", "loyalty program", "upgrade option", "referral",
    "friend", "accessories", "add-ons",
]

NUMBER_MAP = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "eleven": "11", "twelve": "12",
}

# =========================================
# Model loaders (cached)
# =========================================

@st.cache_resource
def load_sentiment_model():
    try:
        return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    except Exception as e:
        st.error(f"Error loading Sentiment model: {e}")
        return None

@st.cache_resource
def load_intent_model():
    try:
        return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    except Exception as e:
        st.error(f"Error loading Intent model: {e}")
        return None

@st.cache_resource
def load_ner_model():
    try:
        return pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)
    except Exception as e:
        st.error(f"Error loading NER model: {e}")
        return None

@st.cache_resource
def load_whisper_model(model_name: str = "base"):
    try:
        return whisper.load_model(model_name)
    except Exception as e:
        st.error(f"Error loading Whisper model '{model_name}': {e}")
        return None

with st.spinner("Initializing AI models... (first run downloads weights)"):
    sentiment_model = load_sentiment_model()
    intent_model = load_intent_model()
    ner_model = load_ner_model()
    whisper_model = load_whisper_model("base") or load_whisper_model("tiny")

# =========================================
# Utilities
# =========================================

def split_sentences(text: str) -> List[str]:
    # More permissive sentence splitter than a simple '.'
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    # Trim, drop empties
    return [p.strip() for p in parts if p.strip()]

def extract_structured_entities(text: str) -> Dict[str, List[str]]:
    data = {
        "prices": [], "numbers": [], "phones": [], "emails": [],
        "products": [], "brands": [], "models": []
    }
    tl = text.lower()

    price_regex = r'(\$|€|£)\s*(\d[\d,.]*)|(\d[\d,.]*)\s*(dollars|euros|pounds|usd|gbp|eur)|(\d+\.\d{2})\b|\b(?:price|cost|for|worth)\s+is?\s*(\d[\d,.]*)'
    prices_found = re.findall(price_regex, text, re.IGNORECASE)
    for m in prices_found:
        pv = ""
        if m[0] and m[1]: pv = f"{m[0]}{m[1]}"
        elif m[2] and m[3]: pv = f"{m[2]} {m[3]}"
        elif m[4]: pv = m[4]
        elif m[5]: pv = m[5]
        if pv and pv.strip() not in data["prices"]:
            data["prices"].append(pv.strip())

    model_regex = r'(?:[a-zA-Z]+\s*)?(\d{2,}[\w\d]+)|(?:model|version|edition|series|trim|corolla|camry|silverado|accord|civic|f150)\s+([\w\d-]+)'
    models_found = re.findall(model_regex, text)
    for m in models_found:
        name = m[0] if m[0] else m[1]
        if name:
            data["models"].append(name)

    email_regex = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    data["emails"].extend(re.findall(email_regex, text))

    phone_regex = r'(?:(?:\+|00)\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?\d{2,4}[-.\s]?\d{4})'
    data["phones"].extend([p.strip() for p in re.findall(phone_regex, text) if len(p.strip()) > 5])

    car_brands = ['ford', 'toyota', 'honda', 'tesla', 'bmw', 'mercedes', 'audi', 'chevy', 'nissan']
    for kw in PRODUCT_KEYWORDS:
        if kw in tl:
            if kw in car_brands or kw in ['gemini', 'google', 'microsoft', 'apple', 'sony']:
                data["brands"].append(kw.title())
            else:
                data["products"].append(kw.title())

    numbers_found = re.findall(r'\b(\d+)\b', text)
    for word, digit in NUMBER_MAP.items():
        if re.search(r'\b' + word + r'\b', tl):
            numbers_found.append(digit)
    # Dedup
    data["numbers"].extend(sorted(set(numbers_found)))

    # Cleanup: remove overlaps
    final_data = {}
    for k in data:
        unique_list = sorted(set(data[k]))
        if k == "numbers":
            specific = set()
            for p in data["prices"]: specific.update(re.findall(r'\d+', p))
            for p in data["phones"]: specific.update(re.findall(r'\d+', p))
            for m in data["models"]: specific.update(re.findall(r'\d+', m))
            unique_list = [n for n in unique_list if n not in specific]
        final_data[k] = unique_list
    return final_data

def analyze_text(text: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if sentiment_model:
        s = sentiment_model(text)[0]
        out["sentiment_label"] = s["label"].capitalize()
        out["sentiment_output"] = f"Emotion: {out['sentiment_label']} (Score: {s['score']:.2f})"
    else:
        out["sentiment_label"] = "Unknown"
        out["sentiment_output"] = "Emotion: Model Failed to Load."

    if intent_model:
        intent_res = intent_model(text, INTENT_LABELS, multi_label=False)
        out["intent_label"] = intent_res["labels"][0]
        display = out["intent_label"].replace("_", " ").capitalize()
        out["intent_output"] = f"Intent: {display} (Score: {intent_res['scores'][0]:.2f})"
    else:
        out["intent_label"] = "unknown_intent"
        out["intent_output"] = "Intent: Model Failed to Load."

    entities = extract_structured_entities(text)
    out["structured_entities"] = entities
    out["entities_output"] = json.dumps(entities, indent=2)
    return out

# =========================================
# Reasoning layer implementations
# =========================================

def direct_prompting_suggestions(analysis: Dict[str, Any], transcript_segment: str) -> Dict[str, str]:
    intent = analysis.get("intent_label", "unknown_intent")
    sentiment = analysis.get("sentiment_label", "Neutral")
    entities = analysis.get("structured_entities", {})

    suggestions = {
        "next_question": "What is the key factor influencing your decision (e.g., price, features, reliability)?",
        "objection_handling": "Thank you for sharing your concerns. I'm here to ensure we find the perfect match for you.",
        "recommendation": "Suggest discussing overall requirements first.",
    }

    if intent == "pricing_request":
        suggestions["next_question"] = "Are you looking for a lower monthly payment or a better long-term financing rate?"
        suggestions["objection_handling"] = "I hear you. Let me break down the total cost of ownership and why this delivers long-term value."
        suggestions["recommendation"] = "Highlight a cost-effective model or flexible payment plan."
    elif intent == "product_query":
        brand = entities.get("brands", [])
        brand_name = brand[0] if brand else "vehicle"
        suggestions["next_question"] = f"Which features matter most in a {brand_name} (e.g., safety, efficiency, capacity)?"
        suggestions["objection_handling"] = "Great question! I can pull up a clear comparison right now."
        suggestions["recommendation"] = f"Recommend a balanced mid-range {brand_name} and offer a test drive."
    elif intent == "scheduling_request":
        suggestions["next_question"] = "Which date and time works best for your test drive, inspection, or delivery?"
        suggestions["objection_handling"] = "We can work around your schedule. What time window suits you this week?"
        suggestions["recommendation"] = "Send a calendar invite immediately with appointment details."
    elif intent == "accessory_inquiry":
        suggestions["next_question"] = "Do you prefer practical accessories or performance/aesthetic upgrades?"
        suggestions["objection_handling"] = "All accessories are genuine and warranty-approved. I can show top sellers."
        suggestions["recommendation"] = "Upsell a protection bundle that includes key accessories."
    elif intent == "service_booking":
        suggestions["next_question"] = "Are you available next Tuesday or Wednesday morning for your first service?"
        suggestions["objection_handling"] = "We value your time. Shall I arrange free pickup and drop-off?"
        suggestions["recommendation"] = "Confirm free first service benefits and set service reminders."
    elif intent == "referral_inquiry":
        friend_entity = "friend" if "friend" in transcript_segment.lower() else "contact"
        suggestions["next_question"] = f"Is your {friend_entity} looking for the same model or something different?"
        suggestions["objection_handling"] = "Referrals get priority and special offers. We can arrange a VIP test drive."
        suggestions["recommendation"] = "Send a referral bonus link and reach out to the friend asap."
    elif intent == "loyalty_program_inquiry":
        suggestions["next_question"] = "Which benefit matters more: discounted servicing or priority access to upgrades?"
        suggestions["objection_handling"] = "Value grows over time. I can show 3-year savings examples."
        suggestions["recommendation"] = "Enroll them now and outline the upgrade pathway."
    elif intent == "negative_feedback" or sentiment == "Negative":
        suggestions["next_question"] = "Can you share exactly what caused the frustration so I can fix it immediately?"
        suggestions["objection_handling"] = "I understand. I'll take ownership and provide a clear path forward."
        suggestions["recommendation"] = "De-escalate and shift to listening/problem-solving mode."

    return suggestions

def template_based_suggestions(analysis: Dict[str, Any], transcript_segment: str) -> Dict[str, str]:
    intent = analysis.get("intent_label", "unknown_intent")
    sentiment = analysis.get("sentiment_label", "Neutral")
    entities = analysis.get("structured_entities", {})

    # Predictable rules
    rules = [
        {
            "if": lambda s, i: s == "Negative",
            "next_question": "I’m sorry that happened. What would a good resolution look like for you today?",
            "objection_handling": "Thank you for telling me—your experience matters. Let me fix this quickly.",
            "recommendation": "Pause pitching, focus on resolution and clarity.",
        },
        {
            "if": lambda s, i: i == "pricing_request",
            "next_question": "Is your priority the upfront price or lower monthly payments?",
            "objection_handling": "We can tailor financing to match your comfort.",
            "recommendation": "Offer a lower-cost alternative and compare total ownership cost.",
        },
        {
            "if": lambda s, i: i == "product_query",
            "next_question": "Top 3 priorities: performance, safety, or budget—what’s first?",
            "objection_handling": "I’ll shortlist options aligned to your priorities.",
            "recommendation": "Provide a concise comparison with one clear recommendation.",
        },
        {
            "if": lambda s, i: i == "service_booking",
            "next_question": "Would Tuesday morning or Wednesday afternoon be better?",
            "objection_handling": "We can arrange pickup and drop-off for convenience.",
            "recommendation": "Confirm booking and send reminders.",
        },
    ]

    # Default
    suggestions = {
        "next_question": "What matters most to you right now: price, features, or timing?",
        "objection_handling": "I hear you. Let’s simplify and get you exactly what you need.",
        "recommendation": "Summarize needs and propose one clear next step.",
    }

    for r in rules:
        try:
            if r["if"](sentiment, intent):
                suggestions = {
                    "next_question": r["next_question"],
                    "objection_handling": r["objection_handling"],
                    "recommendation": r["recommendation"],
                }
                break
        except Exception:
            continue

    # Optional light personalization with entities
    brand = (entities.get("brands") or entities.get("products") or [])
    if brand:
        b = brand[0]
        suggestions["recommendation"] += f" Consider a tailored option around: {b}."

    return suggestions

def build_llm_debug_prompt(analysis: Dict[str, Any], transcript_segment: str) -> str:
    return f"""
You are a real-time sales assistant.
Customer Transcript Segment: "{transcript_segment}"
Customer Sentiment: {analysis.get('sentiment_label')}
Customer Intent: {analysis.get('intent_label')}
Extracted Entities: {json.dumps(analysis.get('structured_entities', {}), indent=2)}

Based on this, suggest:
1. The next best question.
2. A soft objection-handling response.
3. A strategic product/action recommendation.
"""

# =========================================
# Transcription
# =========================================

def transcribe_audio(audio_path: str) -> str:
    if not whisper_model:
        st.error("Transcription model failed to load. Cannot transcribe.")
        return ""
    st.info("Transcribing audio with Whisper...")
    try:
        result = whisper_model.transcribe(audio_path)
        return result.get("text", "")
    except Exception as e:
        st.error(f"Transcription Error: {e}")
        return ""

# =========================================
# Streamlit UI
# =========================================

st.set_page_config(page_title="AI Sales Call Assistant", layout="wide")
st.title("🎙️ AI Sales Call Assistant")
st.caption("Milestone 4: Generate AI Suggestions from .wav recordings using Whisper + NLP + Reasoning Layer.")

with st.sidebar:
    st.markdown("### Analysis settings")
    reasoning_mode = st.selectbox(
        "AI Reasoning Layer",
        ["Direct Prompting", "Template-based"],
        help="Choose how suggestions are generated from NLP outputs."
    )
    customer_filter_mode = st.selectbox(
        "Customer sentence filter",
        ["Analyze every sentence", "Analyze every 2nd sentence (simulate alternate speakers)"],
        help="If you don’t have diarization yet, this approximates customer turns."
    )

    st.markdown("---")
    st.markdown("### Whisper settings")
    st.caption("Using cached 'base' (fallback to 'tiny' if needed).")

tab1, tab2 = st.tabs(["Upload audio files", "Live speech recognization"])

with tab1:
    st.subheader("Batch process multiple recordings")
    files = st.file_uploader(
        "Upload .wav/.mp3/.m4a recordings",
        type=["wav", "mp3", "m4a"],
        accept_multiple_files=True
    )
    if files:
        st.success(f"{len(files)} file(s) ready.")
        if st.button("Analyze all files"):
            if not all([sentiment_model, intent_model, whisper_model]):
                st.error("One or more models failed to load.")
            else:
                for uf in files:
                    st.markdown(f"### File: {uf.name}")
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uf.name)[1]) as tmpf:
                        tmpf.write(uf.getvalue())
                        tmp_path = tmpf.name

                    transcript = transcribe_audio(tmp_path)
                    os.remove(tmp_path)

                    if not transcript:
                        st.warning("Empty transcript or transcription failed.")
                        continue

                    st.markdown("**Transcript preview:**")
                    st.code(transcript[:1000] + ("..." if len(transcript) > 1000 else ""), language="text")

                    sentences = split_sentences(transcript)
                    st.info(f"Detected {len(sentences)} sentences.")

                    analyze_every_second = (customer_filter_mode == "Analyze every 2nd sentence (simulate alternate speakers)")
                    for idx, sent in enumerate(sentences, start=1):
                        use_sentence = (not analyze_every_second) or (analyze_every_second and idx % 2 == 0)
                        if not use_sentence:
                            continue
                        if not sent.strip():
                            continue

                        analysis = analyze_text(sent)
                        if reasoning_mode == "Direct Prompting":
                            suggestions = direct_prompting_suggestions(analysis, sent)
                        else:
                            suggestions = template_based_suggestions(analysis, sent)

                        llm_prompt = build_llm_debug_prompt(analysis, sent)

                        with st.container(border=True):
                            st.markdown(f"**Customer Sentence {idx}:** {sent}")
                            st.markdown(f"- **Sentiment:** {analysis['sentiment_output']}")
                            st.markdown(f"- **Intent:** {analysis['intent_output']}")
                            st.markdown("**Entities:**")
                            st.code(analysis["entities_output"], language="json")

                            st.warning("AI Suggestions")
                            st.markdown(f"- **Next question:** {suggestions['next_question']}")
                            st.markdown(f"- **Objection handling:** {suggestions['objection_handling']}")
                            st.markdown(f"- **Recommendation:** {suggestions['recommendation']}")

                            with st.expander("LLM Prompt Context (debug)"):
                                st.code(llm_prompt, language="text")

                            time.sleep(0.05)

                    st.success("File analysis complete.")
                st.success("Batch analysis complete.")

with tab2:
    st.subheader("🎤 Record live speech and analyze immediately")
    audio = audiorecorder("🎙️ Start Recording", "⏹ Stop Recording")

    if audio is not None and len(audio) > 0:
        # Preview the recorded audio
        audio_bytes = audio.export(format="wav").read()
        st.audio(audio_bytes, format="audio/wav")

        if st.button("Analyze Live Speech"):
            # Save temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                audio.export(tmp.name, format="wav")
                tmp_path = tmp.name

            transcript = transcribe_audio(tmp_path)
            os.remove(tmp_path)

            if not transcript:
                st.warning("Empty transcript or transcription failed.")
            else:
                st.markdown("**Transcription:**")
                st.code(transcript, language="text")

                sentences = split_sentences(transcript)
                st.info(f"Detected {len(sentences)} sentences.")
                analyze_every_second = (customer_filter_mode == "Analyze every 2nd sentence (simulate alternate speakers)")

                for idx, sent in enumerate(sentences, start=1):
                    use_sentence = (not analyze_every_second) or (analyze_every_second and idx % 2 == 0)
                    if not use_sentence or not sent.strip():
                        continue

                    analysis = analyze_text(sent)
                    suggestions = (
                        direct_prompting_suggestions(analysis, sent)
                        if reasoning_mode == "Direct Prompting"
                        else template_based_suggestions(analysis, sent)
                    )
                    llm_prompt = build_llm_debug_prompt(analysis, sent)

                    with st.container(border=True):
                        st.markdown(f"**Customer Sentence {idx}:** {sent}")
                        st.markdown(f"- **Sentiment:** {analysis['sentiment_output']}")
                        st.markdown(f"- **Intent:** {analysis['intent_output']}")
                        st.markdown("**Entities:**")
                        st.code(analysis["entities_output"], language="json")

                        st.warning("AI Suggestions")
                        st.markdown(f"- **Next question:** {suggestions['next_question']}")
                        st.markdown(f"- **Objection handling:** {suggestions['objection_handling']}")
                        st.markdown(f"- **Recommendation:** {suggestions['recommendation']}")

                        with st.expander("LLM Prompt Context (debug)"):
                            st.code(llm_prompt, language="text")

                st.success("Live speech analysis complete.")
        else:
            st.info("Click 'Analyze Live Speech' after recording to process the audio.")
