
import streamlit as st
from transformers import pipeline
import tempfile
import os
import time
import json
import re
import whisper
from audiorecorder import audiorecorder

# --- Intent Labels Configuration ---
INTENT_LABELS = [
    'product_query',
    'pricing_request',
    'issue_report',
    'positive_feedback',
    'negative_feedback',
    'scheduling_request'
]

# --- Mock Product/Service Keywords (For augmenting NER results) ---
PRODUCT_KEYWORDS = [
    'pro-max package',
    'gold subscription',
    'premium tier',
    'billing cycle',
    'dashboard',
    'api integration',
    'support center',
    'trial period',
    'account manager',
    'renewal date',
    'gemini',
    'google',
    'microsoft',
    'apple',
    'sony',
    'x-series',
    'pro-edition',
    'plan a',
    'plan b'
]

# --- 1. Load Models ---
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
        # grouped_entities=True gives combined spans which is easier to work with
        return pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)
    except Exception as e:
        st.error(f"Error loading NER model: {e}")
        return None

@st.cache_resource
def load_whisper_model():
    try:
        return whisper.load_model("base")
    except Exception as e:
        st.error(f"Error loading Whisper model: {e}")
        return None

# Load models with spinner
with st.spinner("Initializing AI Models (may take a minute on first run)..."):
    sentiment_model = load_sentiment_model()
    intent_model = load_intent_model()
    ner_model = load_ner_model()
    whisper_model = load_whisper_model()

# --- Improved Structured Entity Extraction Function ---
def clean_ner_token(token: str) -> str:
    """Clean the token returned by HF NER (remove '##' artifacts and extra spaces)."""
    if not token:
        return token
    token = token.replace("##", "")   # remove subword marker
    token = token.strip()
    return token

def extract_structured_entities(text: str) -> dict:
    """
    Extracts entities into specific, structured categories using:
      - HuggingFace NER (if available)
      - Regex (prices, phones, emails, models)
      - Keyword matching for product/brands
    Returns a dict with lists for keys: prices, numbers, phones, emails, products, brands, models
    """
    structured_data = {
        'prices': [],
        'numbers': [],
        'phones': [],
        'emails': [],
        'products': [],
        'brands': [],
        'models': []
    }

    if not text or not text.strip():
        return structured_data

    text_original = text
    text_lower = text_original.lower()

    # ----- 0) Use NER model if available (grouped entities are easier) -----
    try:
        if ner_model:
            ner_results = ner_model(text_original)
            # ner_results is a list of dicts with keys: entity_group, score, word, start, end (grouped_entities=True)
            for ent in ner_results:
                group = ent.get("entity_group", "").upper()
                word = clean_ner_token(ent.get("word", ""))
                if not word:
                    continue

                # Map some entity groups to our categories (best-effort)
                # DSLIM NER often returns ORG, LOC, PER, MISC.
                if group in ("ORG",):
                    # Organizations -> likely brand
                    structured_data['brands'].append(word)
                elif group in ("MISC", "PRODUCT", "WORK_OF_ART", "EVENT"):
                    # MISC/Product-like -> product
                    structured_data['products'].append(word)
                elif group in ("PER",):
                    # person - ignore for these structured categories
                    pass
                else:
                    # If it's something else, don't fail — try to classify small numeric tokens
                    # e.g., numbers may not be returned by this model; we'll capture via regex
                    pass
    except Exception as e:
        # Non-fatal: if NER fails, proceed with regex extraction
        st.warning(f"NER model error (continuing with regex-only extraction): {e}")

    # ----- 1) Price extraction (permissive) -----
    # Matches: $500, €10.99, 500 dollars, 1,200.00, price is 300
    price_regex = re.compile(
        r'(?:(?:\$|€|£)\s*\d{1,3}(?:[,\d]*)(?:\.\d{1,2})?)'        # $500, $1,200.00
        r'|(?:\d{1,3}(?:[,\d]*)(?:\.\d{1,2})?\s*(?:dollars|usd|euros|eur|pounds|gbp))'  # 500 dollars
        r'|(?:price(?:\s+is)?\s*(?:around|about)?\s*\d{1,3}(?:[,\d]*)(?:\.\d{1,2})?)'  # price is 500
        , flags=re.IGNORECASE)

    for m in price_regex.findall(text_original):
        # m is the matched text (as string) because we used findall with non-capturing groups
        if isinstance(m, tuple):  # safety — but pattern uses non-capturing groups so should be string
            matched = "".join([part for part in m if part])
        else:
            matched = m
        matched = matched.strip()
        if matched and matched not in structured_data['prices']:
            structured_data['prices'].append(matched)

    # ----- 2) Email extraction -----
    email_regex = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', flags=re.IGNORECASE)
    emails_found = email_regex.findall(text_original)
    for e in emails_found:
        if e not in structured_data['emails']:
            structured_data['emails'].append(e)

    # ----- 3) Phone extraction (more permissive) -----
    phone_regex = re.compile(r'(?:(?:\+|00)\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?\d{2,4}[-.\s]?\d{3,4})')
    phones_found = phone_regex.findall(text_original)
    # Note: findall may return tuples sometimes; ensure flattening
    if phones_found:
        # phones_found sometimes returns matches as strings; ensure we cast
        candidates = []
        for ph in phones_found:
            if isinstance(ph, tuple):
                ph = "".join(ph)
            ph = ph.strip()
            if ph and len(re.sub(r'\D', '', ph)) >= 6:  # simple length filter
                candidates.append(ph)
        for p in candidates:
            if p not in structured_data['phones']:
                structured_data['phones'].append(p)

    # ----- 4) Model/Version extraction (alphanumeric) -----
    # captures e.g. "X500", "iPhone 16", "X-500 Pro", "Model X5000"
    model_regex = re.compile(r'\b([A-Za-z]+[-]?\d{2,4}(?:[- ]?[A-Za-z0-9]+)?)\b')
    # Also patterns like "model X500", "version 2.0"
    extra_model_regex = re.compile(r'\b(?:model|version|series|edition)\s+([A-Za-z0-9\-]+)\b', flags=re.IGNORECASE)

    models_found = model_regex.findall(text_original)
    extra_models = extra_model_regex.findall(text_original)
    for m in models_found + extra_models:
        m_clean = clean_ner_token(m)
        # avoid capturing common words like 'the', 'and' etc by simple alnum check
        if re.search(r'\d', m_clean):  # require at least one digit to be considered a 'model' in this heuristic
            if m_clean not in structured_data['models']:
                structured_data['models'].append(m_clean)

    # ----- 5) Keyword matches for known products/brands -----
    for keyword in PRODUCT_KEYWORDS:
        if keyword.lower() in text_lower:
            if keyword.lower() in ['gemini', 'google', 'microsoft', 'apple', 'sony']:
                if keyword.title() not in structured_data['brands']:
                    structured_data['brands'].append(keyword.title())
            else:
                if keyword.title() not in structured_data['products']:
                    structured_data['products'].append(keyword.title())

    # Also deduplicate NER-found brands/products (some might have been added)
    # Normalize brands/products (strip and title-case)
    structured_data['brands'] = list(dict.fromkeys([clean_ner_token(b).title() for b in structured_data['brands'] if b]))
    structured_data['products'] = list(dict.fromkeys([clean_ner_token(p).title() for p in structured_data['products'] if p]))

    # ----- 6) General number extraction (catch-all), but avoid numbers that appear inside phones/prices/models -----
    number_regex = re.compile(r'\b\d{1,6}\b')
    numbers_found = number_regex.findall(text_original)

    # collect digits from prices/phones/models to exclude them from generic numbers
    exclude_nums = set()
    for p in structured_data['prices']:
        exclude_nums.update(re.findall(r'\d+', p))
    for ph in structured_data['phones']:
        exclude_nums.update(re.findall(r'\d+', ph))
    for m in structured_data['models']:
        exclude_nums.update(re.findall(r'\d+', m))

    for n in numbers_found:
        if n not in exclude_nums and n not in structured_data['numbers']:
            structured_data['numbers'].append(n)

    # Final cleanup: ensure uniqueness and sort small lists for stable output
    for k in structured_data:
        # keep insertion order while removing duplicates
        seen = []
        for v in structured_data[k]:
            v_str = v.strip()
            if v_str and v_str not in seen:
                seen.append(v_str)
        structured_data[k] = seen

    return structured_data

# --- 2. Analysis Function ---
def analyze_customer_sentence(text):
    """Performs Sentiment, Intent, and Structured Entity analysis on a given text."""
    results = {}

    # 1. Sentiment Analysis
    if sentiment_model:
        try:
            sent_result = sentiment_model(text)[0]
            label = sent_result['label'].capitalize()
            score = f"{sent_result['score']:.2f}"
            results['sentiment'] = f"Emotion: {label} (Score: {score})"
        except Exception as e:
            results['sentiment'] = f"Emotion: Error ({e})"
    else:
        results['sentiment'] = "Emotion: Model Failed to Load."

    # 2. Intent Recognition (Zero-Shot Classification)
    if intent_model:
        try:
            intent_result = intent_model(text, INTENT_LABELS, multi_label=False)
            top_intent = intent_result['labels'][0].replace('_', ' ').capitalize()
            intent_score = f"{intent_result['scores'][0]:.2f}"
            results['intent'] = f"Intent: {top_intent} (Score: {intent_score})"
        except Exception as e:
            results['intent'] = f"Intent: Error ({e})"
    else:
        results['intent'] = "Intent: Model Failed to Load."

    # 3. Structured Entity Extraction
    structured_entities = extract_structured_entities(text)

    # Filter empty lists for display clarity
    filtered = {k: v for k, v in structured_entities.items() if v}
    results['entities'] = json.dumps(filtered, indent=2) if filtered else "No entities detected"

    return results

# --- 3. Transcription Function ---
def transcribe_audio(audio_path):
    """Transcribes audio using the Whisper model."""
    if not whisper_model:
        st.error("Transcription model failed to load. Cannot transcribe.")
        return "Transcription Failed."

    st.info("Transcribing audio with Whisper...")
    try:
        result = whisper_model.transcribe(audio_path)
        return result.get("text", "")
    except Exception as e:
        st.error(f"Transcription Error: {e}")
        return "Transcription Failed."

# --- 4. Streamlit UI ---
st.set_page_config(page_title="AI Sales Call Assistant", layout="wide")
st.title("🎙️ AI Sales Call Assistant ")
st.caption("Analyzing Customer Sentiment, Intent, and Entities from audio transcripts.")

st.markdown("""
<style>
    .stApp {
        background-color: #0d1117;
        color: white;
    }
    h1, h2, h3, h4, h5 {
        color: #c9d1d9;
    }
    .stButton>button {
        background-color: #238636;
        color: white;
        border-radius: 6px;
        border: 1px solid #30363d;
    }
</style>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Upload Audio File (.wav, .mp3)", "Live Speech Segment"])

# --- TAB 1: UPLOAD FULL CALL ---
with tab1:
    uploaded_file = st.file_uploader("Upload a .wav, .mp3, or .m4a sales call recording:", type=['wav', 'mp3', 'm4a'])

    if uploaded_file is not None:
        st.audio(uploaded_file, format=uploaded_file.type)
        st.success(f"File uploaded: {uploaded_file.name}")

        # Save uploaded file to a temporary location for Whisper
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        if st.button("Analyze Full Call Transcript"):
            # Check models
            if not all([sentiment_model, intent_model, whisper_model]):
                st.error("One or more analysis models failed to load. Please check the logs.")
                os.remove(tmp_file_path)
            else:
                transcript = transcribe_audio(tmp_file_path)

                st.markdown("---")
                st.markdown("### Customer Sentences & Comprehensive Analysis")

                # Basic sentence segmentation
                sentences = re.split(r'(?<=[.!?])\s+', transcript.strip())

                for idx, sentence in enumerate(sentences, start=1):
                    sentence = sentence.strip()
                    # Example: analyze every sentence (you originally analyzed every 2nd to mimic customer speech)
                    if sentence:
                        analysis_results = analyze_customer_sentence(sentence)

                        sentiment_output = analysis_results['sentiment']
                        intent_output = analysis_results['intent']
                        entities_output = analysis_results['entities']

                        st.markdown(
                            f"""
                            <div style='background-color:#1e1e1e; padding: 15px; border-radius: 8px; margin: 15px 0; border: 1px solid #333;'>
                                <p style='margin: 0; color: #ff69b4; font-weight: bold;'>Customer - Sentence {idx}:</p>
                                <p style='margin: 5px 0 10px 0; color: #ffffff;'>{sentence}</p>
                                <p style='margin: 5px 0 0 0; color: #ff4d4d;'>{sentiment_output}</p>
                                <p style='margin: 5px 0 0 0; color: #4dffff;'>{intent_output}</p>
                                <p style='margin: 5px 0 0 0; color: #4dff88;'>Entities:</p>
                                <pre style='color: #4dff88; background-color: transparent; border: none; white-space: pre-wrap; margin-top: 5px;'>{entities_output}</pre>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        time.sleep(0.05)

                st.success("Analysis Complete!")
                os.remove(tmp_file_path)

# --- TAB 2: LIVE SPEECH SEGMENT ---
with tab2:

    audio = audiorecorder("🎙️ Start Recording", "⏹ Stop Recording")

    if audio is not None and len(audio) > 0:
        audio_bytes = audio.export(format="wav").read()
        st.audio(audio_bytes, format="audio/wav")

        if st.button("Analyze Live Speech"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                audio.export(tmp.name, format="wav")
                tmp_path = tmp.name

            transcript = transcribe_audio(tmp_path)

            st.markdown(f"**Transcription:** <span style='color: #ffcc00; font-weight: bold;'>{transcript}</span>", unsafe_allow_html=True)
            st.markdown("---")

            analysis_results = analyze_customer_sentence(transcript)

            sentiment_output = analysis_results['sentiment']
            intent_output = analysis_results['intent']
            entities_output = analysis_results['entities']

            st.markdown("### 🎧 Live Speech Analysis Results")

            st.markdown(
                f"""
                <div style='background-color:#1e1e1e; padding: 15px; border-radius: 8px; margin: 15px 0; border: 1px solid #333;'>
                    <p style='color:#ff4d4d;'>{sentiment_output}</p>
                    <p style='color:#4dffff;'>{intent_output}</p>
                    <p style='color:#4dff88;'>Entities:</p>
                    <pre style='color: #4dff88; background-color: transparent; border: none; white-space: pre-wrap; margin-top: 5px;'>{entities_output}</pre>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.success("Real-time analysis completed!")
            os.remove(tmp_path)
        else:
            st.info("Click 'Analyze Live Speech' after recording to process the audio.")
