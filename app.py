# app_paraphrase_fixed.py
import streamlit as st
import torch
import re
from transformers import AutoTokenizer, T5ForConditionalGeneration

st.set_page_config(page_title="Paraphrase —", layout="centered")
st.title("Paraphrase ")
st.markdown("Masukkan kalimat → Generate → Hasil.")

# ---------- Fixed decode params (tidak diubah di UI) ----------
MODEL_PATH = "tuner007/t5_paraphrase" 
MAX_INPUT_LEN = 128
MAX_OUTPUT_LEN = 64
NUM_BEAMS = 4
NUM_RETURN = 4
NO_REPEAT_NGRAM = 3
REPETITION_PENALTY = 1.2
LENGTH_PENALTY = 0.8
DO_SAMPLE_FALLBACK = True   # sampling hanya untuk fallback
FALLBACK_TEMPERATURE = 1.2
FALLBACK_TOP_P = 0.95
FALLBACK_TOP_K = 50

# ---------- Helpers ----------
def clean_sentence(s: str) -> str:
    s = s.strip()
    s = s.strip('"').strip("'")
    s = re.sub(r'^[“”`´]+', '', s)
    s = re.sub(r'[“”`´]+$', '', s)
    s = re.sub(r'\s+', ' ', s)
    if s:
        s = s[0].upper() + s[1:]
    return s

def canonical_key(s: str) -> str:
    return re.sub(r'[^0-9a-z]+', '', s.lower())

def first_sentence_only(s: str) -> str:
    """
    Ambil hanya kalimat pertama dari string s.
    Definisi kalimat: dipisah oleh ., ?, atau !
    Kembalikan string yang berakhiran titik jika perlu.
    """
    # normalize spaces
    s = s.strip()
    if not s:
        return s
    # split on sentence enders keeping them
    parts = re.split(r'(?<=[.!?])\s+', s)
    first = parts[0].strip()
    # if first doesn't end with punctuation, add a dot
    if not re.search(r'[.!?]$', first):
        first = first + "."
    return first

# ---------- Load tokenizer & model ----------
@st.cache_resource
def load_model_and_tokenizer(path: str):
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
    model = T5ForConditionalGeneration.from_pretrained(path)
    return tokenizer, model

try:
    tokenizer, _model = load_model_and_tokenizer(MODEL_PATH)
except Exception as e:
    st.error(f"Gagal memuat model/tokenizer dari '{MODEL_PATH}': {e}")
    st.stop()

device = "cuda" if torch.cuda.is_available() else "cpu"

# wrapper supaya mirip gaya snippet lama (pakai model.model)
class _Wrap: pass
model = _Wrap()
model.model = _model
model.model.to(device)
model.model.eval()

if device == "cpu":
    st.warning("Menjalankan model di CPU — proses generate bisa lambat. Untuk hasil cepat, gunakan mesin dengan GPU.")

# ---------- UI: input tanpa value default ----------
sentence = st.text_area("Masukkan kalimat untuk diparaphrase:", value="", height=140)

# ---------- Generate ----------
if st.button("🚀 Generate"):
    if not sentence.strip():
        st.warning("Masukkan kalimat dulu.")
    else:
        prompt = "paraphrase: " + sentence.strip()
        encoding = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_INPUT_LEN,
            padding="max_length",
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        try:
            with st.spinner("Sedang generate (beam search)..."):
                with torch.no_grad():
                    outputs = model.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_length=MAX_OUTPUT_LEN,
                        num_beams=NUM_BEAMS,
                        num_return_sequences=NUM_RETURN,
                        no_repeat_ngram_size=NO_REPEAT_NGRAM,
                        repetition_penalty=REPETITION_PENALTY,
                        length_penalty=LENGTH_PENALTY,
                        do_sample=False,   # deterministic primary pass
                    )

            # decode, ambil kalimat pertama, dedup & filter identik
            orig_key = canonical_key(sentence)
            candidates = []
            seen_keys = set()

            for out in outputs:
                raw = tokenizer.decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                # hapus prefix 'paraphrase:' jika ada
                if raw.lower().startswith("paraphrase:"):
                    raw = raw[len("paraphrase:"):].strip()
                # ambil kalimat pertama saja
                first = first_sentence_only(raw)
                first = clean_sentence(first)
                if len(first) < 3:
                    continue
                key = canonical_key(first)
                if key == orig_key:
                    continue
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                candidates.append(first)

            chosen = None

            # jika ada kandidat valid dari beam outputs, pilih yang paling berbeda dari original
            if candidates:
                # pilih kandidat paling berbeda berdasarkan difflib ratio
                import difflib
                best = None
                best_score = -1.0
                for c in candidates:
                    sim = difflib.SequenceMatcher(None, canonical_key(sentence), canonical_key(c)).ratio()
                    diff = 1.0 - sim
                    if diff > best_score:
                        best_score = diff
                        best = c
                chosen = best

            # fallback: jika tidak ada kandidat (semua identik), lakukan sampling sekali untuk coba alternatif
            if chosen is None and DO_SAMPLE_FALLBACK:
                with st.spinner("Tidak menemukan kandidat unik — mencoba fallback sampling..."):
                    with torch.no_grad():
                        out_sample = model.model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            max_length=MAX_OUTPUT_LEN,
                            num_beams=1,
                            num_return_sequences=4,
                            do_sample=True,
                            temperature=FALLBACK_TEMPERATURE,
                            top_p=FALLBACK_TOP_P,
                            top_k=FALLBACK_TOP_K,
                            no_repeat_ngram_size=0,
                            repetition_penalty=1.05,
                        )
                # proses sampling outputs
                for out in out_sample:
                    raw = tokenizer.decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    if raw.lower().startswith("paraphrase:"):
                        raw = raw[len("paraphrase:"):].strip()
                    first = first_sentence_only(raw)
                    first = clean_sentence(first)
                    if len(first) < 3:
                        continue
                    key = canonical_key(first)
                    if key == orig_key:
                        continue
                    if key in seen_keys:
                        continue
                    chosen = first
                    break  # ambil hasil sampling pertama yang valid

            # tampilkan hasil atau fallback message
            st.subheader("Original")
            st.write(sentence)

            st.subheader("Paraphrase")
            if chosen:
                st.success(chosen)
            else:
                st.info("Tidak ada paraphrase unik yang ditemukan. Coba ubah kalimat input atau gunakan model yang di-fine-tune untuk paraphrase.")

        except RuntimeError as e:
            st.error(f"RuntimeError saat generate: {e}. Coba jalankan di mesin dengan GPU atau kurangi ukuran input.")
        except Exception as e:
            st.error(f"Terjadi error: {e}")
