"""
MedScribe Pipeline
==================
Orchestrates MedASR (speech-to-text) + MedGemma (text-to-SOAP) + Clinical Tools.

Three HAI-DEF model uses:
- MedASR: 105M param Conformer, 5.2% WER on medical dictation
- MedGemma (fine-tuned): LoRA adapter for concise SOAP notes
- MedGemma (base): instruction-tuned for Billable Diagnoses, patient summary, completeness,
  differential diagnosis, and medication interaction checks
"""
import os
import re
import time
import json
import torch
import librosa
import numpy as np
from pathlib import Path

try:
    from transformers import AutoModelForCTC, AutoProcessor
    MEDASR_AVAILABLE = True
except ImportError:
    MEDASR_AVAILABLE = False

from src.inference import SOAPGenerator


# ============================================================
# CTC DECODING
# ============================================================
def _manual_ctc_decode(token_ids, blank_id, id_to_token):
    """Collapses duplicates and filters blanks for CTC/RNN-T models."""
    result = []
    prev_id = None
    for token_id in token_ids:
        if token_id != prev_id and token_id != blank_id:
            result.append(id_to_token.get(token_id, ""))
        prev_id = token_id
    return "".join(result)


# ============================================================
# TRANSCRIPT POST-PROCESSING
# ============================================================
def _clean_transcript(raw_text):
    """
    Clean MedASR manual CTC decode output:
    1. CTC stutter-killer (collapse 3+ repeated chars)
    2. SentencePiece ▁ → spaces
    3. Special token + formatting token → punctuation
    4. Bracket/brace removal, spacing, capitalization
    """
    if not raw_text:
        return ""

    text = re.sub(r'(.)\1{2,}', r'\1', raw_text)
    text = text.replace("▁", " ")
    text = re.sub(r'</?s>|<epsilon>|<pad>|<unk>', '', text)

    punctuation_map = {
        r'\{+[\s]*period[\s]*\}+': '.',
        r'\{+[\s]*comma[\s]*\}+': ',',
        r'\{+[\s]*colon[\s]*\}+': ':',
        r'\{+[\s]*semicolon[\s]*\}+': ';',
        r'\{+[\s]*new[\s]*paragraph[\s]*\}+': '\n',
        r'\{+[\s]*question[\s]*mark[\s]*\}+': '?',
        r'\{+[\s]*exclamation[\s]*\}+': '!',
    }
    for pattern, replacement in punctuation_map.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    text = re.sub(r'\[[^\]]*\]', '', text)
    text = re.sub(r'[{}]', '', text)
    text = re.sub(r'\s+([.,;:?!])', r'\1', text)
    text = re.sub(r'([.,;:?!])([A-Za-z])', r'\1 \2', text)
    text = re.sub(r'\n\s+', '\n', text)
    text = re.sub(r' {2,}', ' ', text)
    text = text.strip()
    text = text.strip(' .,;:')

    if text:
        text = text[0].upper() + text[1:]
    text = re.sub(r'(?<=[.!?]\s)([a-z])', lambda m: m.group(1).upper(), text)

    return text.strip()

# ============================================================
# CLINICAL OUTPUT POST-PROCESSING
# ============================================================
def _clean_tool_output(text):
    """Strip model artifacts from clinical tool output."""
    if not text:
        return text

    # Strip 'Final Answer' blocks
    text = re.split(r'Final Answer:', text, maxsplit=1)[0]
    text = re.split(r'The final answer is', text, maxsplit=1)[0]

    # Strip second+ 'Rationale:' sections
    parts = text.split('Rationale:')
    if len(parts) > 2:
        text = parts[0] + 'Rationale:' + parts[1]

    # Strip LaTeX box artifacts
    text = re.sub(r'\$\\boxed\{[^}]*\}\$', '', text)

    # Strip lines with "not documented in source" / "not specified in source"
    lines = text.split('\n')
    lines = [l for l in lines if "not documented in source" not in l.lower()
             and "not specified in source" not in l.lower()]
    text = '\n'.join(lines).strip()

    # Strip trailing incomplete headers
    for tail in ["**RANKED", "**NOTE", "**SUMMARY", "RANKED", "**"]:
        if text.rstrip().endswith(tail):
            text = text[:text.rfind(tail)].strip()

    return text.strip()


class MedScribePipeline:
    """Full voice-to-SOAP pipeline using HAI-DEF models."""

    MEDASR_MODEL_ID = "google/medasr"
    SAMPLE_RATE = 16000

    def __init__(self, adapter_path="./models/checkpoints/medgemma_v2_soap/final_model"):
        self.soap_gen = SOAPGenerator(adapter_path=adapter_path)
        self.asr_model = None
        self.asr_processor = None
        self._asr_loaded = False
        self._inv_vocab = None
        self._blank_id = None

    # --------------------------------------------------------
    # LOADING
    # --------------------------------------------------------
    def load_asr(self):
        if self._asr_loaded:
            return
        if not MEDASR_AVAILABLE:
            raise ImportError("MedASR requires transformers>=5.0.0")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.asr_processor = AutoProcessor.from_pretrained(self.MEDASR_MODEL_ID)
        self.asr_model = AutoModelForCTC.from_pretrained(self.MEDASR_MODEL_ID).to(device)
        self.asr_model.eval()

        vocab = self.asr_processor.tokenizer.get_vocab()
        self._inv_vocab = {v: k for k, v in vocab.items()}
        self._blank_id = self.asr_processor.tokenizer.pad_token_id

        self._asr_loaded = True
        print(f"[MedASR] Loaded on {device} — manual CTC decode + normalized audio")

    def load_soap(self):
        self.soap_gen.load()

    def load_all(self):
        self.load_asr()
        self.load_soap()

    # --------------------------------------------------------
    # TRANSCRIPTION
    # --------------------------------------------------------
    def transcribe(self, audio_path):
        if not self._asr_loaded:
            raise RuntimeError("MedASR not loaded.")

        speech, sr = librosa.load(audio_path, sr=self.SAMPLE_RATE, mono=True)
        speech = librosa.util.normalize(speech)
        audio_duration = len(speech) / self.SAMPLE_RATE

        device = next(self.asr_model.parameters()).device
        inputs = self.asr_processor(
            speech, sampling_rate=self.SAMPLE_RATE,
            return_tensors="pt", padding=True,
        ).to(device)

        start = time.time()
        with torch.inference_mode():
            logits = self.asr_model(**inputs).logits

        predicted_ids = torch.argmax(logits, dim=-1)[0]
        raw = _manual_ctc_decode(
            predicted_ids.tolist(), self._blank_id, self._inv_vocab
        )
        elapsed = time.time() - start

        clean = _clean_transcript(raw)

        print(f"[MedASR] Transcribed {audio_duration:.1f}s audio in {elapsed:.1f}s")
        print(f"[MedASR] RAW (first 200): {raw[:200]}")
        print(f"[MedASR] CLEAN (first 200): {clean[:200]}")

        return {
            "transcript": clean,
            "raw_transcript": raw,
            "time_s": round(elapsed, 1),
            "audio_duration_s": round(audio_duration, 1),
        }

    # --------------------------------------------------------
    # SOAP GENERATION
    # --------------------------------------------------------
    def generate_soap(self, transcript):
        return self.soap_gen.generate(transcript)

    def generate_soap_base(self, transcript):
        """Generate SOAP with adapter disabled (base MedGemma) for comparison."""
        return self.soap_gen.generate_base(transcript)

    # --------------------------------------------------------
    # CLINICAL TOOLS
    # All tools return dict: {"text": str, "time_s": float}
    # --------------------------------------------------------
    def suggest_billable_diagnoses(self, soap_note):
        """Extract billable diagnoses from SOAP note for ICD-10 coding."""
        prompt = (
            "You are a clinical coding assistant. From the following SOAP note, "
            "list each billable clinical diagnosis or finding that warrants an "
            "ICD-10-CM code. For each, state the diagnosis and the specific "
            "documentation evidence that supports it.\n"
            "Do NOT include ICD-10 code numbers.\n"
            "Do NOT include negative findings.\n"
            "Do NOT include symptoms already explained by a listed diagnosis.\n"
            "Maximum 5 diagnoses. Be concise.\n\n"
            f"SOAP NOTE:\n{soap_note}\n\n"
            "BILLABLE DIAGNOSES:"
        )
        result = self.soap_gen.generate_freeform(prompt, max_new_tokens=200)
        result["text"] = _clean_tool_output(result["text"])
        return result

    def patient_summary(self, soap_note):
        """Generate plain-language patient visit summary."""
        prompt = (
            "You are a patient communication assistant. Rewrite the following "
            "clinical SOAP note as a brief, plain-language summary that a patient "
            "can understand. Avoid medical jargon. Use simple, clear sentences. "
            "Include: what was found, what it means, and what happens next.\n"
            "Do not repeat yourself. Do not include rationale, final answer, or "
            "summary sections.\n\n"
            f"SOAP NOTE:\n{soap_note}\n\n"
            "PATIENT SUMMARY:"
        )
        result = self.soap_gen.generate_freeform(prompt, max_new_tokens=300)
        result["text"] = _clean_tool_output(result["text"])
        return result

    def completeness_check(self, soap_note):
        """Review SOAP note for documentation gaps."""
        prompt = (
            "You are a clinical documentation quality reviewer. Review this SOAP "
            "note for documentation gaps. Check for:\n"
            "- Subjective complaints without corresponding objective findings\n"
            "- Diagnoses without supporting evidence in Objective\n"
            "- Missing follow-up or monitoring in Plan\n"
            "- Medications without dosage or frequency\n"
            "- Missing allergies or medication reconciliation\n\n"
            "Be concise. List specific gaps found. If complete, say so.\n"
            "Do not repeat yourself. Do not include rationale, final answer, or "
            "summary sections. Stop after the list.\n\n"
            f"SOAP NOTE:\n{soap_note}\n\n"
            "DOCUMENTATION REVIEW:"
        )
        result = self.soap_gen.generate_freeform(prompt, max_new_tokens=250)
        result["text"] = _clean_tool_output(result["text"])
        return result

    def differential_diagnosis(self, soap_note):
        """Generate differential diagnosis list from SOAP note."""
        prompt = (
            "You are a clinical reasoning assistant. Based on the following SOAP note, "
            "generate a ranked differential diagnosis list. For each diagnosis:\n"
            "- State the diagnosis\n"
            "- Briefly explain supporting evidence from the note\n"
            "- Note any findings that argue against it\n\n"
            "Rank from most likely to least likely. Include 3-5 differentials.\n"
            "Do not repeat yourself. Do not include rationale, final answer, or "
            "summary sections. Stop after the numbered list.\n\n"
            f"SOAP NOTE:\n{soap_note}\n\n"
            "DIFFERENTIAL DIAGNOSIS:"
        )
        result = self.soap_gen.generate_freeform(prompt, max_new_tokens=300)
        result["text"] = _clean_tool_output(result["text"])
        return result

    def medication_check(self, soap_note):
        """Check medications mentioned for interactions and safety concerns."""
        # Pre-check: skip model call if no medications are documented
        med_indicators = [
            "mg", "daily", "bid", "tid", "qid", "prn", "mcg", "units",
            "tablet", "capsule", "inhaler", "patch", "cream", "drops",
            "metformin", "lisinopril", "atorvastatin", "omeprazole",
            "amlodipine", "losartan", "aspirin", "heparin", "warfarin",
            "enoxaparin", "insulin", "sertraline", "gabapentin",
            "hydrochlorothiazide", "furosemide", "prednisone",
        ]
        note_lower = soap_note.lower()
        if not any(ind in note_lower for ind in med_indicators):
            return {
                "text": "No specific medications documented in this note. "
                        "Medication review requires named drugs with dosages.",
                "time_s": 0.0,
            }

        prompt = (
            "You are a clinical pharmacist assistant. Review the medications mentioned "
            "in this SOAP note. For each medication identified:\n"
            "- Note the medication and its indication (if clear)\n"
            "- Flag potential drug-drug interactions\n"
            "- Flag contraindications based on patient conditions in the note\n"
            "- Note if dosage/frequency is missing or appears incorrect\n\n"
            "If no medications are mentioned, state that.\n"
            "Do not repeat yourself. Do not include rationale, final answer, or "
            "summary sections. Stop after the review.\n\n"
            f"SOAP NOTE:\n{soap_note}\n\n"
            "MEDICATION REVIEW:"
        )
        result = self.soap_gen.generate_freeform(prompt, max_new_tokens=300)
        result["text"] = _clean_tool_output(result["text"])
        return result

    # --------------------------------------------------------
    # FULL PIPELINE
    # --------------------------------------------------------
    def transcribe_and_generate(self, audio_path):
        asr_result = self.transcribe(audio_path)
        soap_result = self.generate_soap(asr_result["transcript"])
        return {
            "transcript": asr_result["transcript"],
            "soap_note": soap_result["soap_note"],
            "asr_time_s": asr_result["time_s"],
            "soap_time_s": soap_result["time_s"],
            "total_time_s": round(asr_result["time_s"] + soap_result["time_s"], 1),
            "audio_duration_s": asr_result["audio_duration_s"],
            "tokens": soap_result["tokens"],
            "word_count": soap_result["word_count"],
        }

    @property
    def asr_loaded(self):
        return self._asr_loaded

    @property
    def soap_loaded(self):
        return self.soap_gen.is_loaded

    @property
    def fully_loaded(self):
        return self._asr_loaded and self.soap_gen.is_loaded