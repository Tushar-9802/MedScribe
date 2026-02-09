# import torch
# import librosa
# import huggingface_hub
# import re
# from transformers import AutoModelForCTC, AutoProcessor

# def manual_ctc_decode(ids, blank_id, id_to_token):
#     """Collapses duplicates and filters blanks for CTC/RNN-T models."""
#     result = []
#     prev_id = None
#     for token_id in ids:
#         if token_id != prev_id and token_id != blank_id:
#             result.append(id_to_token[token_id])
#         prev_id = token_id
#     return "".join(result)

# def process_for_medgemma(raw_text):
#     """
#     Cleans raw MedASR output into a narrative format 
#     suitable for LLM prompt context.
#     """
#     # 1. Handle word separators and special tokens
#     text = raw_text.replace("▁", " ")
    
#     # 2. Map punctuation tokens to actual characters
#     replacements = {
#         r"\{period\}": ".",
#         r"\{comma\}": ",",
#         r"\{colon\}": ":",
#         r"\{new paragraph\}": "\n",
#         r"\[": "", # Remove headers for raw transcript
#         r"\]": "",
#         r"<epsilon>": "",
#         r" {2,}": " "  # Collapse multiple spaces
#     }
    
#     for pattern, sub in replacements.items():
#         text = re.sub(pattern, sub, text)
        
#     # 3. Fix space-before-punctuation artifacts (e.g., "lobe . " -> "lobe.")
#     text = re.sub(r'\s+([.,:])', r'\1', text)
    
#     return text.strip()

# def run_pipeline():
#     model_id = "google/medasr"
#     device = "cuda" if torch.cuda.is_available() else "cpu"
    
#     # Initialize
#     processor = AutoProcessor.from_pretrained(model_id)
#     model = AutoModelForCTC.from_pretrained(model_id).to(device)
    
#     # Load Audio (Ensuring 16kHz for MedASR consistency)
#     audio_path = huggingface_hub.hf_hub_download(model_id, 'test_audio.wav')
#     speech, _ = librosa.load(audio_path, sr=16000)
    
#     # Inference
#     inputs = processor(speech, sampling_rate=16000, return_tensors="pt").to(device)
#     with torch.no_grad():
#         logits = model(**inputs).logits
        
#     # Decoding Logic
#     predicted_ids = torch.argmax(logits, dim=-1)[0]
#     blank_id = processor.tokenizer.pad_token_id
#     id_to_token = processor.tokenizer.get_vocab()
#     inv_vocab = {v: k for k, v in id_to_token.items()}

#     raw_decoded = manual_ctc_decode(predicted_ids.tolist(), blank_id, inv_vocab)
    
#     # Get the clean version for MedGemma
#     transcript = process_for_medgemma(raw_decoded)
    
#     print("\n" + "="*40)
#     print("CLEAN TRANSCRIPT FOR MEDGEMMA")
#     print("="*40)
#     print(transcript)
#     return transcript

# if __name__ == "__main__":
#     clean_transcript = run_pipeline()
    
#     # Example Kaggle Prompt format
#     # prompt = f"Context: {clean_transcript}\n\nTask: Generate a SOAP note based on the clinical findings above."
from src.pipeline import MedScribePipeline, _clean_transcript
# Quick test of just the clean function on your known-good output:
raw = "EXAM TYPE CT chest PE protocol {period} INDICATION 54-year-old female {comma} shortness of breath"
print(_clean_transcript(raw))