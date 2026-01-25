"""
End-to-end pipeline test
Audio → Whisper → MedGemma → SOAP Note
"""

import torch
import time
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import librosa
import os

print("="*60)
print("MedScribe End-to-End Pipeline Test")
print("="*60)

# Load Whisper
print("\n[1/3] Loading Whisper...")
asr_start = time.time()

asr = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-small",
    device=0 if torch.cuda.is_available() else -1,
    chunk_length_s=30,
    stride_length_s=5
)

asr_load = time.time() - asr_start

# Load MedGemma
print("[2/3] Loading MedGemma 4B...")
llm_start = time.time()

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

llm = AutoModelForCausalLM.from_pretrained(
    "./models/medgemma",
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained("./models/medgemma", trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

llm_load = time.time() - llm_start

print(f"\nModels loaded:")
print(f"  Whisper: {asr_load:.2f}s")
print(f"  MedGemma: {llm_load:.2f}s")
print(f"  Total: {asr_load + llm_load:.2f}s")

# Test audio
audio_path = "data/audio_samples/test_case_01_cardiology.wav"

print(f"\n[3/3] Processing: {os.path.basename(audio_path)}")

# Stage 1: ASR
print("\n  Stage 1: Whisper transcription...")
audio, sr = librosa.load(audio_path, sr=16000, mono=True)

stage1_start = time.time()
transcript_result = asr(
    {"array": audio, "sampling_rate": sr},
    return_timestamps=False
)
transcript = transcript_result["text"]
stage1_time = time.time() - stage1_start

print(f"    Latency: {stage1_time:.2f}s")
print(f"    Transcript: {transcript[:100]}...")

# Stage 2: SOAP generation
print("\n  Stage 2: MedGemma SOAP generation...")

prompt = f"""You are a clinical documentation assistant. Convert this transcript to a SOAP note.

TRANSCRIPT:
{transcript}

SOAP NOTE:"""

stage2_start = time.time()
inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(llm.device)

with torch.no_grad():
    outputs = llm.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id
    )

full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
soap_note = full_output.split("SOAP NOTE:")[-1].strip() if "SOAP NOTE:" in full_output else full_output
stage2_time = time.time() - stage2_start

print(f"    Latency: {stage2_time:.2f}s")
print(f"    SOAP: {soap_note[:100]}...")

# Summary
total_time = stage1_time + stage2_time

print("\n" + "="*60)
print("Pipeline Results")
print("="*60)
print(f"Total latency: {total_time:.2f}s")
print(f"  ASR: {stage1_time:.2f}s ({stage1_time/total_time*100:.1f}%)")
print(f"  LLM: {stage2_time:.2f}s ({stage2_time/total_time*100:.1f}%)")

# Save
os.makedirs("results", exist_ok=True)

with open("results/pipeline_baseline_results.txt", "w", encoding="utf-8") as f:
    f.write("="*60 + "\n")
    f.write("End-to-End Pipeline Test\n")
    f.write("="*60 + "\n\n")
    f.write(f"Audio: {audio_path}\n")
    f.write(f"Total: {total_time:.2f}s\n")
    f.write(f"  ASR: {stage1_time:.2f}s\n")
    f.write(f"  LLM: {stage2_time:.2f}s\n\n")
    f.write("="*60 + "\n")
    f.write("TRANSCRIPT:\n")
    f.write("="*60 + "\n")
    f.write(transcript + "\n\n")
    f.write("="*60 + "\n")
    f.write("SOAP NOTE:\n")
    f.write("="*60 + "\n")
    f.write(soap_note + "\n")

print(f"\nSaved: results/pipeline_baseline_results.txt")

if torch.cuda.is_available():
    vram = torch.cuda.max_memory_allocated() / 1e9
    print(f"VRAM peak: {vram:.2f} GB")

print(f"\nPipeline complete")
print(f"Target: <3s, Current: {total_time:.2f}s")