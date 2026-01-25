"""
Test Whisper baseline (replaces MedASR)
Works with TTS audio - handles long files
"""

import torch
import time
from transformers import pipeline
import librosa
import os

print("="*60)
print("Whisper ASR Baseline Test")
print("="*60)

# Load Whisper
print("\n[1/3] Loading Whisper Small...")
start = time.time()

asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-small",
    device=0 if torch.cuda.is_available() else -1,
    chunk_length_s=30,  # ← ADDED: Handle long audio
    stride_length_s=5   # ← ADDED: Overlap for better results
)

load_time = time.time() - start
print(f"✓ Model loaded in {load_time:.2f}s")

# Test audio files
audio_files = [
    "data/audio_samples/test_case_01_cardiology.wav",
    "data/audio_samples/test_case_02_respiratory.wav",
    "data/audio_samples/test_case_03_pediatric.wav"
]

results = []

print(f"\n[2/3] Testing {len(audio_files)} audio files...")

for audio_path in audio_files:
    if not os.path.exists(audio_path):
        print(f"✗ File not found: {audio_path}")
        continue
    
    print(f"\nProcessing: {os.path.basename(audio_path)}")
    
    # Load audio
    audio_array, sample_rate = librosa.load(audio_path, sr=16000, mono=True)
    duration = len(audio_array) / sample_rate
    
    print(f"  Duration: {duration:.2f}s")
    
    # Transcribe
    start = time.time()
    result = asr_pipeline(
        {"array": audio_array, "sampling_rate": sample_rate},
        return_timestamps=False  # ← ADDED: Disable timestamps for simplicity
    )
    inference_time = time.time() - start
    
    transcript = result["text"]
    
    # Display results
    print(f"  Latency: {inference_time:.2f}s")
    print(f"  Transcript length: {len(transcript)} chars")
    print(f"  Preview: {transcript[:100]}...")
    
    # Save
    results.append({
        "file": audio_path,
        "transcript": transcript,
        "latency": inference_time
    })

# Save all transcripts
print("\n[3/3] Saving results...")
os.makedirs("results", exist_ok=True)

with open("results/whisper_baseline_transcripts.txt", "w", encoding="utf-8") as f:
    for r in results:
        f.write(f"{'='*60}\n")
        f.write(f"File: {r['file']}\n")
        f.write(f"Latency: {r['latency']:.2f}s\n")
        f.write(f"{'='*60}\n")
        f.write(f"{r['transcript']}\n\n")

# Summary
print("\n" + "="*60)
print("Whisper ASR Summary")
print("="*60)
avg_latency = sum(r['latency'] for r in results) / len(results)
print(f"Files processed: {len(results)}")
print(f"Average latency: {avg_latency:.2f}s")
print(f"\nTranscripts saved to: results/whisper_baseline_transcripts.txt")
print("="*60)

if torch.cuda.is_available():
    vram_used = torch.cuda.max_memory_allocated() / 1e9
    print(f"\nVRAM peak usage: {vram_used:.2f} GB")

print("\n✓ Whisper baseline test complete")