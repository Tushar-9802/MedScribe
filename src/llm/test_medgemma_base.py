"""
Test MedGemma 4B baseline inference (unfinetuned)
Input: Whisper transcripts from Test 1
Output: Generated SOAP notes
Measures: Latency, VRAM, quality
"""

import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os

print("="*60)
print("MedGemma 4B Baseline Test (Unfinetuned)")
print("="*60)

# [1/4] Configure 4-bit quantization
print("\n[1/4] Configuring 4-bit quantization...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

# [2/4] Load MedGemma
print("[2/4] Loading MedGemma 4B (this takes ~30s)...")
start = time.time()

model = AutoModelForCausalLM.from_pretrained(
    "./models/medgemma",
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(
    "./models/medgemma",
    trust_remote_code=True
)

# Set pad token if not present
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

load_time = time.time() - start
print(f"✓ Model loaded in {load_time:.2f}s")

# [3/4] Load Whisper transcripts
print("\n[3/4] Loading Whisper transcripts...")
transcript_file = "results/whisper_baseline_transcripts.txt"

if not os.path.exists(transcript_file):
    print(f"✗ File not found: {transcript_file}")
    print("Run src/asr/test_baseline.py first!")
    exit(1)

# Parse transcripts (they're separated by === lines)
with open(transcript_file, "r", encoding="utf-8") as f:
    content = f.read()

# Extract transcripts
transcripts = []
blocks = content.split("="*60)

for block in blocks:
    lines = block.strip().split("\n")
    # Skip header blocks
    if "File:" in block or "Latency:" in block:
        continue
    # Get the transcript text
    text = block.strip()
    if len(text) > 50:  # Valid transcript
        transcripts.append(text)

print(f"✓ Found {len(transcripts)} transcripts")

# Prompt template
prompt_template = """You are a clinical documentation assistant. Convert the following medical transcript into a structured SOAP note.

TRANSCRIPT:
{transcript}

Generate a SOAP note with these sections:
- SUBJECTIVE: Patient-reported symptoms and history
- OBJECTIVE: Physical exam findings and vital signs
- ASSESSMENT: Clinical impressions and diagnoses
- PLAN: Diagnostic tests, treatments, and follow-up

SOAP NOTE:"""

# [4/4] Generate SOAP notes
print("\n[4/4] Generating SOAP notes...")
results = []

for i, transcript in enumerate(transcripts[:3], 1):
    print(f"\n{'='*60}")
    print(f"Case {i}/{min(3, len(transcripts))}")
    print(f"{'='*60}")
    
    # Create prompt
    prompt = prompt_template.format(transcript=transcript.strip())
    
    print(f"Input length: {len(transcript)} chars")
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    
    # Generate
    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    generation_time = time.time() - start
    
    # Decode
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract SOAP note (after "SOAP NOTE:")
    if "SOAP NOTE:" in full_output:
        soap_note = full_output.split("SOAP NOTE:")[-1].strip()
    else:
        soap_note = full_output.strip()
    
    # Display results
    print(f"Generation time: {generation_time:.2f}s")
    print(f"Output length: {len(soap_note)} chars")
    print(f"\nGenerated SOAP note preview:")
    print("-"*60)
    print(soap_note[:300] + "..." if len(soap_note) > 300 else soap_note)
    print("-"*60)
    
    # Save result
    results.append({
        "case": i,
        "transcript": transcript,
        "soap_note": soap_note,
        "latency": generation_time
    })

# Save all outputs
print("\n[5/5] Saving results...")
os.makedirs("results", exist_ok=True)

with open("results/medgemma_baseline_outputs.txt", "w", encoding="utf-8") as f:
    f.write("="*60 + "\n")
    f.write("MedGemma 4B Baseline Results (Unfinetuned)\n")
    f.write("="*60 + "\n\n")
    
    for r in results:
        f.write(f"{'='*60}\n")
        f.write(f"CASE {r['case']}\n")
        f.write(f"Generation time: {r['latency']:.2f}s\n")
        f.write(f"{'='*60}\n\n")
        
        f.write(f"INPUT TRANSCRIPT:\n")
        f.write(f"{'-'*60}\n")
        f.write(f"{r['transcript']}\n\n")
        
        f.write(f"GENERATED SOAP NOTE:\n")
        f.write(f"{'-'*60}\n")
        f.write(f"{r['soap_note']}\n\n\n")

# Summary
print("\n" + "="*60)
print("MedGemma Baseline Summary")
print("="*60)
avg_latency = sum(r['latency'] for r in results) / len(results)
print(f"Cases processed: {len(results)}")
print(f"Average generation time: {avg_latency:.2f}s")
print(f"\nOutputs saved to: results/medgemma_baseline_outputs.txt")
print("="*60)

# VRAM check
if torch.cuda.is_available():
    vram_used = torch.cuda.max_memory_allocated() / 1e9
    vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"\nVRAM usage:")
    print(f"  Peak: {vram_used:.2f} GB")
    print(f"  Total: {vram_total:.2f} GB")
    print(f"  Used: {vram_used/vram_total*100:.1f}%")

print("\n" + "="*60)
print("MedGemma baseline test complete")
print("="*60)
print("\nNOTE: This is the UNFINETUNED model.")
print("Output quality will improve significantly after fine-tuning.")
print("\nExpected issues with unfinetuned model:")
print("  - May not follow SOAP format perfectly")
print("  - May miss sections")
print("  - May have inconsistent structure")
print("\nThis is NORMAL and expected. Training will fix these issues.")