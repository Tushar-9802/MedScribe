# MedScribe Baseline Results (Unfinetuned)

**Date:** January 26, 2026
**Hardware:** RTX 5070 Ti (16GB VRAM)
**Software:** PyTorch Nightly, Transformers 5.0

---

## System Performance Summary

### Whisper ASR
- Model: openai/whisper-small (approximately 1GB)
- Average latency: approximately 2.6s
- VRAM usage: approximately 2.3GB
- Quality: Excellent (medical terms preserved)

### MedGemma 4B (Unfinetuned)
- Model: google/medgemma-4b-it (4-bit quantized)
- Average latency: approximately 2.4s
- VRAM usage: approximately 7.8GB
- Quality: Requires fine-tuning

### End-to-End Pipeline
- Total latency: approximately 5.0s
- Breakdown: ASR (50%) + LLM (50%)
- VRAM peak: approximately 9.0GB

---

## Detailed Results

### Whisper Transcripts

See: `results/whisper_baseline_transcripts.txt`

### MedGemma Outputs

See: `results/medgemma_baseline_outputs.txt`

### Pipeline Flow

See: `results/pipeline_baseline_results.txt`

---

## Key Findings

### What Works Well
- Whisper: Accurate medical terminology
- MedGemma: Fast inference, fits in VRAM
- Pipeline: Stable, no crashes

### What Needs Improvement
- SOAP structure: Inconsistent or missing sections
- Completeness: Missing key details
- Format: Does not follow templates

---

## Next Steps

1. **Days 2-3:** Prepare MTSamples training data
2. **Days 4-5:** Prompt engineering, training setup
3. **Day 6:** First LoRA training run (overnight)
4. **Day 7:** Evaluate trained model, build UI

**Target improvements after training:**
- ROUGE-L: 0.50 (baseline) to 0.88+ (trained)
- Structure completeness: 40% to 95%+
- Physician rating: 2/5 to 4+/5
