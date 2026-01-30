
# MedScribe - Voice-to-SOAP Clinical Documentation

**Google MedGemma Healthcare Application Challenge**
**Submission Deadline:** February 24, 2026

## Project Status

**Current Phase:** Data Preparation (Day 2 complete)
**Next Milestone:** Batch SOAP generation (51 hours over 4 nights)

## Overview

MedScribe transforms physician voice recordings into structured clinical SOAP notes in under 3 seconds using:

- **Whisper ASR**: Speech recognition (medical-capable)
- **MedGemma 4B**: Medical language model with LoRA fine-tuning
- **Target**: Save physicians 2+ hours daily on documentation

## Problem Statement

Physicians spend 40% of patient encounters on documentation. MedScribe combines accurate speech recognition with AI-powered SOAP note generation to reduce documentation burden while maintaining clinical quality.

## Architecture

```
[Audio Input] 
    ↓
[Whisper Small - Speech Recognition]
    ↓
[MedGemma 4B + LoRA - SOAP Generation]
    ↓
[Structured Clinical Note]
```

## Current Results (Day 1 Baseline)

**Performance:**

- ASR latency: 2.6s average (Whisper Small)
- LLM latency: 40s (unfinetuned, to be optimized)
- Total: 43s (target: <3s after optimization)
- VRAM: 8.27GB peak (RTX 5070 Ti)

**Quality:**

- Whisper: Excellent medical terminology preservation
- MedGemma (unfinetuned): Complete SOAP structure, clinically appropriate
- Expected post-training: Consistent formatting, improved conciseness

## Dataset Preparation

**Source:** MTSamples (4,999 medical transcriptions)

**Processing pipeline:**

- Cleaned: 4,795 samples (96% retention)
- Structured SOAP: 983 samples (20%)
- Unstructured: 4,621 samples (require generation)

**Final dataset (after generation):**

- Training: 4,497 samples
- Validation: 562 samples
- Test: 562 samples
- Total: 5,621 samples

**Generation status:** Ready to process (51 hours over 4 nights)

## Hardware Requirements

**Development System:**

- GPU: NVIDIA RTX 5070 Ti (16GB VRAM)
- OS: Windows 11
- PyTorch: Nightly build (CUDA 12.8)

**Minimum (Inference):**

- 8GB GPU VRAM
- 16GB RAM

**Recommended (Training):**

- 12GB+ GPU VRAM
- 16GB+ RAM

## Quick Start

### Prerequisites

- Python 3.10+
- CUDA 12.1+ compatible GPU
- HuggingFace account with model access

### Installation

```bash
# Clone repository
git clone https://github.com/Tushar-9802/MedScribe-1.git
cd MedScribe-1

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Download models
python scripts/download_base_models.py
```

### Run Baseline Tests

```bash
# Test Whisper ASR
python src/asr/test_baseline.py

# Test MedGemma (unfinetuned)
python src/llm/test_baseline.py

# Test full pipeline
python test_pipeline.py
```

## Project Structure

```
MedScribe-1/
├── src/
│   ├── asr/
│   │   └── test_baseline.py        # Whisper ASR pipeline
│   ├── llm/
│   │   └── test_baseline.py        # MedGemma inference
│   ├── training/
│   │   └── train.py                # LoRA fine-tuning
│   └── evaluation/
│       └── evaluate.py             # Metrics calculation
├── scripts/
│   ├── clean_mtsamples.py          # Data cleaning
│   ├── split_structured_samples.py # Train/val/test split
│   ├── prepare_unstructured_samples.py  # Batch prep
│   └── generate_soap_batch.py      # Overnight processor
├── configs/
│   └── training.yaml               # Training configuration
├── data/
│   ├── raw/
│   │   └── mtsamples.csv           # Original dataset
│   ├── processed/
│   │   ├── structured/             # SOAP-formatted samples
│   │   └── generated_soap/         # Generated outputs
│   └── audio_samples/              # Test audio files
├── models/
│   ├── medgemma/                   # Base model (4B)
│   └── checkpoints/                # Training outputs
├── results/
│   ├── whisper_baseline_transcripts.txt
│   ├── medgemma_baseline_outputs.txt
│   └── pipeline_baseline_results.txt
└── docs/
    └── baseline_results.md         # Day 1 documentation
```

## Development Timeline

**Week 1 (Jan 20-27):**

- Day 1: Baseline testing complete ✓
- Day 2: Data preparation complete ✓
- Days 3-4: Batch SOAP generation (51 hours)
- Day 5: Dataset finalization and prompt engineering
- Day 6: Training dry run
- Day 7: First training run

**Week 2 (Jan 28-Feb 3):**

- Performance optimization (LLM speed)
- Hyperparameter tuning
- Model selection

**Week 3 (Feb 4-10):**

- UI development
- Video production
- Technical writeup

**Week 4 (Feb 11-17):**

- Testing and refinement
- Documentation
- Deployment preparation

**Week 5 (Feb 18-24):**

- Final testing
- Submission preparation
- Buffer time

## Training Configuration

**Model:** MedGemma 4B (4-bit quantized)

**LoRA parameters:**

- Rank: 16
- Alpha: 16
- Dropout: 0.05
- Target modules: All attention layers

**Training parameters:**

- Batch size: 8 (per device)
- Gradient accumulation: 4 (effective batch: 32)
- Learning rate: 2e-4
- Epochs: 3
- Precision: BFloat16

**Expected training time:** 4-5 hours (300 steps)

**Checkpointing:** Every 250 steps with resume capability

## Key Decisions

**ASR Model Choice:**

- Chosen: Whisper Small (openai/whisper-small)
- Alternative considered: MedASR (domain mismatch with TTS audio)
- Rationale: Whisper handles synthetic audio, preserves medical terms

**Data Strategy:**

- Hybrid approach: 983 structured + 4,621 generated samples
- Quality: Human-written validation set ensures accuracy
- Scale: 5,621 total samples sufficient for LoRA fine-tuning

**Quantization:**

- 4-bit NF4 with double quantization
- Model: 16GB FP32 → 2GB 4-bit (8x compression)
- Quality: ~98% of full precision performance

## Current Bottleneck

**LLM Generation Speed:**

- Current: 40s per inference
- Target: <3s per inference
- Optimization scheduled: Week 2
- Expected improvements: torch.compile, ONNX conversion, parameter tuning

## Next Steps

1. Run batch SOAP generation (Friday-Monday nights)
2. Quality check generated samples
3. Combine structured + generated datasets
4. Format for training (conversational JSONL)
5. Prompt engineering and dry run
6. First training run (Tuesday night)

## Performance Targets

| Metric        | Current | Target | Week 2 Goal |
| ------------- | ------- | ------ | ----------- |
| Total Latency | 43s     | <3s    | <5s         |
| ASR Latency   | 2.6s    | <1s    | <1.5s       |
| LLM Latency   | 40s     | <2s    | <3.5s       |
| VRAM Usage    | 8.3GB   | <12GB  | <10GB       |
| ROUGE-L       | 0.45*   | >0.85  | >0.80       |

*Unfinetuned baseline estimate

## Troubleshooting

**Batch generation interrupted:**

```bash
# Script automatically resumes from last checkpoint
python scripts/generate_soap_batch.py
```

**Out of Memory:**

```bash
# Training: Reduce batch size in configs/training.yaml
per_device_train_batch_size: 4  # Was 8

# Inference: Use smaller max_tokens
max_new_tokens: 256  # Was 512
```

**Slow generation:**

- Expected: 40s per sample currently
- Will be optimized in Week 2
- Use overnight processing for batch generation

## Repository Status

- Baseline testing: Complete
- Data preparation: Complete
- Training scripts: Ready
- Batch generation: Ready to run
- Model optimization: Scheduled Week 2

## License

MIT License (code)
Model weights subject to Google terms

## Contact

- GitHub: [@Tushar-9802](https://github.com/Tushar-9802)
- Repository: https://github.com/Tushar-9802/MedScribe-1
