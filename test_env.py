#!/usr/bin/env python3
"""Test .env file loading"""

import os
from dotenv import load_dotenv

load_dotenv()

print("="*60)
print("Environment Variable Check")
print("="*60)

# Critical variables
critical = {
    "HF_TOKEN": os.getenv("HF_TOKEN"),
    "DEVICE": os.getenv("DEVICE"),
    "MEDGEMMA_MODEL": os.getenv("MEDGEMMA_MODEL"),
    "MEDASR_MODEL": os.getenv("MEDASR_MODEL"),
}

all_good = True
for key, value in critical.items():
    if value and "your_" not in value.lower():
        print(f"✓ {key}: {value[:20]}..." if len(value) > 20 else f"✓ {key}: {value}")
    else:
        print(f"✗ {key}: NOT SET or still placeholder")
        all_good = False

print("\nOptional variables:")
print(f"  WANDB_DISABLED: {os.getenv('WANDB_DISABLED')}")
print(f"  DEBUG: {os.getenv('DEBUG')}")

print("\n" + "="*60)
if all_good:
    print("✓✓✓ Environment configured correctly")
else:
    print("✗ Fix missing variables above")
    print("\nMost common issue: HF_TOKEN not set")
    print("Get token from: https://huggingface.co/settings/tokens")