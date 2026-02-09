import huggingface_hub
path = huggingface_hub.hf_hub_download("google/medasr", "test_audio.wav")
print(path)