# Models

Place local pretrained model files here.

Expected layout:

```text
models/
  polyBERT/                                      Pretrained polymer language model files
  tabpfn-v2.5-regressor-v2.5_default.ckpt        TabPFN checkpoint
```

The PolyTab runner uses `models/polyBERT/` by default. Override it with the `POLYBERT_MODEL_PATH` environment variable when the model is stored elsewhere.

Checkpoint-like files are tracked by Git LFS through `.gitattributes`.
