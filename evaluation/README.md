# Evaluation

## How to

1. Download the dataset,

```bash
wget https://huggingface.co/malaysia-ai/xcodec2-25TPS-24k/resolve/main/test-set/sample-common-voice-17-test-set.zip
unzip sample-common-voice-17-test-set.zip
```

2. Run evaluation such as,

```bash
python3 run_snac.py --folder 'snac'
```

3. Calculate MOS using UTMOSV2 such as,

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 calculate_utmosv2.py --folder 'xcodec2'
```