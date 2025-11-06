# X-Codec-2.0-25TPS-24k: Improving X-Codec-2.0 for Multi-Lingual Speech, 25Hz Latent Rate and 24 kHz Sampling

We scale up to 24k sample rate, also add 25 TPS.

## Training

1. Use virtual environment,

```bash
python3 -m venv xcodec
./xcodec/bin/pip3 install -r requirements.txt
```

2. Prepare dataset,

```bash
./xcodec/bin/python3 get_tsv.py
```

Or you can prepare yourself the data, example, [example.txt],

```text
/path/audio1.wav
/path/audio2.wav
```

But this is how we prepare our dataset, [prepare-audio-files.ipynb](prepare-audio-files.ipynb).

3. Download checkpoint, optional,

```bash
wget https://huggingface.co/HKUSTAudio/xcodec2/resolve/main/ckpt/epoch%3D4-step%3D1400000.ckpt
```

4. Interpolate to support 24k, [interpolate.ipynb](interpolate.ipynb).

5. Run finetune,

```bash
CUDA_VISIBLE_DEVICES="0,1" \
python3 train.py log_dir=24k \
train.trainer.devices=2 \
dataset.train.filelist="train-files.txt" \
dataset.train.batch_size=20 \
dataset.val.filelist="val-files.txt"
```

## Evaluation

All scripts to evaluate SNAC, Encodec, DAC, Mimi, BigCodec, SpeechTokenizer, WavTokenizer, BiCodec, Uni-Codec, DistilCodec, Neucodec, X-Codec-2.0 baseline and X-Codec-2.0-25TPS-24k in [evaluation](evaluation).