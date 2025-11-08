"""
codebook size: 2048
nq: 32
tps: 12.5
"""

import os
from tqdm import tqdm
from functools import partial
from multiprocess import Pool
from glob import glob
import librosa
import soundfile as sf
import click

def chunks(l, devices):
    chunk_size = len(l) // len(devices)
    remainder = len(l) % len(devices)
    start = 0
    for i in range(len(devices)):
        extra = 1 if i < remainder else 0
        end = start + chunk_size + extra
        yield (l[start:end], devices[i])
        start = end

def loop(
    files_device,
    folder,
):
    files, device = files_device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
    
    import torch
    import librosa
    from transformers import MimiModel, AutoFeatureExtractor

    torch.set_grad_enabled(False)

    model = MimiModel.from_pretrained("kyutai/mimi").eval().cuda()
    feature_extractor = AutoFeatureExtractor.from_pretrained("kyutai/mimi")

    for f in tqdm(files):
        y, _ = librosa.load(f, sr = feature_extractor.sampling_rate)
        inputs = feature_extractor(raw_audio=y, sampling_rate=feature_extractor.sampling_rate, return_tensors="pt").to('cuda')
        outputs = model(inputs["input_values"], inputs["padding_mask"])
        audio_values = outputs.audio_values
        new_f = os.path.join(folder, os.path.split(f)[1]).replace('.mp3', '.wav')
        sf.write(new_f, audio_values[0, 0].cpu().numpy(), feature_extractor.sampling_rate)

@click.command()
@click.option('--folder')
@click.option('--replication', default = 1)
def main(folder, replication):
    devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    if devices is None:
        
        import torch
        devices = list(range(torch.cuda.device_count()))
    else:
        devices = [d.strip() for d in devices.split(',')]
    
    devices = replication * devices
    os.makedirs(folder, exist_ok = True)
    files = glob('test-set/*/*/*/*.mp3', recursive = True)
    splitted = list(chunks(files, devices))
    loop_partial = partial(loop,folder=folder)

    with Pool(len(devices)) as pool:
        pooled = pool.map(loop_partial, splitted)

if __name__ == '__main__':
    main()
