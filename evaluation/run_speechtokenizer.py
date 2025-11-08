"""
codebook size: 1024
nq: 8
tps: 50
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
    
    import torchaudio
    import torch
    import librosa
    from speechtokenizer import SpeechTokenizer
    from huggingface_hub import hf_hub_download

    torch.set_grad_enabled(False)

    config_path = hf_hub_download(repo_id="OpenMOSS-Team/AnyGPT-speech-modules", filename='speechtokenizer/config.json')
    ckpt_path = hf_hub_download(repo_id="OpenMOSS-Team/AnyGPT-speech-modules", filename='speechtokenizer/ckpt.dev')

    model = SpeechTokenizer.load_from_checkpoint(config_path, ckpt_path)
    model.eval().cuda()

    for f in tqdm(files):
        y, _ = librosa.load(f, sr = model.sample_rate)
        wav = torch.tensor(y)[None, None]

        with torch.no_grad():
            codes = model.encode(wav.cuda())
        
        RVQ_1 = codes[:1, :, :]
        RVQ_supplement = codes[1:, :, :]
        wav = model.decode(torch.cat([RVQ_1, RVQ_supplement], axis=0))
        new_f = os.path.join(folder, os.path.split(f)[1]).replace('.mp3', '.wav')
        sf.write(new_f, wav[0, 0].cpu().numpy(), model.sample_rate)

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
