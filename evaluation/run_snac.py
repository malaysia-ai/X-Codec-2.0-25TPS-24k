"""
codebook size: 4096
nq: 3
tps: 27
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
    from snac import SNAC

    torch.set_grad_enabled(False)
    snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
    snac = snac.to('cuda')

    for f in tqdm(files):
        y = librosa.load(f, sr = 24000)[0]
        audio = torch.tensor(y)[None, None].cuda()
        codes = snac.encode(audio)
        audio_hat = snac.decode(codes)
        y_ = audio_hat[0, 0].cpu().numpy()
        new_f = os.path.join(folder, os.path.split(f)[1]).replace('.mp3', '.wav')
        sf.write(new_f, y_, 24000)

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
