"""
wget https://huggingface.co/Yidiii/UniCodec_ckpt/resolve/main/unicode.ckpt
wget https://raw.githubusercontent.com/mesolitica/UniCodec-fix/refs/heads/main/configs/unicodec_frame75_10s_nq1_code16384_dim512_finetune.yaml
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
    
    from encodec.utils import convert_audio
    from unicodec.decoder.pretrained import Unicodec
    import torchaudio
    import torch

    torch.set_grad_enabled(False)
    
    config = 'unicodec_frame75_10s_nq1_code16384_dim512_finetune.yaml'
    model = Unicodec.from_pretrained0802(config, 'unicode.ckpt')
    _ = model.cuda()
    bandwidth_id = torch.tensor([0]).cuda()

    for f in tqdm(files):

        wav, sr = torchaudio.load(f)
        wav = convert_audio(wav, sr, 24000, 1).cuda()

        _, discrete_code = model.encode_infer(wav, '2', bandwidth_id=bandwidth_id)
        features = model.codes_to_features(discrete_code)

        y_ = model.decode(features, bandwidth_id=bandwidth_id).cpu().numpy()
        
        new_f = os.path.join(folder, os.path.split(f)[1]).replace('.mp3', '.wav')
        sf.write(new_f, y_[0], 24000)

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
