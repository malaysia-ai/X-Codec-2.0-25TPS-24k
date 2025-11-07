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
    
    import torchaudio
    import torch
    from distilcodec import DistilCodec, demo_for_generate_audio_codes
    from huggingface_hub import hf_hub_download

    torch.set_grad_enabled(False)

    codec_model_config_path = hf_hub_download(repo_id="IDEA-Emdoor/DistilCodec-v1.0", filename='model_config.json')
    codec_ckpt_path = hf_hub_download(repo_id="IDEA-Emdoor/DistilCodec-v1.0", filename='g_00204000')

    codec = DistilCodec.from_pretrained(
        config_path=codec_model_config_path,
        model_path=codec_ckpt_path,
        use_generator=True,
        is_debug=False).eval()

    for f in tqdm(files):

        audio_tokens = demo_for_generate_audio_codes(
            codec, 
            f,
            target_sr=24000, 
            plus_llm_offset=False
        )
        y_gen = codec.decode_from_codes(
            audio_tokens, 
            minus_token_offset=False
        )
        new_f = os.path.join(folder, os.path.split(f)[1]).replace('.mp3', '.wav')
        sf.write(new_f, y_gen[0, 0].cpu().numpy(), 24000)

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
