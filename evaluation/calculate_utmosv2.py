import torch
import utmosv2
import click
import json

@click.command()
@click.option('--folder')
@click.option('--batch-size', default=16)
def main(folder, batch_size):
    model = utmosv2.create_model(pretrained=True)
    _ = model.eval().cuda()
    mos = model.predict(
        input_dir=folder, 
        remove_silent_section=True, 
        num_repetitions=1, 
        device='cuda',
        batch_size=batch_size,
    )
    with open(folder + '.json', 'w') as fopen:
        json.dump(mos, fopen)

if __name__ == '__main__':
    main()