import torch
import utmosv2
import click
import json

@click.command()
@click.option('--folder')
@click.option('--batch-size', default=16)
@click.option('--num-workers', default=4)
def main(folder, batch_size, num_workers):

    filename = folder + '.json'
    try:
        with open(filename) as fopen:
            json.load(fopen)
        return
    except:
        pass
    
    model = utmosv2.create_model(pretrained=True)
    _ = model.eval().cuda()
    mos = model.predict(
        input_dir=folder, 
        remove_silent_section=True, 
        num_repetitions=1, 
        device='cuda',
        batch_size=batch_size,
        num_workers=num_workers,
    )
    with open(filename, 'w') as fopen:
        json.dump(mos, fopen)
    

if __name__ == '__main__':
    main()