import pathlib
import click
import shutil
import time
import numpy as np
from scipy import ndimage

from iomed import volume, config
from . import fwsifs


@click.command('fwsifs', context_settings={'show_default': True})
@click.argument('subject')
@click.option('--dest', default='results', type=click.Path(), help='Output directory.')
@click.option('--suffix', default=None, type=click.Path(), help='Output suffix.')
@click.option('--overwrite', is_flag=True, help='Overwrite output.')
@click.option('--variables', default='b0,r2,wchi', help='Non-linear variables to estimate (among: b0, wchi, r2, r2w, r2f)')
@click.option('--fat-model', default='Bydder2011', type=click.Choice(['Bydder2011', 'Azzabou2017', 'Ren2008', 'Yu2008']), help='Lipid model.')
@click.option('--decay-function', default='11', type=click.Choice(['11', '22', '12']), help='Decay function exponents for r2w and r2f')
@click.option('--niter', default=30, help='Number of iterations.')
@click.option('--coarse', is_flag=True, default=False, help='Coarse estimation only.')
@click.option('--silent', is_flag=True, default=False, help='Hide convergence details.')
def cli(subject, dest, suffix, overwrite, variables, fat_model, decay_function, niter, coarse, silent):
    """ Run the FWSIFS algorithm for the given SUBJECT """

    # source data
    here = pathlib.Path(__file__).parent
    datadir = here.parent.parent / 'data'

    srcdir = datadir / subject
    if not srcdir.is_dir():
        click.echo(f'Unknown subject: {subject}')
        click.echo('Available subjects:')
        for dirname in datadir.glob('subject*/'):
            click.echo(f'\t{dirname.name}')
        return
    
    # output data
    dstdir = pathlib.Path(dest) / (subject + (suffix or ''))
    if dstdir.exists():
        click.echo(f'Output directory exists: {dstdir}')
        if overwrite:
            shutil.rmtree(dstdir)
        else:
            return
    dstdir.mkdir(parents=True)

    # load data
    click.echo('Loading data')
    info = config.read(srcdir / 'info.yml')
    volumes = []
    for i in range(1 ,7):
        real = volume.read(srcdir / f'volr_{i}.nii.gz')
        imag = volume.read(srcdir / f'voli_{i}.nii.gz')
        volumes.append(real + 1j * imag)

    # get background mask
    absvals = np.mean(np.abs(volumes), axis=0)
    mask = absvals > np.percentile(np.unique(absvals),5)
    mask = ndimage.binary_opening(mask, structure=np.ones((3,3,1)))
    mask = ndimage.binary_fill_holes(mask, structure=np.ones((3,3,1)))

    # run fwsifs
    click.echo('Running FWSIFS')
    echo_times = info['echo_times']
    variables = variables.split(',')
    r2pow = decay_function
    opts = dict(
        niter_refine=niter, 
        disp=not silent, 
        fat_model=fat_model, 
        variables=variables, 
        coarse=coarse,
        pixel_spacing=volumes[0].spacing,
        field_strength=info['imaging_frequency'] * 1e3 / fwsifs.GAMMA_1H ,
        r2wpow={'11': 1, '22': 2, '12': 1}[r2pow],
        r2fpow={'11': 1, '22': 2, '12': 2}[r2pow],
        mu_r2w={'11': 1e3, '22': 1e4, '12': 1e3}[r2pow],
        mu_r2f={'11': 1e3, '22': 1e4, '12': 1e4}[r2pow],
        )
    tic = time.time()
    res, info = fwsifs.pdff(echo_times, volumes, mask=mask, **opts)
    
    duration = time.time() - tic
    
    # store result
    click.echo(f'Storing results (dest={dstdir})')
    config.write(dstdir / 'info.yml', info)
    for vol in res:
        volume.write(dstdir / vol, volume.tovolume(res[vol], ref=volumes[0]))
    click.echo(f'Done (duration: {duration/60:.1f}min)')

    

