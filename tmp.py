import pathlib
from iomed import volume

outdir = pathlib.Path('data')
for dirname in pathlib.Path('data_').glob('subject*/'):
    dest = outdir / dirname.name
    dest.mkdir(parents=True, exist_ok=True)
    for file in dirname.glob('*.nii'):
        vol = volume.read(file)
        volume.write(dest / (file.stem + '.nii.gz'), vol)