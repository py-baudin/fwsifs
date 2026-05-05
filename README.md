# fwsifs

## Installation

```shell
# clone repository
git clone git@github.com:py-baudin/fwsifs.git
cd fwsifs

# create environment
conda create -n fwsifs python=3.12 numpy scipy numexpr
conda activate fwsifs

pip install .

```


## Running

```shell

# show arguments and options
fwsifs --help

# run on subject1 (default settings)
fwsifs subject1

# run on subject2 with dual R2* and linear/quadratic decay function
fwsifs subject2 --variables b0,wchi,r2w,r2f --decay-function=12 --suffix=_pow12

```


## Outputs files

- `ffmap`: fat fraction map
- `fmap` / `wmap`: fat and water maps [a.u.]
- `b0map`: field map [rad/ms]
- `phi0`: initial phase [rad]
- `wchi`: frequency shift map [ppm]
- `r2` / `r2w` / `r2f`: total/water/fat R2' relaxation rates [1/s] or [`1/s^2] (if decay function is linear or quadratic)
- `t2w`: water T2* relaxation time [ms] 
- `resids`: normalized root mean squares deviation 
- `bic`: Bayesian information criterion
- `mask`: binary foreground mask


