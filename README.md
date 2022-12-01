# DIFFUSION

## USAGE

clone this repository and install python requirements:

```shell
$ git clone <repo>
$ pip install -r requirements.txt
```

install git-lfs for large file storage support.

clone huggingface repositories into `~/src/huggingface.co/`:

```shell
$ git clone --recurse-submodules \
    https://huggingface.co/prompthero/openjourney/ \
    ~/src/huggingface.co/prompthero/openjourney/
```

generate a few images using the default model (`prompthero/openjourney`) and options:

```shell
$ python ./diffusion.py \
    --name='female-elf-portrait' \
    'mdjrny-v4 style portrait of female elf, intricate, elegant, highly detailed, digital painting, artstation, concept art, smooth, sharp focus, illustration, art by artgerm and greg rutkowski and alphonse mucha, 8k'
```

images will be written to `./output/female-elf-portrait.{0,1,2,3}.png`

to enable automatic fetching of a model from huggingface with a manual seed:

```shell
$ python ./diffusion.py \
    --download_models \
    --model='prompthero/funko-diffusion' \
    --seed=31911 \
    --num_outputs=2 \
    --name='morgan-freeman-funko' \
    'Morgan Freeman, funko style'
```

images will be written to `./output/morgan-freeman-funko.{0,1}.png`

automatic model downloads will be stored in `~/.cache/huggingface/`

for more detailed usage, see help:

```shell
$ python ./diffusion.py --help
```
