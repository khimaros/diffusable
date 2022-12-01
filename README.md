# DIFFUSION

## usage

clone this repository and install python requirements:

```shell
$ git clone <repo>
$ pip install -r requirements.txt
```

install git-lfs for large file storage support.

clone huggingface repositories into `~/src/huggingface.co/`:

```shell
$ git clone --recurse-submodules https://huggingface.co/prompthero/openjourney ~/src/huggingface.co/prompthero/openjourney/
```

execute the diffusion model with some prompts:

```shell
$ python ./diffusion.py --name='female-elf-portrait' 'mdjrny-v4 style portrait of female elf, intricate, elegant, highly detailed, digital painting, artstation, concept art, smooth, sharp focus, illustration, art by artgerm and greg rutkowski and alphonse mucha, 8k'
```

the images will be written to `./output/female-elf-portrait.{0,1,2,3}.png`

for more detailed usage, see help:

```shell
$ python ./diffusion.py --help
```
