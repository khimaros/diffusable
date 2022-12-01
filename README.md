# DIFFUSABLE

 command line utility for working with huggingface diffuser pipelines 

## USAGE

clone this repository.

install python requirements:

```shell
pip install -r requirements.txt
```

install git-lfs for large file storage support.

clone huggingface repositories into `~/src/huggingface.co/`:

```shell
git clone --recurse-submodules \
    https://huggingface.co/prompthero/openjourney/ \
    ~/src/huggingface.co/prompthero/openjourney/
```

generate a few images using the default model (`prompthero/openjourney`) and options:

```shell
python ./diffusable.py \
    --name='female-elf-portrait' \
    'mdjrny-v4 style portrait of female elf, intricate, elegant, highly detailed, digital painting, artstation, concept art, smooth, sharp focus, illustration, art by artgerm and greg rutkowski and alphonse mucha, 8k'
```

images will be written to `./output/female-elf-portrait.{0,1,2,3}.png`

to enable automatic fetching of a model from huggingface with a manual seed:

```shell
python ./diffusable.py \
    --download_models \
    --model='prompthero/funko-diffusion' \
    --seed=31911 \
    --num_outputs=2 \
    --name='morgan-freeman-funko' \
    'Morgan Freeman, funko style'
```

images will be written to `./output/morgan-freeman-funko.{0,1}.png`

automatic model downloads will be stored in `~/.cache/huggingface/`

to dump the configuration for inspection, append the `--dump` flag to any command.

for more detailed usage, see help:

```shell
python ./diffusable.py --help
```

## CONFIGURATION

diffusable tasks can be configured in TOML format.

by default, tasks will be read from `./diffusion.toml` if it exists.

the config file uses the same keys as the flag names above.

see [diffusable.example.toml](diffusable.example.toml) for an example.

to execute a task from the config:

```shell
$ python ./diffusable.py -c ./diffusable.example.toml -t female-elf-portrait
```

by default, the TOML section is used as the output name.

you can override config options with flags:

```shell
python ./diffusable.py \
    -c ./diffusable.example.toml \
    -t female-elf-portrait \
    -m runwayml/stable-diffusion-v1-5 \
    -n female-elf-portrait-sd
```

it is also possible to run multiple tasks in sequence:

```shell
python ./diffusable.py \
    -c ./diffusable.example.toml \
    -t female-elf-portrait \
    -t morgan-freeman-funko
```

to run all tasks from the config in sequence:

```shell
python ./diffusable.py -c ./diffusable.example.toml -a
```
