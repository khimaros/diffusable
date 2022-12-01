# DIFFUSABLE

command line utility for working with huggingface diffuser pipelines

## FEATURES

 - works with or without a GPU for [almost any](MODELS.md) Stable Diffusion derived model
 - by default runs completely offline, on your local machine
 - simple TOML configuration file for storing parameters, prompts, and common trigger words
 - can be configured to auto-download models from HuggingFace
 - writes all parameters of each image to a text file for future reference
 - simple and fun command line interface

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
    https://huggingface.co/runwayml/stable-diffusion-v1-5/ \
    ~/src/huggingface.co/runwayml/stable-diffusion-v1-5/
```

generate a few images using the default model (`runwayml/stable-diffusion-v1-5`):

```shell
python ./diffusable.py \
    --name='astronaut-rides-horse' \
    --num_outputs=2 \
    'a photo of an astronaut riding a horse on mars'
```

images will be written to `./output/astronaut-rides-horse.0.png`

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

to disable any per-model trigger prompts use `--disable_triggers`
