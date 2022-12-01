#!/usr/bin/env python

import argparse
import os
import os.path
import toml

parser = argparse.ArgumentParser(description='text to image with diffuser toolkit')
parser.add_argument(
        '-c', '--config', metavar='PATH', type=str, default='./diffusable.toml',
        help='toml file to load task parameters from (default: "./diffusable.toml")')
parser.add_argument(
        '-t', '--tasks', action='append', default=[],
        help='tasks from configuration file to execute')
parser.add_argument(
        '-n', '--name', type=str,
        help='base file name to use for generated images')
parser.add_argument(
        '-m', '--model', type=str, default='prompthero/openjourney',
        help='diffusion model to use for inference (default: "prompthero/openjourney")')
parser.add_argument(
        '-p', '--num_outputs', type=int, default=4,
        help='number of images to generate per prompt (default: 4)')
parser.add_argument(
        '-i', '--num_inference_steps', type=int, default=50,
        help='number of denoising steps, higher increases quality (default: 50)')
parser.add_argument(
        '-W', '--width', type=int, default=512,
        help='width of generated image (default: 512)')
parser.add_argument(
        '-H', '--height', type=int, default=512,
        help='height of generated image (default: 512)')
parser.add_argument(
        '-s', '--seed', type=int, default=0,
        help='seed to use for generator (default: random)')
parser.add_argument(
        '-g', '--guidance_scale', type=int, default=7,
        help='how closely to link images to prompt, higher can reduce image quality (default: 7)')
parser.add_argument(
        '-r', '--models_dir', type=str, default='~/src/huggingface.co/',
        help='root directory containing huggingface models (default: "~/src/huggingface.co")')
parser.add_argument(
        '-d', '--download_models', action='store_true', default=False,
        help='allow automatic downloading of diffuser models (default: False)')
parser.add_argument(
        '-o', '--output_dir', type=str, default='./output/',
        help='directory to write image output (default: "./output/")')
parser.add_argument(
        '--dump', action='store_true',
        help='dump configuration and exit')
parser.add_argument(
        '-x', '--negative_prompts', action='append',
        help='prompts to negate from the generated image')
parser.add_argument(
        'prompts', metavar='PROMPT', nargs='*',
        help='prompt to generate images from')
FLAGS = parser.parse_args()

# voodoo magic to find explicitly defined flags
FLAGS_SENTINEL = list()
FLAGS_SENTINEL_NS = argparse.Namespace(**{ key: FLAGS_SENTINEL for key in vars(FLAGS) })
parser.parse_args(namespace=FLAGS_SENTINEL_NS)
EXPLICIT_FLAGS = vars(FLAGS_SENTINEL_NS).items()

CONFIG = {'DEFAULT': {}}

if FLAGS.config:
    if os.path.exists(FLAGS.config):
        print('[*] loading configuration from', FLAGS.config)
        CONFIG = toml.load(FLAGS.config)


def task_config(task):
    config = CONFIG['DEFAULT']
    if task not in CONFIG:
        print('[!] task not found in configuration file:', task)
        return config
    config.update(CONFIG[task])
    config['name'] = task

    # calculate which flags were set explicitly and override config options
    for key, value in EXPLICIT_FLAGS:
        if key in ['config', 'tasks', 'prompts']:
            continue
        if value is not FLAGS_SENTINEL:
            config[key] = value
        elif key not in config:
            config[key] = getattr(FLAGS, key)

    return config


def task_config_from_flags():
    config = CONFIG['DEFAULT']
    for key, value in vars(FLAGS).items():
        if key in ['config', 'tasks']:
            continue
        config[key] = value
    return config


def invoke_task(config):
    if not config.get('prompts'):
        print('[!] prompt must be defined in config or on command line, not running pipeline')
        return

    if not config.get('name'):
        print('[!] --name must be specified in config or on command line, not running pipeline')
        return

    if not config.get('seed'):
        config['seed'] = int.from_bytes(os.urandom(2), 'big')

    if FLAGS.dump:
        print(config)
        return

    print('[*] using generator seed:', config['seed'])

    local_files_only = not config.get('download_models')
    model_path = config['model']
    if not config.get('download_models'):
        model_path = os.path.expanduser(os.path.join(config['models_dir'], config['model']))

    print('[*] preparing diffusion pipeline from', model_path)

    import torch
    from torch import autocast
    from diffusers.models import AutoencoderKL
    from diffusers import StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained(model_path, local_files_only=local_files_only)

    def dummy(images, **kwargs):
        return images, False

    pipe.safety_checker = dummy

    print('[*] executing diffusion pipeline with prompt:', config['prompts'])
    print('[*] images will be written to', config['output_dir'], config['name'])

    if config.get('negative_prompts'):
        print('[*] executing with negative prompts:', config['negative_prompts'])
    #print('[*] will generate', FLAGS.num_outputs, 'images per prompt')

    generator = torch.Generator().manual_seed(config['seed'])

    # Reference pipeline parameters here:
    # https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L467
    output = pipe(
            prompt=config['prompts'],
            negative_prompt=config['negative_prompts'],
            num_images_per_prompt=config['num_outputs'],
            num_inference_steps=config['num_inference_steps'],
            width=config['width'],
            height=config['height'],
            generator=generator,
            guidance_scale=config['guidance_scale'],
    )
    os.makedirs(FLAGS.output_dir, exist_ok=True)

    for i, image in enumerate(output.images):
        output_file = '%s.%d.png' % (config['name'], i)
        output_path = os.path.expanduser(os.path.join(config['output_dir'], output_file))
        print('[*] writing image', i, 'to', output_path)
        image.save(output_path)


def run():
    if not FLAGS.prompts and not FLAGS.tasks:
        print('[!] at least one prompt or one config/task must be provided')
        return

    if FLAGS.prompts and FLAGS.tasks:
        print('[!] must provide EITHER prompt arguments OR config/tasks')
        return

    if len(FLAGS.tasks) > 1 and FLAGS.name:
        print('[!] flag --name cannot be used with multiple --tasks from config')
        return

    for task in FLAGS.tasks:
        print('[*] loaded task from configuration file:', task)
        config = task_config(task)
        invoke_task(config)

    if FLAGS.prompts:
        print('[*] loaded task from command line prompts')
        config = task_config_from_flags()
        invoke_task(config)


if __name__ == '__main__':
    run()