#!/usr/bin/env python

import argparse
import os
import os.path
import toml

parser = argparse.ArgumentParser(description='text to image with diffuser toolkit')
parser.add_argument(
        '-t', '--toml', metavar=('PATH', 'SECTION'), type=str, nargs=2,
        help='toml file and section to load task parameters')
parser.add_argument(
        '-n', '--name', type=str,
        help='base file name for generated images')
parser.add_argument(
        '-m', '--model', type=str, default='prompthero/openjourney',
        help='diffusion model to use for inference (default: "prompthero/openjourney")')
parser.add_argument(
        '-c', '--num_outputs', type=int, default=4,
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
        '-x', '--negative_prompts', action='append',
        help='prompts to negate from the generated image')
parser.add_argument(
        'prompts', metavar='PROMPT', nargs='*',
        help='prompt to generate images from')
FLAGS = parser.parse_args()

CONFIG = {}

if FLAGS.toml:
    toml_path, toml_section = FLAGS.toml
    if os.path.exists(toml_path):
        toml_data = toml.load(toml_path)
        CONFIG = toml_data['DEFAULT']
        CONFIG.update(toml_data[toml_section])
        CONFIG['name'] = toml_section


def run():
    # calculate which flags were set explicitly and override config options
    sentinel = object()
    sentinel_ns = argparse.Namespace(**{ key: sentinel for key in vars(FLAGS) })
    parser.parse_args(namespace=sentinel_ns)
    for key, value in vars(sentinel_ns).items():
        if key == 'prompts':
            continue
        if value is not sentinel:
            CONFIG[key] = value
        elif key not in CONFIG:
            CONFIG[key] = getattr(FLAGS, key)

    #print('[*] using configuration:', CONFIG)

    if not CONFIG.get('prompts'):
        print('[!] prompt must be defined in config or on command line')
        return

    if not CONFIG.get('seed'):
        CONFIG['seed'] = int.from_bytes(os.urandom(2), 'big')

    print('[*] using generator seed:', CONFIG['seed'])

    local_files_only = not CONFIG.get('download_models')
    model_path = CONFIG['model']
    if not CONFIG.get('download_models'):
        model_path = os.path.expanduser(os.path.join(CONFIG['models_dir'], CONFIG['model']))

    print('[*] preparing diffusion pipeline from', model_path)

    import torch
    from torch import autocast
    from diffusers.models import AutoencoderKL
    from diffusers import StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained(model_path, local_files_only=local_files_only)

    def dummy(images, **kwargs):
        return images, False

    pipe.safety_checker = dummy

    print('[*] executing diffusion pipeline with prompt:', CONFIG['prompts'])

    if CONFIG.get('negative_prompts'):
        print('[*] executing with negative prompts:', CONFIG['negative_prompts'])
    #print('[*] will generate', FLAGS.num_outputs, 'images per prompt')

    generator = torch.Generator().manual_seed(CONFIG['seed'])

    # Reference pipeline parameters here:
    # https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L467
    output = pipe(
            prompt=CONFIG['prompts'],
            negative_prompt=CONFIG['negative_prompts'],
            num_images_per_prompt=CONFIG['num_outputs'],
            num_inference_steps=CONFIG['num_inference_steps'],
            width=CONFIG['width'],
            height=CONFIG['height'],
            generator=generator,
            guidance_scale=CONFIG['guidance_scale'],
    )

    os.makedirs(FLAGS.output_dir, exist_ok=True)

    for i, image in enumerate(output.images):
        output_file = '%s.%d.png' % (CONFIG['name'], i)
        output_path = os.path.expanduser(os.path.join(CONFIG['output_dir'], output_file))
        print('[*] writing image', i, 'to', output_path)
        image.save(output_path)


if __name__ == '__main__':
    run()
