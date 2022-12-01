#!/usr/bin/env python

import argparse
import os
import os.path

parser = argparse.ArgumentParser(description='text to image with diffuser toolkit')
parser.add_argument(
        '-m', '--model', type=str, default='prompthero/openjourney',
        help='diffusion model to use for inference (default: prompthero/openjourney)')
parser.add_argument(
        '-n', '--name', type=str, default='result',
        help='base file name for generated images: (default: "result")')
parser.add_argument(
        '-c', '--num_outputs', type=int, default=4,
        help='number of images to generate per prompt (default: 4)')
parser.add_argument(
        '-i', '--num_inference_steps', type=int, default=50,
        help='number of denoising steps, higher increases quality (default: 50)')
parser.add_argument(
        '-s', '--seed', type=int, default=0,
        help='seed to use for generator (default: random)')
parser.add_argument(
        '-g', '--guidance_scale', type=int, default=7,
        help='how closely to link images to prompt, higher can reduce image quality (default: 7)')
parser.add_argument(
        '-r', '--models_dir', type=str, default='~/src/huggingface.co/',
        help='root directory containing huggingface models (default: ~/src/huggingface.co)')
parser.add_argument(
        '-d', '--download_models', action='store_true', default=False,
        help='allow automatic downloading of diffuser models (default: False)')
parser.add_argument(
        '-o', '--output_dir', type=str, default='./output/',
        help='directory to write image output (default: ./output/')
parser.add_argument(
        '-x', '--negative_prompts', action='append',
        help='prompts to negate from the generated image')
parser.add_argument(
        'prompts', metavar='PROMPT', nargs='+', type=str,
        help='prompt to generate images from')
FLAGS = parser.parse_args()


def run():
    import torch
    from torch import autocast
    from diffusers.models import AutoencoderKL
    from diffusers import StableDiffusionPipeline

    seed = FLAGS.seed
    if not seed:
        seed = int.from_bytes(os.urandom(2), 'big')
    print('[*] using generator seed:', seed)

    local_files_only = not FLAGS.download_models
    model_path = FLAGS.model
    if not FLAGS.download_models:
        model_path = os.path.expanduser(os.path.join(FLAGS.models_dir, FLAGS.model))

    print('[*] preparing diffusion pipeline from', model_path)

    pipe = StableDiffusionPipeline.from_pretrained(model_path, local_files_only=local_files_only)

    def dummy(images, **kwargs):
        return images, False

    pipe.safety_checker = dummy

    print('[*] executing diffusion pipeline with prompt:', FLAGS.prompts)
    #print('[*] will generate', FLAGS.num_outputs, 'images per prompt')

    generator = torch.Generator().manual_seed(seed)

    # Reference pipeline parameters here:
    # https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L467
    output = pipe(
            prompt=FLAGS.prompts,
            negative_prompt=FLAGS.negative_prompts,
            num_images_per_prompt=FLAGS.num_outputs,
            num_inference_steps=FLAGS.num_inference_steps,
            width=512,
            height=512,
            generator=generator,
            guidance_scale=FLAGS.guidance_scale,
    )

    os.makedirs(FLAGS.output_dir, exist_ok=True)

    for i, image in enumerate(output.images):
        output_file = '%s.%d.png' % (FLAGS.name, i)
        output_path = os.path.expanduser(os.path.join(FLAGS.output_dir, output_file))
        print('[*] writing image', i, 'to', output_path)
        image.save(output_path)


if __name__ == '__main__':
    run()
