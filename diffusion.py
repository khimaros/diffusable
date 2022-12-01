#!/usr/bin/env python

import argparse
import os.path

parser = argparse.ArgumentParser(description='text to image with diffuser toolkit')
parser.add_argument(
        '--model', type=str, default='prompthero/openjourney',
        help='diffusion model to use for inference')
parser.add_argument(
        '--name', type=str, default='result',
        help='base file name for generated images')
parser.add_argument(
        '--num_outputs', type=int, default=4,
        help='number of images to generate per prompt')
parser.add_argument(
        '--num_inference_steps', type=int, default=50,
        help='number of denoising steps, higher increases quality')
parser.add_argument(
        '--guidance_scale', type=int, default=7,
        help='how closely to link images to prompt, higher can reduce image quality')
parser.add_argument(
        '--models_dir', type=str, default='~/src/huggingface.co/',
        help='root directory containing huggingface models')
parser.add_argument(
        '--output_dir', type=str, default='./output/',
        help='directory to write image output')
parser.add_argument(
        'prompt', metavar='int', nargs='+', type=str,
        help='prompt to generate images from')
FLAGS = parser.parse_args()


def run():
    import torch
    from torch import autocast
    from diffusers.models import AutoencoderKL
    from diffusers import StableDiffusionPipeline

    model_path = os.path.expanduser(os.path.join(FLAGS.models_dir, FLAGS.model))

    print('[*] preparing diffusion pipeline from', model_path)

    pipe = StableDiffusionPipeline.from_pretrained(model_path, local_files_only=True)

    def dummy(images, **kwargs):
        return images, False

    pipe.safety_checker = dummy

    print('[*] executing diffusion pipeline with prompt:', FLAGS.prompt)
    #print('[*] will generate', FLAGS.num_outputs, 'images per prompt')

    # Reference pipeline parameters here:
    # https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L467
    output = pipe(
            prompt=FLAGS.prompt,
            num_images_per_prompt=FLAGS.num_outputs,
            num_inference_steps=FLAGS.num_inference_steps,
            width=512,
            height=512,
            guidance_scale=FLAGS.guidance_scale,
    )

    for i, image in enumerate(output.images):
        output_file = '%s.%d.png' % (FLAGS.name, i)
        output_path = os.path.expanduser(os.path.join(FLAGS.output_dir, output_file))
        print('[*] writing image', i, 'to', output_path)
        image.save(output_path)


if __name__ == '__main__':
    run()
