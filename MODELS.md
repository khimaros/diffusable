# MODELS

notes for working with stable-diffusion-v1-5 based models

## KNOWN WORKING

some models work better with `-p 1` depending on system RAM available.

 - runwayml/stable-diffusion-v1-5
 - prompthero/funko-diffusion
 - prompthero/openjourney

## BARE CHECKPOINTS

if the only available file is a `.ckpt`, you will need to prepare the pipeline configs manually.

the simplest way to do this is to copy them from the runwayml repo:

```shell
cd ~/src/huggingface.co/

mkdir -p $USER/testmodel/

cp <the ckpt files> $USER/model/

cp -r runwayml/stable-diffusion-v1-5/{feature_extractor,safety_checker,scheduler,text_encoder,tokenizer,unet,vae,model_index.json} $USER/testmodel/

picklescan -p $USER/model/
```
