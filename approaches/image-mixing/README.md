## Installation
```
pip install -r requirements.txt
```

## Usage
To run:  
```
python main.py </path/to/first/image> </path/to/second/image> --second_image_weight <weight> --backend cuda:0
```

This should create result.jpg in the root of the repo folder.  
Caution: default device (`--backend` argument) is `mps`, change it if you need to `cuda:0` or `cpu`

## How does it work
This is a latent diffusion pipeline. Generation of image is conditioned on embeddings from CLIP. `lambdalabs/sd-image-variations-diffusers`
implements conditioning on image embeddings instead of text embeddings (as in `CompVis/stable-diffusion-v1-5`).  
Image mixing is done via weightening embeddings of base images.   
```
embedding_for_generation = w * embedding_from_the_first_image + (1 - w) * embedding_from_the_second_image
```
It is possible due to the latent space properties like in `word2vec`.
