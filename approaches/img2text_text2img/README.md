## Installation
```
pip install -r requirements.txt
```

## Usage

Run
```
python3 run.py /home/username/image1.jpg /home/username/image2.jpg
```

### Img To Text + Text To Image

This approach uses an already well-trained [Img2Img Stable Diffusion](https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/img2img).
And generates prompt from images using Image To Text [Transformers](https://huggingface.co/tasks/image-to-text).

It is also possible to use CLIP Interrogator for generating prompt from image.