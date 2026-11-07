# Vision and Multimodal Transformers

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-red)
![Transformers](https://img.shields.io/badge/🤗_Transformers-v5-yellow)
![License](https://img.shields.io/badge/License-Apache_2.0-green)

My notebooks for learning how Vision Transformers and multimodal models work. I started by building a ViT by hand, then moved on to fine-tuning pretrained models, and finally to models that understand images and text together (CLIP, BLIP-2, Gemini). The last two notebooks turn that into small projects that work on my own photos.

Everything is written in PyTorch with Hugging Face Transformers.

## Results

| Notebook | Task | Model | Result |
|---|---|---|---|
| `VisionTransformers` | Fine-tuning on Oxford-IIIT Pets (37 breeds) | ViT-B/16 (`vit-base-patch16-224-in21k`) | **92.1%** accuracy (3 epochs) |
| `Vit_101FOOD` | Fine-tuning on Food-101 (101 dishes) | ViT-B/16 (`vit-base-patch16-224`) | **89.6%** accuracy (4 epochs) |
| `Vit_101FOOD` | Zero-shot on Food-101, no training | CLIP ViT-B/32 | **83.1%** accuracy |
| `MultimodalTransformers` | Captioning a photo of my cats | BLIP-2 (OPT-2.7B) | "two cats laying on a couch" |
| `MultimodalTransformers` | Asking a question about the same photo | Gemini 2.5 Flash | `[cat, 2]` |

The two Food-101 numbers come from the same 5,250 test images.

One thing worth knowing before you compare them: the ViT uses smaller patches (B/16 vs B/32) and was pretrained on labeled ImageNet data, so it starts with an advantage. Still, CLIP gets to 83% without seeing a single Food-101 training image, only the class names.

## What's inside

### `VisionTransformers.ipynb`
- A Vision Transformer built from scratch: patch embedding done with a `Conv2d`, a CLS token, learned positional embeddings and a Transformer encoder.
- Fine-tuning a pretrained ViT on Oxford-IIIT Pets with the Hugging Face `Trainer`. Accuracy goes 90.2% → 92.3% → 92.1% over three epochs, and it takes about 16 minutes on an Apple Silicon GPU.
- Looking at the CLS token's attention in the last layer, one map per head, drawn over the original image.

### `Vit_101FOOD.ipynb`
- Fine-tuning a ViT on Food-101 (about 46 minutes for 4 epochs on a CUDA GPU).
- The same task with CLIP used zero-shot: I turn each class name into a sentence like "This is a photo of pizza, a type of food", embed it, and pick the closest one for every image.
- Written mostly by hand: a custom data collator, an accuracy metric, and batched GPU inference for CLIP.

### `MultimodalTransformers.ipynb`
- **CLIP**: zero-shot classification, first with the `pipeline`, then step by step (embeddings → similarity → temperature → softmax) to see what the pipeline does underneath.
- **Fourier positional encodings**, as used by the Perceiver, with a plot of every encoding channel.
- **BLIP-2**: generating a caption for an image.
- **Gemini API**: sending my own photo (`my_cats_photo.jpg`) and asking what is in it and how many.

### `PhotoSearchEngine.ipynb`
A small search engine for my own photos.
1. Every photo is embedded with CLIP and the vectors are normalized.
2. A query, either text or an image, is embedded the same way.
3. Cosine similarity against all stored photos gives the best match.

For example, searching for *"a photo with guys sitting on chairs"* returns the right photo from my folder. The similarity search is a plain matrix multiplication, which is fine for a few photos.

### `Caption_your_Photos.ipynb`
Automatic captions for a folder of photos with BLIP-2. The model is wrapped in a `TheCaptioner` class that loads it in `bfloat16`, spreads it across the available devices, and prompts it in a question-answer style (`Question: What is shown in this image? Answer:`). On one of my photos it produced "A man sitting at a computer".

## Setup

```bash
pip install torch torchvision "transformers>=5" datasets accelerate scikit-learn matplotlib pillow python-dotenv google-genai
```

A few things to know if you want to run the notebooks:

- **Hardware.** I ran Pets, the multimodal notebook and the photo search on a Mac (`mps`), and Food-101 and the captioning on an NVIDIA GPU (`cuda`). The device is written directly in some cells, so change it to match your machine. BLIP-2 needs a GPU with roughly 8 GB of memory or more.
- **Gemini.** Create a `.env` file with `API_KEY=your_key`.
- **My photos.** The `Imagini/` folder is not in the repo, because those are personal pictures. Put your own images in a folder with that name and the search and captioning notebooks will work on them.
- **Datasets and checkpoints.** Food-101 is downloaded into `datasets/` the first time you run the notebook. Trained models are saved locally and are not tracked by git.

## Ideas for next steps

- Use FAISS for the photo search and return the top-k results instead of only the best one.
- Caption the whole photo folder in a loop and save the captions to a file, then search through them.
- Wrap the search in a small Gradio app.

## License

Apache 2.0
