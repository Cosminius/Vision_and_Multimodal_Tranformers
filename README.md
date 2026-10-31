# Vision and Multimodal Transformers

This repository contains my notebooks for learning how **Vision Transformers** and **multimodal models** work. I go from building a ViT from scratch, to fine-tuning pretrained models, to using models that connect images and text (CLIP, BLIP-2, Gemini).

Built with **PyTorch** and **Hugging Face Transformers**.

## What's inside

### `VisionTransformers.ipynb` – Vision Transformers
- Implementing the original ViT architecture from scratch: patch embedding with a `Conv2d` layer, CLS token, positional embeddings and a Transformer encoder.
- Fine-tuning a pretrained ViT (`google/vit-base-patch16-224-in21k`) on the **Oxford-IIIT Pets** dataset (37 breeds) with the Hugging Face `Trainer`.
- Exploring self-supervised learning (DINO) by visualizing what each attention head focuses on in an image.

### `Vit_101FOOD.ipynb` – Fine-tuning vs zero-shot on Food-101
- Fine-tuning a ViT on the **Food-101** dataset (101 dishes).
- Comparing it with **CLIP used zero-shot**, without any training, only by giving it the names of the 101 classes.
- Building the full pipeline by hand: data collators, metrics, batched GPU inference and text–image similarity.

| Model | Approach | Accuracy |
|---|---|---|
| ViT-B/16 (`google/vit-base-patch16-224`) | Fine-tuned, 4 epochs | **89.6%** |
| CLIP ViT-B/32 (`openai/clip-vit-base-patch32`) | Zero-shot | **83.1%** |

*Both models were evaluated on the same 5,250 held-out test images.*

### `MultimodalTransformers.ipynb` – Multimodal models
- **CLIP**: zero-shot image classification, first with a pipeline, then step by step (embeddings, similarity, temperature, softmax).
- **Perceiver**: implementing and visualizing Fourier positional encodings.
- **BLIP-2**: generating captions for images.
- **Gemini API**: asking questions about my own photo.

### Coming next
- `PhotoSearchEngine.ipynb`: a search engine over my own photos, using CLIP to find images by text or by another image.
- `Caption_your_Photos.ipynb`: automatic captions for a photo collection with BLIP-2.

## Setup

```bash
pip install torch torchvision "transformers>=5" datasets accelerate scikit-learn matplotlib pillow python-dotenv google-genai
```

For the Gemini example, add a `.env` file with `API_KEY=your_key`.

## License

Apache 2.0
