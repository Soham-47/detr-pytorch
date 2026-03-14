# DETR: End-to-End Object Detection with Transformers

This repository contains a clean, modular implementation of **DETR (DEtection TRansformer)** from scratch using PyTorch.

## Project Structure

- `models/`: Model architecture
  - `backbone/`: CNN backbone (ResNet) to extract feature maps.
  - `transformer/`: Multi-head attention, Encoder, and Decoder modules.
  - `positional_encoding/`: Sine/Learned positional encodings.
- `matcher/`: Hungarian Matcher for bipartite matching loss.
- `engine/`: Training and evaluation loops.
- `losses/`: Set-based loss functions (Hungarian loss).

## Getting Started

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Training:**
   ```bash
   python train.py --config configs/coco_config.yaml
   ```

3. **Inference:**
   ```bash
   python inference.py --image path/to/image.jpg
   ```

## Roadmap

- [x] Basic Project Structure
- [x] CNN Backbone (ResNet-50)
- [ ] Transformer Encoder/Decoder
- [ ] Hungarian Matcher
- [ ] Training Pipeline
- [ ] COCO Evaluation

## Acknowledgments

Based on the paper: [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872) by Carion et al.
