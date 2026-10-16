# SNN training

Run commands from the repository root. Use Python 3.8, PyTorch 1.9.1+cu111, timm 0.4.12 and the dependencies listed in the main README, including spikingjelly.

Prepare:
- ImageNet images under `data/train/<class>/` and `data/val/<class>/`.
- BLIP retrieval weights (`model_base_retrieval_flickr.pth`).
- The 5000-entry codebook (`flicker_purning_05w_codebook.pth`).
- BERT files in `BERT/`, as described in the main README.

```bash
python training/train_SNN.py \
  --data-dir /path/to/imagenet \
  --teacher-weights /path/to/model_base_retrieval_flickr.pth \
  --codebook /path/to/flicker_purning_05w_codebook.pth \
  --output output/train
```

Training settings are in `training/imagenet.yml`. To resume, add `--resume /path/to/checkpoint.pth.tar`. Checkpoints are saved under `--output`.

For two GPUs, replace `python training/train_SNN.py` with `python -m torch.distributed.launch --nproc_per_node=2 training/train_SNN.py` and keep the same arguments.
