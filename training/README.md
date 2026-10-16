# Original codebook trimming and SNN training source

This directory contains the recovered server training scripts and their original model/data modules. Source files are unchanged; hashes are listed in `SOURCE_MANIFEST.json`. Work from this directory so that training imports the original model, which returns logits. The inference model at the repository root returns discrete indices and is intentionally kept separate.

## Training stages

1. `codebook_purning.py` counts codeword assignments on Flickr30k training images and retains the 5000 most frequent codewords. It saves `flicker_purning_05w_codebook.pth`.
2. `train_SNN.py` extracts BLIP teacher features, assigns codeword-index targets, and trains the SNN using cross-entropy. The observed codebook shape is **5000 x 768**, and the SNN output head has 5000 classes.

## Setup and commands

The verified server environment used Python 3.8, PyTorch 1.9.1+cu111 and timm 0.4.12, with the existing spikingjelly and BLIP dependencies. See the repository's dependency and pretrained-asset instructions. Prepare ImageNet train/validation images, BLIP retrieval weights, BERT assets and the 5000-entry codebook. Codebook trimming additionally needs the large source codebook and Flickr30k data.

The source preserves original absolute filesystem paths. Update them in the scripts/model modules and configuration for your installation. No datasets or pretrained weights are bundled here.

```bash
cd training
# Optional: construct the small codebook when the large source codebook is available.
python codebook_purning.py
# Original two-GPU training launch; foreground execution makes errors visible.
python -m torch.distributed.launch --nproc_per_node=2 train_SNN.py
```

`train.sh` preserves the original background launch. `imagenet.yml` is the current recovered configuration (40 epochs). The historical `output/train/20241215-215219-resnet101-224/args.yaml` and `summary.csv` record a 20-epoch run; its checkpoint was inspected on the server and has a 5000-class output head. The generic `num_classes: 1000` configuration field does not determine this head: the source constructs `sdt` with its 5000-class model default. Use the recorded settings appropriate to the run being reproduced rather than treating the current config as the exact historical setup.

## Verification

The unchanged `train_SNN.py` entry point passed a short single-GPU, full-precision run with four real training images and four validation images (batch size 2, one epoch). It completed teacher-label generation, forward/backward computation, optimizer updates, validation and checkpoint saving. Parameters changed and remained finite. Only temporary test configuration and dataset size were adjusted; the training source was not patched. This verifies basic training execution, not a full dual-GPU run or reproduction of published metrics.

Existing inference scripts at the repository root remain unchanged.
