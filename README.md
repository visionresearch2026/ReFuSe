# ReFuSe: Region-level Fusion of a Source model and a VLM for source-free domain adaptive segmentation

ReFuSe adapts a segmentation model trained on a synthetic source domain (GTA5 or SYNTHIA) to
Cityscapes **without source data and without target labels**. It fuses two teachers inside
SAM2 regions: the source model and an open-vocabulary vision-language model (CorrCLIP or
ProxyCLIP). At test time only the adapted segmentation network is used.

- **Stage 1** (`train_stage1.py`): a region-level agreement / reliability rule decides, for every
  SAM2 region, between a hard label (both teachers agree), a soft label from the more reliable
  teacher, or no supervision. It adds class-balanced CE, text z-score verification of rare
  labels (partial labels), and a rule for classes the VLM cannot name.
- **Stage 2** (`train_stage2.py`): refinement with the frozen Stage 1 model as reference and a dual
  prototype bank (target features and VLM features) that resolves disagreements; losses
  `L_compress` and `L_consist`.
- **Model selection without target labels**: 200 held-out training images. Stage 1 keeps the lowest
  disagreement rate with the VLM; Stage 2 keeps the highest mIoU against the consensus
  pseudo-label (source pseudo-label where it equals the VLM label).

One script per stage covers every setting. The experiment config picks the setting:

| config | source | backbone | VLM |
|---|---|---|---|
| `configs/gta5_deeplabv2_corrclip.yaml` | GTA5 | DeepLabV2-ResNet101 | CorrCLIP |
| `configs/gta5_mitb5_corrclip.yaml` | GTA5 | MiT-B5 + DAFormer head | CorrCLIP |
| `configs/synthia_deeplabv2_corrclip.yaml` | SYNTHIA | DeepLabV2-ResNet101 | CorrCLIP |
| `configs/synthia_mitb5_corrclip.yaml` | SYNTHIA | MiT-B5 + DAFormer head | CorrCLIP |
| `configs/*_proxyclip.yaml` | same four | | ProxyCLIP |

## Results (Cityscapes val, single seed)

CorrCLIP as the VLM; source-only → Stage 1 → Stage 2.

| | DeepLabV2-ResNet101 | MiT-B5 |
|---|---|---|
| GTA5 → Cityscapes, mIoU | 36.9 → 48.9 → **52.6** | 45.4 → 49.7 → **54.6** |
| SYNTHIA → Cityscapes, mIoU16 | 35.2 → 53.4 → **55.5** | 45.0 → 53.9 → **54.4** |

SYNTHIA: DeepLabV2 numbers use the 16-class-head protocol, MiT-B5 numbers the DAFormer
protocol (see `evaluate.py`).

## Repository layout

```
train_stage1.py            Stage 1
train_stage2.py            Stage 2
evaluate.py                Cityscapes val evaluation (run once per final model)
generate_pseudolabels.py   source pseudo-labels on Cityscapes train
refuse/
  backbones.py             DeepLabV2-ResNet101 and MiT-B5 (mmseg) wrappers
  vlm.py                   CorrCLIP (pre-extracted) and ProxyCLIP (on the fly) teachers
  proxyclip_extractor.py   frozen ProxyCLIP feature / logit extractor
  dual_prototype_bank.py   target / VLM prototype banks (Stage 2)
  config.py, common.py
configs/                   one YAML per experiment
source_models/             mmseg configs of the MiT-B5 source models, SYNTHIA split and label conversion
```

## Installation

- **DeepLabV2 experiments**: Python 3.8+, PyTorch ≥ 1.10, `opencv-python`, `numpy`, `tqdm`,
  `pyyaml`, `tensorboard` (`pip install -r requirements.txt`).
- **MiT-B5 experiments**: the environment of [DAFormer](https://github.com/lhoyer/DAFormer) /
  [HRDA](https://github.com/lhoyer/HRDA) (PyTorch 1.10, `mmcv-full==1.4.0`) plus `pyyaml`. Set
  `mmseg_root` in the config to that repository (its own `mmseg` package is used).
- **ProxyCLIP experiments**: clone [ProxyCLIP](https://github.com/mc-lan/ProxyCLIP), install its
  requirements and set `vlm.proxyclip_root`. DINO ViT-B/8 is downloaded through `torch.hub`.

## Data preparation

All paths are set in the config. Every file is keyed by the Cityscapes image stem
`<city>_<seq>_<frame>`.

1. **Cityscapes** (`data.cityscapes_root`): `leftImg8bit/{train,val}` and
   `gtFine/val/*_gtFine_labelTrainIds.png` (val labels are read only by `evaluate.py`).
2. **SAM2 masks** (`data.sam2_root`): one `<city>/<stem>_leftImg8bit.npz` per training image with
   key `instance_mask`, a (1024, 2048) int map of region ids from the SAM2 automatic mask
   generator (`points_per_side=32`).
3. **Source models** (`source_ckpt`):
   - DeepLabV2, GTA5 / SYNTHIA: the source-only checkpoints of
     [MaxSquareLoss](https://github.com/ZJULearning/MaxSquareLoss) (a 16-class SYNTHIA head is
     expanded to 19 outputs automatically).
   - MiT-B5: train with the DAFormer/HRDA code and `source_models/*_source_only.py` (checkpoints
     are selected on source validation data only).
4. **CorrCLIP outputs** (for `*_corrclip.yaml`), per training image:
   `vlm.seg_root/<city>/<stem>_leftImg8bit.png` (CorrCLIP label map with SAM2 voting, 1024×2048)
   and `vlm.feat_root/<city>/<stem>_leftImg8bit.pt` (dense CLIP-space feature map, (512, h, w)),
   plus the 19 class text embeddings `vlm.text_embed` (`{'embeddings': (19, 512)}`), produced with
   [CorrCLIP](https://github.com/zdk258/CorrCLIP). ProxyCLIP needs no preprocessing.
5. **Source pseudo-labels**:
   ```bash
   python generate_pseudolabels.py --config configs/gta5_deeplabv2_corrclip.yaml --gpu 0
   ```

## Training and evaluation

```bash
C=configs/gta5_deeplabv2_corrclip.yaml
python train_stage1.py --config $C --gpu 0          # -> <work_dir>/stage1/best.pth
python train_stage2.py --config $C --gpu 0          # -> <work_dir>/stage2_contrastoff_denoiseoff/best.pth
python evaluate.py --config $C --checkpoint work_dirs/gta5_deeplabv2_corrclip/stage2_contrastoff_denoiseoff/best.pth
```

Use one GPU per run (24 GB is enough). Stage 1 resumes with `--start-epoch N` (from
`dsp_epoch<N-1>.pth`), Stage 2 with `--resume <run>/dsp_stage_epochNN.pth`. `--debug` runs on two
images per city.

### Ablations

```bash
# Stage 1
python train_stage1.py --config $C --lambda-partial 0 --run-name stage1_noPartial
python train_stage1.py --config $C --mu-balance 0 --run-name stage1_noBalance
# Stage 2 losses
python train_stage2.py --config $C --lambda-cons 0 --run-name stage2_noConsist
python train_stage2.py --config $C --lambda-comp 0 --run-name stage2_noCompress
python train_stage2.py --config $C --lambda-comp 0 --lambda-cons 0 --run-name stage2_noCompConsist
# optional Stage 2 components (off in the main results)
python train_stage2.py --config $C --proto-contrast on
python train_stage2.py --config $C --proto-denoise on
```

## Acknowledgements

This code builds on [DAFormer](https://github.com/lhoyer/DAFormer) /
[HRDA](https://github.com/lhoyer/HRDA), [MaxSquareLoss](https://github.com/ZJULearning/MaxSquareLoss),
[CorrCLIP](https://github.com/zdk258/CorrCLIP), [ProxyCLIP](https://github.com/mc-lan/ProxyCLIP)
and [SAM 2](https://github.com/facebookresearch/sam2).

## Citation

```bibtex
@article{refuse,
  title  = {ReFuSe: Region-level Fusion of a Source model and a VLM for source-free domain adaptive segmentation},
  author = {},
  year   = {}
}
```
