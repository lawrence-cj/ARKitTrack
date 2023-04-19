<div align="center">

## [CVPR 2023] ARKitTrack: A New Diverse Dataset for Tracking Using Mobile RGB-D Data

### [Project Page](https://arkittrack.github.io/) |  [arXiv](https://arxiv.org/abs/2303.13885)
![teaser](figures/data_vis.jpg)
</div>


This is a PyTorch implementation of the paper [ARKitTrack: A New Diverse Dataset for Tracking Using Mobile RGB-D Data](https://arkittrack.github.io/). Code will be released here.

<div class="is-size-5 publication-authors">
  <span class="author-block">
    <a href="https://scholar.google.com/citations?hl=en&user=rk1ozXMAAAAJ">Haojie Zhao</a><sup>*1</sup>,</span>
  <span class="author-block">
    <a>Junsong Chen</a><sup>*1</sup>,</span>
  <span class="author-block">
    <a href="http://faculty.dlut.edu.cn/wanglj/zh_CN/index.htm">Lijun Wang</a><sup>1</sup>,
  </span>
  <span class="author-block">
    <a href="https://scholar.google.com/citations?hl=en&user=D3nE0agAAAAJ">Huchuan Lu</a><sup>1,2</sup>
  </span>
(* indicates equal contributions)
</div>

<div class="is-size-5 publication-authors">
<span class="author-block"><sup>1</sup>Dalian University of Technology, China,</span>
<span class="author-block"><sup>2</sup>Peng Cheng Laboratory, China</span>
</div>
Contact at: jschen@mail.dlut.edu.cn, haojie_zhao@mail.dlut.edu.cn

[//]: # (<nobr>&#40;* indicates equal contributions&#41;</nobr>)


---
### News

- Data and code for VOS are coming soon ...
- [2023/04/20] Release code for VOT.

---


### Dataset
- VOT_test_set: [[Baidu Drive]](), VOS_test_set: [[Baidu Drive]](), Training_set: [[Baidu Drive]]()

---


### 1. Installation
```shell
# 1. Clone this repo
git clone https://github.com/lawrence-cj/ARKitTrack.git
cd ARKitTrack

# 2. Create conda env
conda env create -f art_env.yml
conda activate art

# 3. Install mmcv-full, mmdet, mmdet3d for the BEV pooling, which is from bevfusion.
pip install openmim
mim install mmcv-full==1.4.0
mim install mmdet==2.20.0
python setup.py develop  # mmdet3d
```

### 2. Set project paths
Run the following command to set paths for this project.
```shell
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output
```
After running this command, you can also modify paths by editing these two files:
`lib/train/admin/local.py` and `lib/test/evaluation/local.py`.


### 3. Evaluation
Download our trained models from [Google Drive](https://drive.google.com/file/d/1uNcUSTXDGkegQ_1r5XtHiftnlH7D-eSC/view?usp=share_link) and uncompress them to `output/checkpoints/`.

Change the corresponding dataset paths in `lib/test/evaluation/local.py`.

Run the following command to test on different datasets.
```shell
python tracking/test.py --tracker art --param vitb_384_mae_ce_32x4_ep300 --dataset depthtrack --threads 2 --num_gpus 2
```
- `--config vitb_384_mae_ce_32x4_ep300` is used for cdtb and depthtrack.
- `--config vitb_384_mae_ce_32x4_ep300_art` is used for arkittrack.
- `--debug 1` for visualization.
- `--dataset`: [depthtrack, cdtb, arkit].

The raw results are stored in [Google Drive](https://drive.google.com/file/d/14jCQTpl3B5oPUuVncV-5R3pWlh7z_T53/view?usp=share_link).
### 4. Training
Download the pre-trained weights from [Google Drive](https://drive.google.com/file/d/1FVxEnyESw-10A2dvJj2OVzJGC2tFlR1X/view?usp=share_link) and uncompress it to `pretrained_models/`.

Change the corresponding dataset paths in `lib/train/admin/local.py`.

Run the following command to train for vot.
```shell
python tracking/train.py --script art --config vitb_384_mae_ce_32x4_ep300 --save_dir ./output --mode multiple --nproc_per_node 2
```
- `--config vitb_384_mae_ce_32x4_ep300`: train with depthtrack, test on cdtb and depthtrack.
- `--config vitb_384_mae_ce_32x4_ep300_art`: train with arkittrack, test on arkittrack.
- You can modify the config `yaml` files for your own datasets.

---
### Acknowledgments
Thanks for the [OSTrack](https://github.com/botaoye/OSTrack) and [BEVFusion](https://github.com/mit-han-lab/bevfusion) projects, which help us to quickly implement our ideas.

### Citation
```
@inproceedings{hjz2023arkittrack,
      author    = {Haojie Zhao and Junsong Chen and Lijun Wang and Huchuan Lu},
      title     = {ARKitTrack: A New Diverse Dataset for Tracking Using Mobile RGB-D Data},
      booktitle = {CVPR},
      year      = {2023},
}
```


### License

This project is under the MIT license. See [LICENSE](LICENSE) for details.
