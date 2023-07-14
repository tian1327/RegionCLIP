### Dependency set up

```bash
conda create -n regionclip python=3.9
conda activate regionclip

# the following conda install command may install cpu version of torch,
# make sure you run conda list to check the torch is compiled with CPU
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

# to install torch compiled with GPU, run the following
pip3 install torch==2.0.1 torchvision torchaudio

pip install opencv-python timm diffdist h5py sklearn ftfy
pip install git+https://github.com/lvis-dataset/lvis-api.git

python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

pip install setuptools==59.0.1

# if running following innference command gives you error: `KeyError: 'Non-existent config key: MODEL.CLIP'`, install detectron2 by
pip install -e .

```

#### Env setup

```bash
# to export current env to yml file
conda env export > environment.yml

# to create env from yml file
conda env create -f environment.yml

```


### Test zero-shot object detection in ActivityNet dataset

1. Slice the frames of selected videos in folder `../data/RegionCLIP_samples/`
    * `python ../ActivityNet-Video-Downloader/video2image.py ../data/RegionCLIP_samples/ ../data/RegionCLIP_samples/ --level 1 --lib ffmpeg -fps 1`
2. Copy the selected sliced frames to folder `datasets/custom_images/`
    * rename the sliced frames so they start with video id by `rename 's/^img/yjazHd6a5SQ_img/' img_*.jpg`
    * `cp ../data/RegionCLIP_samples/CN01Gm2Yc4k/*.jpg datasets/custom_images/`
3. Run inference for object detection with pretrained RegionCLIP model

```bash
python3 ./tools/train_net.py \
--eval-only \
--num-gpus 1 \
--config-file ./configs/LVISv1-InstanceSegmentation/CLIP_fast_rcnn_R_50_C4_custom_img.yaml \
MODEL.WEIGHTS ./pretrained_ckpt/regionclip/regionclip_pretrained-cc_rn50x4.pth \
MODEL.CLIP.TEXT_EMB_PATH ./pretrained_ckpt/concept_emb/lvis_1203_cls_emb_rn50x4.pth \
MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/LVISv1-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
MODEL.CLIP.TEXT_EMB_DIM 640 \
MODEL.RESNETS.DEPTH 200 \
MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION 18 \
```

4. Generate the boundings boxes with categories and scores

```bash
 python ./tools/visualize_json_results.py \
--input ./output/inference/lvis_instances_results.json \
--output ./output/regions \
--dataset lvis_v1_val_custom_img \
--conf-threshold 0.05 \
--show-unique-boxes \
--max-boxes 25 \
--small-region-px 8100\ 
```


