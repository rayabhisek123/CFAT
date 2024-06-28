## [CFAT: Unleashing Triangular Windows for Image Super-resolution](https://openaccess.thecvf.com/content/CVPR2024/html/Ray_CFAT_Unleashing_Triangular_Windows_for_Image_Super-resolution_CVPR_2024_paper.html)<sup>1</sup>
![](https://i.imgur.com/waxVImv.png)
[Abhisek Ray](https://scholar.google.co.in/citations?user=a7HOeC8AAAAJ&hl=en)<sup>1</sup>, [Gaurav Kumar]()<sup>1</sup>, [Maheshkumar H. Kolekar]()<sup>1</sup>

<sup>1</sup>Indian Institute of Technology Patna, India

[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2403.16143)
[![GitHub Stars](https://img.shields.io/github/stars/rayabhisek123/CFAT?style=social)](https://github.com/rayabhisek123/CFAT)
![Visitors](https://api.visitorbadge.io/api/visitors?path=https%3A%2F%2Fgithub.com%2Frayabhisek123%2FCFAT&countColor=%23263759)

⭐ If you find "**CFAT**" helpful to your research, Don't forget to give a star to this repo. Thanks! 🤗

## :climbing: Updates
- ✅ 2024-03-24: Release the first version of the paper at Arxiv.
- ✅ 2024-03-24: Release the supplementary material of the paper at Arxiv.
- ✅ 2024-03-24: Release the codes, models and results of CFAT.
- ✅ 2024-06-11: Update repo with CVPR version.
- **(To do)** Release the small (CFAT-S) and large (CFAT-L) versions of our model.
- **(To do)** Add the pre-trained model of CFAT for SRx4.
- **(To do)** Add a Replicate demo for CFAT model implementation. 
- **(To do)** Release extensive code of CFAT for Multiple Image Restoration tasks.

<hr />

![main figure](figures/model_vs_SOTA.png)
> **Abstract:** *Transformer-based models have revolutionized the field of image super-resolution (SR) by harnessing their inherent ability to capture complex contextual features. The overlapping rectangular shifted window technique used in transformer architecture nowadays is a common practice in super-resolution models to improve the quality and robustness of image upscaling. However, it suffers from distortion at the boundaries and has limited unique shifting modes. To overcome these weaknesses, we propose a non-overlapping triangular window technique that synchronously works with the rectangular one to mitigate boundary-level distortion and allows the model to access more unique sifting modes. In this paper, we propose a Composite Fusion Attention Transformer (CFAT) that incorporates triangular-rectangular window-based local attention with a channel-based global attention technique in image super-resolution. As a result, CFAT enables attention mechanisms to be activated on more image pixels and captures long-range, multi-scale features to improve SR performance. The extensive experimental results and ablation study demonstrate the effectiveness of CFAT in the SR domain. Our proposed model shows a significant 0.7 dB performance improvement over other state-of-the-art SR architectures.*
<hr />

## Highlight

The triangular window mechanism that we proposed is beneficial not only in super-resolution tasks but also in various other computer vision applications that implement the rectangular window technique in their mainframe.

<hr />

## Results

- **Quantitative Results**

<br>
<p align="center">
  <img src="figures/quantitative_results.png" align="center" width="700">
  <br>
  Fig. Comparing performance (PSNR in dB) of various state-of-the-art models with CFAT.
</p>
<be>

- **Qualitative Results**

<br>
<p align="center">
  <img src="figures/All_fig.png" align="center" width="700">
  <br>
  Fig. Visual Comparison of CFAT with other state-of-the-art methods.
</p>
<be>

- **LAM Results**

<br>
<p align="center">
  <img src="figures/LAM_Compare.png" align="center" width="700">
  <br>
  Fig. LAM results and corresponding Diffusion Index for CFAT and various SOTA methods.
</p>
<be>

<hr />

## Training Settings
- Requirements
  > - Platforms: Ubuntu 20.04.2, cuda-11.2.0
  > - Python 3.8.18, PyTorch == 2.1.0
  > - Requirements: see requirements.txt

- Installation

```shell

# download code
git clone https://github.com/rayabhisek123/CFAT
cd CFAT
pip install -r requirements.txt
pip install basicsr
python setup.py develop
```
- Datasets
  
  Download the training datasets:

  > - [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)
  > - [Flickr2K](https://www.kaggle.com/datasets/hliang001/flickr2k?select=Flickr2K)

  Download the testing datasets:

  > - [Set5, Set14, BSD100，Urban100](https://figshare.com/articles/dataset/BSD100_Set5_Set14_Urban100/21586188)
  > - [Manga109](https://www.kaggle.com/datasets/guansuo/manga109)

<hr />
  
## Training

Command to train **CFAT** after placing datasets at their respective repositories

```shell
CUDA_VISIBLE_DEVICES=0,1,2 python3 train.py
```

<hr />

## Checkpoints (Pre-trained Models)

The inference results on benchmark datasets will be available soon.

<hr />

## Testing

Run the following command after placing the Pre-trained Models in the given repository

```shell
CUDA_VISIBLE_DEVICES=0,1 python3 test.py
```
**Note: For different configurations, change the argument values. We will update the corresponding configuration files(.yml) soon.

<hr />

## Testing Results

<br>
<p align="center">
  <img src="figures/quantitative_results_table.png" align="center" width="700">
  <br>
  Fig. Quantitative comparison of the CFAT with various state-of-the-art SR methods. Red: Best & Green: Second Best.
</p>
<be>

<hr />

## Citations
#### BibTeX

```
@article{ray2024cfat,
  title={CFAT: Unleashing TriangularWindows for Image Super-resolution},
  author={Ray, Abhisek and Kumar, Gaurav and Kolekar, Maheshkumar H},
  journal={arXiv preprint arXiv:2403.16143},
  year={2024}
}

```
<hr />

## Acknowledgement
**Some parts of this code are adapted from:**
- [HAT](https://github.com/XPixelGroup/HAT)
- [SwinIR](https://github.com/JingyunLiang/SwinIR)
  
We thank the authors for sharing codes of their great works.

<hr />

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT) and was originally developed by [@abhisek-ray](https://github.com/rayabhisek123).

<hr />

## Contact
If you have any questions, please email rayabhisek0610@gmail.com to discuss with the authors.



