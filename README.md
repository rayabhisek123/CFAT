### [CVPR 2024] CFAT: Unleashing TriangularWindows for Image Super-resolution [[Paper Link]](https://arxiv.org/abs/2403.16143)
![Visitors](https://api.visitorbadge.io/api/visitors?path=https%3A%2F%2Fgithub.com%2Frayabhisek123%2FCFAT&countColor=%23263759)

[Abhisek Ray](https://scholar.google.co.in/citations?user=a7HOeC8AAAAJ&hl=en)<sup>1</sup>, [Gaurav Kumar]()<sup>1</sup>, [Maheshkumar H. Kolekar]()<sup>1</sup>

<sup>1</sup>Indian Institute of Technology Patna, India

## Updates
- ✅ 2022-05-09: Release the first version of the paper at Arxiv.
- ✅ 2022-05-09: Release the supplementary material of the paper at Arxiv.
- ✅ 2022-05-20: Release the codes, models and results of CFAT.
- **(To do)** Update repo with CVPR version.
- **(To do)** Release the small (CFAT-S) and large (CFAT-L) versions of our model.
- **(To do)** Add the pre-trained model of CFAT for SRx4.
- **(To do)** Add a Replicate demo for CFAT model implementation. 
- **(To do)** Release extensive code of CFAT for Multiple Image Restoration tasks.
- **(To do)** Update the Replicate demo for Real-World SR. 


<br>
<p align="center">
  <img src="figures/model_vs_SOTA.png" align="center" width="70%">
  <br>
  Fig. Proposed CFAT vs other SOTA models.
</p>
<be>

## Abstract

Transformer-based models have revolutionized the field of image super-resolution (SR) by harnessing their inherent ability to capture complex contextual features. The overlapping rectangular shifted window technique used in transformer architecture nowadays is a common practice in super-resolution models to improve the quality and robustness of image upscaling. However, it suffers from distortion at the boundaries and has limited unique shifting modes. To overcome these weaknesses, we propose a non-overlapping triangular window technique that synchronously works with the rectangular one to mitigate boundary-level distortion and allows the model to access more unique sifting modes. In this paper, we propose a Composite Fusion Attention Transformer (CFAT) that incorporates triangular-rectangular window-based local attention with a channel-based global attention technique in image super-resolution. As a result, CFAT enables attention mechanisms to be activated on more image pixels and captures long-range, multi-scale features to improve SR performance. The extensive experimental results and ablation study demonstrate the effectiveness of CFAT in the SR domain. Our proposed model shows a significant 0.7 dB performance improvement over other state-of-the-art SR architectures.

## Highlight

The triangular window mechanism that we proposed is beneficial not only in super-resolution tasks but also in various other computer vision applications that implement the rectangular window technique in their mainframe.

## Citations
#### BibTeX

```
@article{ray2023cfat,
      title={CFAT: Unleashing TriangularWindows for Image Super-resolution},
      author={Ray, Abhisek and Kumar, Gaurav and Kolekar, Maheshkumar},
      journal={arXiv preprint arXiv:2403.16143},
      year={2023}
    }

```
## Acknowledgement
**Some parts of this code are adapted from:**
- HAT
- SwinIR
  
We thank the authors for sharing codes of their great works.


## Contact
If you have any questions, please email rayabhisek0610@gmail.com to discuss with the authors.



