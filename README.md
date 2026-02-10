# SpectralMAE: Spectral Masked Autoencoder for Hyperspectral Remote Sensing Image Reconstruction

## Overview
SpectralMAE is a self-supervised learning framework designed for hyperspectral image (HSI) reconstruction. By leveraging a spectral masking strategy, the model learns the inherent redundancy and continuity of spectral data, enabling robust reconstruction from limited input bands (e.g., RGB).

### Key Features
- **Spectral Masking Strategy**: High masking rates (up to 90%) for efficient self-supervised pre-training.
- **Flexible Input**: Capable of handling various spectral sensor inputs.
- **Transformer-based Backbone**: Efficiently captures long-range spectral dependencies.

---

## 📝 Publication
Our paper is published in **Sensors (Volume 23, Issue 7)**. 
You can access the full paper here: [https://www.mdpi.com/1424-8220/23/7/3728](https://www.mdpi.com/1424-8220/23/7/3728)

## 🎓 Citation
If you find our work or this code useful for your research, please cite our paper using the following BibTeX entry:

@Article{s23073728,
AUTHOR = {Zhu, Lingxuan and Wu, Jiaji and Biao, Wang and Liao, Yi and Gu, Dandan},
TITLE = {SpectralMAE: Spectral Masked Autoencoder for Hyperspectral Remote Sensing Image Reconstruction},
JOURNAL = {Sensors},
VOLUME = {23},
YEAR = {2023},
NUMBER = {7},
ARTICLE-NUMBER = {3728},
URL = {https://www.mdpi.com/1424-8220/23/7/3728},
PubMedID = {37050788},
ISSN = {1424-8220},
ABSTRACT = {Accurate hyperspectral remote sensing information is essential for feature identification and detection. Nevertheless, the hyperspectral imaging mechanism poses challenges in balancing the trade-off between spatial and spectral resolution. Hardware improvements are cost-intensive and depend on strict environmental conditions and extra equipment. Recent spectral imaging methods have attempted to directly reconstruct hyperspectral information from widely available multispectral images. However, fixed mapping approaches used in previous spectral reconstruction models limit their reconstruction quality and generalizability, especially dealing with missing or contaminated bands. Moreover, data-hungry issues plague increasingly complex data-driven spectral reconstruction methods. This paper proposes SpectralMAE, a novel spectral reconstruction model that can take arbitrary combinations of bands as input and improve the utilization of data sources. In contrast to previous spectral reconstruction techniques, SpectralMAE explores the application of a self-supervised learning paradigm and proposes a masked autoencoder architecture for spectral dimensions. To further enhance the performance for specific sensor inputs, we propose a training strategy by combining random masking pre-training and fixed masking fine-tuning. Empirical evaluations on five remote sensing datasets demonstrate that SpectralMAE outperforms state-of-the-art methods in both qualitative and quantitative metrics.},
DOI = {10.3390/s23073728}
}

