# TAFSL：Text-Aware Few-Shot Learning for Cross-Domain Hyperspectral Image Classification

Thank you for your interest in the code related to our paper, "Text-Aware Few-Shot Learning for Cross-Domain Hyperspectral Image Classification".

# Abstract

Few-shot learning (FSL) models have demonstrated effectiveness in cross-domain hyperspectral image (HSI) classification. The rationale behind this is that such models trained on knowledge from the source domain can be easily generalized to the target domain using a small number of labeled samples. Recently, multimodal foundation models, particularly image-text models, have shown significant advantages in remote sensing image processing. More importantly, text information contains valuable prior knowledge about ground objects, which is independent of data domains. However, existing FSL methods for cross-domain HSI classification usually ignore the linguistic modality and fail to fully exploit it to enhance the generalization performance of models. To address this issue, we propose a novel text-aware few-shot learning (TAFSL) method for cross-domain HSI classification. Firstly, we introduce a domain-agnostic class-label text description (DCTD) method to generate class-label text that effectively describes both intra-class and inter-class relationships through secondary classification of class names. Secondly, we propose a novel backbone network called visual-textual dual-branch transformer (VTDFormer), which effectively extracts both visual and textual spatial-spectral features across domains. Moreover, we propose a spatial-spectral-textual domain adversarial module (SSTDAM) to further mitigate domain shift by leveraging the shared linguistic domain. Finally, extensive experiments on six HSI datasets demonstrate the superiority of the proposed TAFSL when compared with state-of-the-art FSL methods. The codes will be available from the website: https://github.com/HongyiLi-GZHU/TAFSL.

![Alt text](figures/fig1.pdf)

![Alt text](figures/fig2.pdf)

# Current Status
The source code for this paper is currently being prepared and will be made available soon. Please check back for updates. We are working to ensure that the code is well-documented and easy to use.

