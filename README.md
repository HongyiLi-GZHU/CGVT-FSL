# CGVT-FSL：Concept-Guided Visual-Textual Few-Shot Learning for Cross-Domain Hyperspectral Image Classification

Thank you for your interest in the code related to our paper, "Concept-Guided Visual-Textual Few-Shot Learning for Cross-Domain Hyperspectral Image Classification".

# Abstract

Visual-textual few-shot learning (VT-FSL) models have demonstrated effectiveness in cross-domain hyperspectral image (HSI) classification. The rationale behind this is that text information contains valuable prior knowledge about ground objects, which is independent of data domains. However, existing VT-FSL methods typically perform global image-label alignment, where textual features are derived from simple class names or short label phrases. However, such label-level alignment captures only coarse-grained semantics and is highly sensitive to domain shifts. To overcome this limitation, we argue that concept-level semantics—such as material or density—are more stable and transferable across domains. Therefore, we propose a novel method called concept-guided visual-textual few-shot learning (CGVT-FSL). Firstly, we design a concept-embedded class-label text description (C2TD) method that embeds LLM-guided knowledge into class-level textual representations, enabling interpretable and transferable semantic alignment. Secondly, we propose a novel backbone network called visual-textual dual-branch transformer (VTDFormer), which effectively extracts both visual and textual spatial-spectral features across domains. Moreover, we propose a spatial-spectral-textual domain adversarial module (SSTDAM) to further mitigate domain shift by leveraging the shared linguistic domain. Finally, extensive experiments on six HSI datasets demonstrate the superiority of the proposed TAFSL when compared with state-of-the-art FSL methods.

![Alt text](figures/Fig1.png)


# Current Status
The source code for this paper is currently being prepared and will be made available soon. Please check back for updates. We are working to ensure that the code is well-documented and easy to use.

