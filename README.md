# PU-EVA
# PU-EVA: An Edge-Vector based Approximation Solution for Flexible-scale Point Cloud Upsampling
This repository contains codes of ICCV2021 paper: Edge-Vector based Approximation for Flexible-scale Point clouds Upsampling (PU-EVA). The proposed PU-EVA decouples the upsampling scales with network architecture, making the upsampling rate flexible in one-time end-to-end training. 

[[Project]](https://github.com/GabrielleTse/PU-EVA) [[Paper]](https://openaccess.thecvf.com/content/ICCV2021/html/Luo_PU-EVA_An_Edge-Vector_Based_Approximation_Solution_for_Flexible-Scale_Point_Cloud_ICCV_2021_paper.html)     

## Overview
High-quality point clouds have practical significance for point-based rendering, semantic understanding, and surface reconstruction. Upsampling sparse, noisy and non-uniform point clouds for a denser and more regular approximation of target objects is a desirable but challenging task. Most existing methods duplicate point features for upsampling, constraining the upsampling scales at a fixed rate. In this work, the arbitrary point clouds upsampling rates are achieved via edge-vector based affine combinations, and a novel design of Edge-Vector based Approximation for Flexible-scale Point clouds Upsampling (PU-EVA) is proposed. The edge-vector based approximation encodes neighboring connectivity via affine combinations based on edge vectors, and restricts the approximation error within a second-order term of Taylor's Expansion. Moreover, the EVA upsampling decouples the upsampling scales with network architecture, achieving the arbitrary upsampling rates in one-time training. Qualitative and quantitative evaluations demonstrate that the proposed PU-EVA outperforms the state-of-the-arts in terms of proximity-to-surface, distribution uniformity, and geometric details preservation.
![图片3](https://user-images.githubusercontent.com/37495877/137577669-842f2a99-fed1-4f53-b302-4ecf667b8b6e.png)
![luo2](https://user-images.githubusercontent.com/37495877/137577671-fccc356c-e930-49c2-bdc0-575160a361b5.png)

## Requirements
* [TensorFlow](https://www.tensorflow.org/)

## Point Cloud Classification
* Run the training script:
``` bash
python main.py
```
* Run the evaluation script after training finished:
``` bash
python evalutate.py

```

## Citation
@InProceedings{Luo_2021_ICCV,

    author    = {Luo, Luqing and Tang, Lulu and Zhou, Wanyi and Wang, Shizheng and Yang, Zhi-Xin},
    
    title     = {PU-EVA: An Edge-Vector Based Approximation Solution for Flexible-Scale Point Cloud Upsampling},
    
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    
    month     = {October},
    
    year      = {2021},
    
    pages     = {16208-16217}
    
}

## License
MIT License

## Acknowledgement
This code is heavily borrowed from [PointNet](https://github.com/charlesq34/pointnet), [DGCNN](https://github.com/charlesq34/pointnet) and [PU-GAN](https://github.com/charlesq34/pointnet).
