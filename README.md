# PU-EVA
# PU-EVA: An Edge-Vector based Approximation Solution for Flexible-scale Point Cloud Upsampling
This repository contains codes of ICCV2021 paper: Edge-Vector based Approximation for Flexible-scale Point clouds Upsampling (PU-EVA). The proposed PU-EVA decouples the upsampling scales with network architecture, making the upsampling rate flexible in one-time end-to-end training. 

[[Project]](https://github.com/GabrielleTse/PU-EVA) [[Paper]](https://openaccess.thecvf.com/content/ICCV2021/html/Luo_PU-EVA_An_Edge-Vector_Based_Approximation_Solution_for_Flexible-Scale_Point_Cloud_ICCV_2021_paper.html)     

## Overview
`PU-EVA` . 


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
This code is heavily borrowed from [PointNet](https://github.com/charlesq34/pointnet), [dgcnn](https://github.com/charlesq34/pointnet) and [PU-GAN](https://github.com/charlesq34/pointnet).
