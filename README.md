
Welcome to the Virtual Try-On GitHub repository, an open-source codebase designed to empower developers, researchers, and fashion-tech enthusiasts in creating innovative, AI-driven virtual try-on solutions for the fashion industry.

This repository provides comprehensive tools, models, and sample code to help fashion retailers and developers integrate virtual try-on technology seamlessly. By leveraging computer vision and deep learning, users can simulate realistic clothing fits on virtual avatars or real images, creating a dynamic and engaging experience for online shoppers.

# Key Features:
1. Realistic Garment Fitting: Accurate simulation of clothing on different body shapes and sizes.
2. Cloth Warping and Draping: Advanced algorithms to render clothing that adapts naturally to body poses and movements.
3. Easy Integration: Modular and scalable architecture designed to plug into existing e-commerce platforms or custom applications.
4. Customizable for Various Use Cases: Suitable for a range of fashion items, from tops and bottoms to accessories.

Dive into the code, contribute, and help shape the future of virtual fashion!

![Myra (1)](https://github.com/user-attachments/assets/1cd3a3cb-4b04-435c-9088-f64f1d6d5c9a)


# Virtual Try-On with Garment-Pose Keypoints Guided Inpainting

This codes repository provides the pytorch implementation of the KGI virtual try-on method proposed in ICCV23 paper [Virtual Try-On with Garment-Pose Keypoints Guided Inpainting.](https://openaccess.thecvf.com/content/ICCV2023/papers/Li_Virtual_Try-On_with_Pose-Garment_Keypoints_Guided_Inpainting_ICCV_2023_paper.pdf)


## Data Preparation
1. The VITON-HD dataset could be downloaded from [VITON-HD](https://www.dropbox.com/s/10bfat0kg4si1bu/zalando-hd-resized.zip?dl=0) .
   
2. In addition to above content, some other preprocessed conditions are in use in KGI. The content are generated with the data preprocessing codes [WIP]. 
   
## Model Training
The model training of the Myra Virtual Try On consists of three steps: Training of the Keypoints Generator, Training of the Parse Generator, Training of the Semantic Conditioned Inpainting Model.

### Keypoints Generator
* The Keypoints Generator is trained with the following scripts:
   ```
   cd codes_kg
   python3 train_kg.py
   ```
 ### Parse Generator
* The Parse Generator is trained with the following scripts:
  ```
  cd codes_pg
  python3 train_pg.py
  ```
 ### Semantic Conditioned Inpainting Model
* The Semantic Conditioned Inpainting Model is trained with the following scripts:
  ```
  cd codes_sdm
  python3 train_sdm.py
  ```

## Demo with Pretrained Model
   With the pretrained models, the final try-on results and the visualizations of the intermediate results could be generated with the following demo scripts:
   ```
   python3 generate_demo.py
   ```
  ## Acknowledgement and Citations
* The implementation of Keypoints Generator is based on codes repo [SemGCN](https://github.com/garyzhao/SemGCN).
* The implementation of Semantic Conditioned Inpainting Model is based on [semantic-diffusion-model](https://github.com/WeilunWang/semantic-diffusion-model) and [RePaint](https://github.com/andreas128/RePaint).
* The implementation of datasets and dataloader is based on codes repo [HR-VITON](https://github.com/sangyun884/HR-VITON).
* If you find our work is useful, please use the following citation:
  ```
  @InProceedings{Li_2023_ICCV,
    author    = {Li, Zhi and Wei, Pengfei and Yin, Xiang and Ma, Zejun and Kot, Alex C.},
    title     = {Virtual Try-On with Pose-Garment Keypoints Guided Inpainting},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {22788-22797}
  }
  ```
