# multi_plant_id
Plant species identification using deep learning methods to address the PlantCLEF 2025 Kaggle challenge.

# Abstract

This project aims at the PlantCLEF2025 Kaggle/LifeCLEF challenge on multi-species plant identification in unlabeled vegetation quadrat images. We describe a pipeline for zero-shot multi-species plant identification using a fine-tuned pretrained Vision Transformer (ViT) model combined with methods of high-resolution image tiling, false positive reduction, and prediction aggregation.


### Ablations

We did a lot of study on a baseline resnet50 model studying various ablations in order to improve performance, the full implementation can be seen in the `resnet50` file directory.

The code for the ablation studies on the ViT model can be seen in the `main` folder and can be run using the slurm script that is included.

### Final Model 
