# Geometry Enhanced Molecular Representation Learning for Property Prediction

# Background
In this study, we developed a highly accurate and interpretable human DILI prediction model named GeoDILI. An overview of the proposed model is shown in following figure:  

![image-20230411182436222](C:\Users\qjy41\AppData\Roaming\Typora\typora-user-images\image-20230411182436222.png)

The GeoDILI model used a pre-trained 3D spatial structure-based GNN to extract molecular representations, followed by a residual neural network to make an accurate DILI prediction. The gradient information from the final graph convolutional layer of GNN was utilized to obtain atom-based weights, which enabled the identification of dominant substructures that significantly contributed to DILI prediction. We evaluated the performance of GeoDILI by comparing it with the SOTA DILI prediction tools, popular GNN models, as well as conventional Deep Neural Networks (DNN) and ML models, confirming its effectiveness in predicting DILI. In addition, we applied our model to three different human DILI datasets from various sources, namely DILIrank, DILIst, and a dataset recently collected by Yan et al.. Results showed performance differences across datasets and suggested that a smaller, high-quality dataset DILIrank may lead to better results. Finally, we applied the dominant substructure inference method to analyze the entire DILIrank dataset and identified seven significant SAs with both high precision and potential mechanisms. 


# Installation guide
## Prerequisites

* OS support: Windows, Linux
* Python version: 3.6, 3.7, 3.8

## Dependencies

| name         | version |
| ------------ | ---- |
| numpy        | - |
| pandas       | - |
| networkx     | - |
| paddlepaddle | \>=2.0.0 |
| pgl          | \>=2.1.5 |
| rdkit-pypi   | - |
| sklearn      | - |

# Usage

To train a model with an existing dataset:

    $ python main.py --dataset_name dilirank --task train --split_type random
    $ python main.py --dataset_name rega --task train --split_type random
    $ python main.py --dataset_name diliset --task train --split_type random
    $ python main.py --dataset_name bbbp --task train --split_type random

To test with an existing model:

    $ python main.py --dataset_name dilirank --task test --split_type random
    $ python main.py --dataset_name rega --task test --split_type random

## Result

|     Dataset      |    AUC    |    ACC    |    MCC    |
| :--------------: | :-------: | :-------: | :-------: |
|     DILIrank     | **0.905** | **0.875** | **0.732** |
|      DILIst      | **0.856** | **0.786** | **0.553** |
| Yan et al (rega) | **0.842** | **0.773** | **0.549** |

|                           DILIrank                           |                     Yan et al(rega)                     |
| :----------------------------------------------------------: | :-----------------------------------------------------: |
| ![dilirank](C:\Users\qjy41\OneDrive\桌面\药物设计\dilirank.png) | ![raga](C:\Users\qjy41\OneDrive\桌面\药物设计\raga.png) |



## Citation

If you use the code or data in this package, please cite:

```bibtex

```
