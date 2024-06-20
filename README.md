# Class Agonistic Semantic Edges

## Overview
This project focuses on developing a deep learning model for detecting class-agnostic semantic edges using the PASCAL VOC dataset. Semantic edges allow us to separate various semantic classes within images. However, these edges represent a very small fraction of an image's data, leading to a significant imbalance. Our main goal is to refine the model's ability to detect these sparse features.

To handle the issue of data imbalance in edge detection, our approach incorporates a weighted loss function (weighted binary cross-entropy) where errors in predicting edge pixels are given more significance over errors for non-edge pixels. This strategy ensures that our model prioritizes accurate edge detection during training, consequently improving the model's precision in detecting these features. The architectural backbone of our model is based on VGGNet, known for its feature extraction capabilities in image processing tasks. We modify the network's decision head to use sigmoid (instead of softmax), which outputs a binary mask where each pixel value indicates the likelihood of being an edge.

Moreover, the model integrates the principles of Holistically Nested Edge Detection (HED), an architecture that builds upon traditional deep convolutional networks. HED performs deep supervision at multiple scales of the image simultaneously, effectively improving edge detection accuracy. HED comprises a single-stream deep network with multiple side outputs at each convolution layer to enforce rich hierarchical representation. This representation resolves the challenges caused by the high variability and fine granularity of edges within images.

The training process focuses on optimizing the average and fused outputs from the model to determine which approach results in higher precision and recall for predicting edges. Averaging combines side outputs produced at different stages within the network into a mean to stabilize predictions by reducing variance and noise in individual outputs. Conversely, fusing combines them through a learned fusion layer where each side output contributes according to its predictive power. Fusion allows the model to detect complex patterns and subtle variations in edge characteristics that may not be captured using average pooling.

We delve deeper into the concept of weighted binary cross-entropy and how it helps solve the problem of imbalanced data in the notebook. The notebook is designed to run in a GPU-accelerated environment, specifically utilizing a T4 GPU. It is recommended to use Google Colab if you don't have access to a GPU and wish to view the project. The primary programming language used in this project is Python 3.

## Requirements

To run this project, you will need the following:

- **Python 3**: Ensure you have Python 3 installed on your system.
- **Jupyter Notebook**: You can install Jupyter Notebook using `pip install notebook`.
- **GPU Support**: A GPU is recommended for optimal performance. This project has been configured for a T4 GPU.

## Setup Instructions

1. **Clone the repository**:
    ```bash
    git clone git@github.com:IbrahimKashif1/Class-Agonistic-Semantic-Edges.git
    cd Class-Agonistic-Semantic-Edges
    ```

2. **Run the Jupyter Notebook**:
    ```bash
    jupyter notebook main.ipynb
    ```

## Usage

1. **Open the notebook**:
    Launch Jupyter Notebook and open `main.ipynb`.

2. **Execute cells**:
    Follow the sequence of cells to execute the code. Each cell should be run in order to ensure the notebook functions correctly.

3. **GPU Utilization**:
    The notebook is optimized for GPU usage. Make sure your environment supports GPU acceleration to take full advantage of the performance improvements.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
