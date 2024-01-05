# IU INTERNATIONAL UNIVERSITY OF APPLIED SCIENCES

### Project: Artificial Intelligence (DLBDSEAIS02)
Utilizing TensorFlow as the primary framework, the project focuses on emotion detection from facial expressions using a Convolutional Neural Network (CNN). TensorFlow's deployment capabilities, including TensorFlow Serving and TensorFlow Lite, ensure seamless transition to production. The workflow involves a pre-processing pipeline, transforming images to a 48x48 dimension from the mma-facial-expression dataset. The CNN architecture comprises convolution layers, max pooling, ReLU activation, dense layers, and a fully connected layer with Softmax activation. Model testing involves metrics like accuracy, recall, and precision, with the F1-score for multi-classification. Deployment is done via FastAPI and Docker, enabling local or cloud-based execution.

## Purpose
Build an AI tool for the marketing department to analyze emotional responses to company ads using facial expressions in images. Develop a system to categorize emotions (e.g., joy, anger, fear) from an image dataset. Choose relevant technologies for components and communication. The goal is to identify at least three emotional states through facial expressions.

# Development Planning - UML Schemas
## UML Diagram
Diagram - Design of Convolutional Neural Network to classify emotions and serve using API
![Design of Convolutional Neural Network to classify emotions and serve using API](/images/UML.jpg)

## Dataset Overview
Source
Folders, Image Data, Count/Size, Balanced or Unbalanced?

# Model Development
..

## Model Metrics and Hyper Parameters
Accurary, Recall, Precision ...
optimizer, loss function, etc.

<div style="display: flex; flex-direction: column; align-items: flex-start; background-color: #000000;">
    <img src="/models/training/diagram/model_1_concept_diagram.png" alt="Model 1 Image" width="200">
    <img src="/models/training/diagram/model_2_best_diagram.png" alt="Model 2 Image" width="200">
    <img src="/models/training/diagram/model_3_best_nodropout_diagram.png" alt="Model 3 Image" width="200">
    <img src="/models/training/diagram/model_4_best_nodroupout_nobatchn_diagram.png" alt="Model 4 Image" width="200">
</div>

Results from iteratively training 4 different models that can be found in /models:
- model_1_concept - Model as initially described in the concept phase
- model_2_best - Model that was found in experimenting with different architectures and reading literature
- model_3_best_nodropout - The result of discarding dropout from model_2
- model_4_best_nodroupout_nobatchn - The result of discarding dropout and batch normalization from model_2

Using three channels (RGB) for image input the following results were obtained:
![RGB Model Training Results](/images/training_results_rgb.jpg)

Using one channel (GRAYSCALE) for image input the following results were obtained:
![GRAYSCALE Model Training Results](/images/training_results_grayscale.jpg)

As it can be seen the RGB training results are slightly better eventough a large part of the training data is grayscale images. The model seems to learn aspects from the RGB data that are not available with only one channel.

# How to get started
## Dependencies
...
...

## Installation instructions
...

## API usage
...

# Reflection
What I learned ...
What challenges occured ...
Installing CUDA drivers for NVIDIA Graphic Card
What recommendations are suggested ...
What could be improved ...

# Conclusion
Did we achieve the goal etc and how?

# Disclaimer
The developed application is licensed under the GNU General Public License.