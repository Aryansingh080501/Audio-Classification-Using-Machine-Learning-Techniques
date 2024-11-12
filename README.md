# Audio Classification Using Machine Learning Techniques

## Project Overview
The objective of this project is to explore and implement various machine learning techniques for the accurate classification of audio data. Specifically, the project focuses on classifying spoken digits (0-9) from audio files using the following methods:

- Singular Value Decomposition (SVD) / Principal Component Analysis (PCA)
- Support Vector Machines (SVM)
- Decision Trees
- k-Nearest Neighbors (k-NN)

The audio files are in `.wav` format, and different machine learning models are applied to classify these spoken digits.

## Theoretical Background

### 1. SVD/PCA for Dimensionality Reduction
SVD (Singular Value Decomposition) and PCA (Principal Component Analysis) are dimensionality reduction techniques commonly used in machine learning. They are used to reduce the complexity of data while preserving the important features.

- **SVD**: A matrix factorization technique that decomposes a matrix into three orthogonal matrices. SVD is used in this project to reduce the dimensionality of the audio data by retaining the most significant features and discarding noise.
- **PCA**: PCA identifies the principal components that capture the most variance in the data. By projecting the data onto these components, PCA reduces the dimensionality of the audio files while retaining as much variance as possible.

### 2. SVM (Support Vector Machines)
SVM is a supervised learning algorithm used for classification tasks. It finds the hyperplane that best separates the data into distinct classes. In this project, SVM is used to classify the spoken digits in the audio files. The input data is mapped to a high-dimensional feature space, and SVM iterates through the data to learn how to separate the digits.

### 3. Decision Trees
Decision Trees are non-parametric supervised learning algorithms used for classification and regression tasks. They partition the feature space into distinct regions based on feature values, creating a tree-like structure. In this project, decision trees are used to classify the audio files by splitting the data based on the features extracted from the `.wav` files.

### 4. k-NN (k-Nearest Neighbor)
k-NN is a supervised learning algorithm that classifies data based on the majority class of the k-nearest neighbors. This method does not make assumptions about the data distribution and is simple to implement. In this project, k-NN is used to classify the audio files by calculating the distance between the features of the `.wav` files and their k-nearest neighbors in the feature space.

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/your-username/audio-classification.git
   cd audio-classification
