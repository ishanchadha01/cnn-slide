# CS 8803 SMR Final Project
## Team 11 -- Ishan Chadha, Raj Janardhan, Manoj Niverthi

This repository contains an implementation of our final project for CS 8803, where we try and extend methods proposed in ![SLIDE](https://arxiv.org/pdf/1903.03129.pdf) to convolutional neural networks.

## Project Description
With the rapid rise and proliferation of different forms of visual data, Convolutional Neural Networks (CNNs) have emerged as a powerful deep learning technique applicable across a wide range of tasks, including image classification, object detection, and image segmentation. However, these networks are typically expansive concerning the number of parameters they possess and require expensive hardware, including GPUs and TPUs to train. The subject of CNN-SLIDE is to accelerate the training time of CNN models in a hardware-agnostic manner. Specifically, this work leverages two previously explored techniques -- (1) tensor sketching and (2) locality-sensitive hashing -- that have been used to construct a variety of network architectures, such as multi-layer perceptrons, in isolation. This work proposes a unique fusion of these techniques to reduce model training time further. The sketching component focuses on reducing the dimensionality of convolutional kernel filters, while the LSH component lowers the number of filters that are applied. Experimental results showcase CNN-SLIDE's effectiveness in significantly reducing model runtimes.

## File Structure
- `conv` - contains implementations of convolutional primitives (e.g. SketchConv, ALSHSketchConv) that are used in downstream CNN tasks.
- `lsh` - contains implementations of primitives that the locality-sensitive hashing approach relies on (e.g. hyperplane LSH, hash tables)
- `test` - contains tests for the aforementioned directories
- `Ablations.ipynb` - notebook containing examples of conducted experiments

## Environment Setup
We provision the environment for this code using `conda`. To set up the environment, run `conda env create -f environment.yml`

## Running the Code
We can run test `SketchConv`, one of the main modules that were implemented in this work, by running `test/test_sketch_conv.py`. This validates the inputs/outputs of the sketch convolution behave identically (in terms of expected input/output dimension) to a regular convolution.

To run through a sampling of the experiments that were done, we can run the `Ablations.ipynb` notebook. This notebook goes through the process of:
- Loading in data modules
- Defining sample networks on which the tensor sketching and locality-sensitive hashing is tested
- Training the defined models on this data
