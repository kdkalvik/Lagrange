---
layout: post
title: "NeuralWave: Gait-Based User Identification Through Commodity WiFi and Deep Learning"
author: "Kalvik Jakkala"
venue: IECON
year: 2018
date: 2018-01-01 00:00:00
---

<center>See paper: <a href="https://ieeexplore.ieee.org/document/8591820"><i class="fa fa-file-text" aria-hidden="true"></i></a></center>

## Abstract
This paper proposes NeuralWave, an intelligent and non-intrusive user identification system based on human gait biometrics extracted from WiFi signals. In particular, the channel state information (CSI)measurements are first collected from commodity WiFi devices. Then, a collection of data preprocessing schemes are applied to sanitize and calibrate the noisy and erroneous CSI data samples to manifest and augment the gait-induced radio-frequency (RF)signatures. Next, a 23-layer deep convolutional neural network, namely RadioNet, is developed to automatically learn the salient features from the preprocessed CSI data samples. The extracted features constitute a latent representation for the gait biometric that is discriminative enough to distinguish one person from another. Using the latent biometric representation, a softmax multi-class classifier is adopted to achieve accurate user identification. Extensive experiments in a typical indoor environment are conducted to show the effectiveness of our system. In particular, NeuralWave can achieve 87.76 ± 2.14% user identification accuracy for a group of 24 people. To the best of our knowledge, NeuralWave is the first in the literature to exploit deep learning for feature extraction and classification of physiological and behavioral gait biometrics embedded in CSI signals from commodity WiFi.
