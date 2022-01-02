---
layout: publication
title: "Deep CSI Learning for Gait Biometric Sensing and Recognition"
author: "Kalvik Jakkala"
venue: BalkanCom
year: 2019
date: 2019-01-01 00:00:00
---

<center>
Paper: <a href="https://arxiv.org/pdf/1902.02300.pdf"><i class="fa fa-file-text" aria-hidden="true"></i></a>
&nbsp;&nbsp;
Code: <a href="https://github.com/kdkalvik/WiFi-user-recognition"><i class="fa fa-github" aria-hidden="true"></i></a>
</center>

## Abstract
Gait is a person's natural walking style and a complex biological process that is unique to each person. Recently, the channel state information (CSI) of WiFi devices have been exploited to capture human gait biometrics for user identification. However, the performance of existing CSI-based gait identification systems is far from satisfactory. They can only achieve limited identification accuracy (maximum 93%) only for a very small group of people (i.e., between 2 to 10). To address such challenge, an end-to-end deep CSI learning system is developed, which exploits deep neural networks to automatically learn the salient gait features in CSI data that are discriminative enough to distinguish different people Firstly, the raw CSI data are sanitized through window-based denoising, mean centering and normalization. The sanitized data is then passed to a residual deep convolutional neural network (DCNN), which automatically extracts the hierarchical features of gait-signatures embedded in the CSI data. Finally, a softmax classifier utilizes the extracted features to make the final prediction about the identity of the user. In a typical indoor environment, a top-1 accuracy of 97.12±1.13% is achieved for a dataset of 30 people.
