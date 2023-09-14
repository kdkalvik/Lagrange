---
layout: publication
title: "Multi-Robot Informative Path Planning from Regression with Sparse Gaussian Processes"
author: "Kalvik Jakkala"
venue: Under Review
year: 2023
date: 2023-08-29 00:00:00
---

<p>
<center>
  <a href="https://webpages.uncc.edu/kjakkala"
   style="text-decoration: none"><b style="color:Black">Kalvik Jakkala</b></a>
   &nbsp;&nbsp;
  &nbsp;&nbsp;
  <a href="https://webpages.uncc.edu/sakella/"
   style="text-decoration: none"><b style="color:Black">Srinivas Akella</b></a>
</center>
</p>

<center>
Paper: <a href="https://arxiv.org/pdf/2309.07050.pdf"><span style="color: #4285F4;"><i class="fa fa-file-text"></i></span></a>
&nbsp;&nbsp;
</center>

## Abstract
This paper addresses multi-robot informative path planning (IPP) for environmental monitoring. The problem involves determining informative regions from which to collect data and estimating the current state of the environment. While earlier IPP approaches predominantly utilized discrete optimization techniques, we propose an efficient sparse Gaussian process-based approach that uses gradient descent to select informative sensing locations in continuous environments. Our approach efficiently scales to both spatially and spatio-temporally correlated environments. Moreover, our approach can simultaneously optimize the informative sensing route locations while accounting for routing constraints, such as a distance budget and limits on the robot's velocity and acceleration. Additionally, we can even leverage past data for path planning. Our approach can be used for IPP with both discrete and continuous sensing robots, with point and non-point field-of-view shapes, and for multi-robot IPP. The proposed approach is demonstrated to be fast and accurate on real-world data.
