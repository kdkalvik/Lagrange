---
layout: publication
title: "Multi-Robot Informative Path Planning from Regression with Sparse Gaussian Processes"
author: "Kalvik Jakkala"
venue: ICRA
year: 2024
date: 2024-01-29 00:00:00
---

<p>
<center>
  <a href="https://itskalvik.github.io/cv.html"
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
Appendix: <a href="https://nbviewer.org/github/itskalvik/itskalvik.github.io/blob/gh-pages/assets/SGP_IPP_APP.pdf"><span style="color: #4285F4;"><i class="fa fa-file-text"></i></span></a>
</center>

## Abstract
This paper addresses multi-robot informative path planning (IPP) for environmental monitoring.  The problem involves determining informative regions in the environment that should be visited by robots in order to gather the most amount of information about the environment. We propose an efficient sparse Gaussian process-based approach that uses gradient descent to optimize paths in continuous environments. Our approach efficiently scales to both spatially and spatio-temporally correlated environments. Moreover, our approach can simultaneously optimize the informative paths while accounting for routing constraints, such as a distance budget and limits on the robot's velocity and acceleration. Our approach can be used for IPP with both discrete and continuous sensing robots, with point and non-point field-of-view sensing shapes, and for both single and multi-robot IPP. We demonstrate that the proposed approach is fast and accurate on real-world data.
