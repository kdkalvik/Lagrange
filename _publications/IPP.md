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

\
&nbsp;

<p float="left">
  <img src="{{ site.github.url }}/assets/img/ICRA2024_logo.png" width="40%" style="vertical-align:middle"/>
</p>

\
&nbsp;

## Abstract
This paper addresses multi-robot informative path planning (IPP) for environmental monitoring.  The problem involves determining informative regions in the environment that should be visited by robots in order to gather the most amount of information about the environment. We propose an efficient sparse Gaussian process-based approach that uses gradient descent to optimize paths in continuous environments. Our approach efficiently scales to both spatially and spatio-temporally correlated environments. Moreover, our approach can simultaneously optimize the informative paths while accounting for routing constraints, such as a distance budget and limits on the robot's velocity and acceleration. Our approach can be used for IPP with both discrete and continuous sensing robots, with point and non-point field-of-view sensing shapes, and for both single and multi-robot IPP. We demonstrate that the proposed approach is fast and accurate on real-world data.

<style>
.yt {
  position: relative;
  display: block;
  width: 90%; /* width of iframe wrapper */
  height: 0;
  margin: auto;
  padding: 0% 0% 56.25%; /* 16:9 ratio */
  overflow: hidden;
}
.yt iframe {
  position: absolute;
  top: 0; bottom: 0; left: 0;
  width: 100%;
  height: 100%;
  border: 0;
}
</style>

<div class="yt">
  <iframe width="560" height="315" src="https://www.youtube.com/embed/6PmpuqfQUv8?si=xfz6pa4n48optZ1P" allowfullscreen></iframe>
</div>
