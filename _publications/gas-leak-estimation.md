---
layout: publication
title: "Probabilistic Gas Leak Rate Estimation using Submodular Function Maximization with Routing Constraints"
author: "Kalvik Jakkala"
venue: RA-L, ICRA
year: 2022
date: 2022-01-01 00:00:00
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
Paper: <a href="https://nbviewer.org/github/UNCCharlotte-CS-Robotics/Gas-Leak-Estimation/blob/main/paper.pdf"><span style="color: #4285F4;"><i class="fa fa-file-text"></i></span></a>
&nbsp;&nbsp;
Derivation: <a href="https://nbviewer.org/github/UNCCharlotte-CS-Robotics/Gas-Leak-Estimation/blob/main/Supplemental.pdf"><span style="color: #4285F4;"><i class="fa fa-file-text"></i></span></a>
&nbsp;&nbsp;
Code: <a href="https://github.com/UNCCharlotte-CS-Robotics/Gas-Leak-Estimation"><span style="color: #4285F4;"><i class="fa fa-github"></i></span></a>
&nbsp;&nbsp;
Slides: <a href="https://nbviewer.org/github/UNCCharlotte-CS-Robotics/Gas-Leak-Estimation/blob/main/presentation_slides.pdf"><span style="color: #4285F4;"><i class="fa-file-powerpoint-o"></i></span></a>
</center>

<p float="left">
  <img src="{{ site.github.url }}/assets/img/methane-leak-rate-estimation/ieee_ras_logo.png" width="40%" style="vertical-align:middle"/>
  &nbsp;&nbsp;
  <img src="{{ site.github.url }}/assets/img/wastewater/ieee_icra_logo.png" width="55%" style="vertical-align:middle"/>
</p>

## Abstract
Methane, a harmful greenhouse gas, is prone to leak during extraction from oil wells. Therefore, we must monitor oil well leak rates to keep such emissions in check. However, most currently available approaches incur significant computational costs to generate informative data collection walks for mobile sensors and estimate leak rates. As such, they do not scale to large oil fields and are infeasible for real-time applications. We address these problems by deriving an efficient analytical approach to compute the leak rate distribution and Expected Entropy Reduction (EER) metric used for walk generation. Moreover, a faster variant of a submodular function maximization algorithm is introduced, along with a generalization of the algorithm to find informative data collection walks with arc routing constraints. Our simulation experiments demonstrate the approach's validity and substantial computational gains.

![]({{ site.github.url }}/assets/img/methane-leak-rate-estimation/oil_field.jpg)

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

<div class="yt"><iframe width="560" height="315" src="https://www.youtube.com/embed/7S0bLzwRrlE" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe></div>
