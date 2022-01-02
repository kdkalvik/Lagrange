---
layout: publication
title: "Probabilistic Methane Leak Rate Estimation using Submodular Function Maximization with Routing Constraints"
author: "Kalvik Jakkala"
venue: RA-L, Under Review
year: 2021
date: 2021-01-01 00:00:00
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
Paper: <a href="https://kdkalvik.github.io/methane-leak-rate-estimation/paper.pdf"><i class="fa fa-file-text" aria-hidden="true"></i></a>
&nbsp;&nbsp;
Derivation: <a href="https://kdkalvik.github.io/methane-leak-rate-estimation/supplemental.pdf"><i class="fa fa-file-text" aria-hidden="true"></i></a>
&nbsp;&nbsp;
Code: <a href="https://github.com/kdkalvik/methane-leak-rate-estimation"><i class="fa fa-github" aria-hidden="true"></i></a>
</center>

<p float="left">
  <img src="{{ site.github.url }}/assets/img/methane-leak-rate-estimation/uncc_logo.png" width="300" style="vertical-align:middle"/>
  &nbsp;&nbsp;
  <img src="{{ site.github.url }}/assets/img/methane-leak-rate-estimation/ieee_ras_logo.png" width="250" style="vertical-align:middle"/>
</p>

## Abstract
Methane, a harmful greenhouse gas, is prone to leak during extraction from oil wells. Therefore, we must monitor oil well leak rates to keep such emissions in check. However, most currently available approaches incur significant computational costs to generate informative data collection walks for mobile sensors and estimate leak rates. As such, they do not scale to large oil fields and are infeasible for real-time applications. We address these problems by deriving an efficient analytical approach to compute the leak rate distribution and Expected Entropy Reduction (EER) metric used for walk generation. Moreover, a faster variant of a submodular function maximization algorithm is introduced, along with a generalization of the algorithm to find informative data collection walks with arc routing constraints. Our simulation experiments demonstrate the approach's validity and substantial computational gains.

![]({{ site.github.url }}/assets/img/methane-leak-rate-estimation/oil_field.jpg)
