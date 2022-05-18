---
layout: post
title: "Tutorial on sparse Gaussian processes (work in progress)"
author: "Kalvik Jakkala"
categories: journal
tags: [documentation, sample]
abstract: "My notes on the derivation of the variational sparse Gaussian process approach [Titsias 2009]."
---

Gaussian processes are one of the most, if not the most, mathematically beautiful and elegant machine learning methods in history. We can use them for classification, regression, or generative problems. Also, the best part, they are probabilistic, so we can quantify the uncertainty in our predictions and have a lower risk of overfitting. 

Gaussian processes are composed of latent random variables $\mathbf{f}$ each with a corresponding input point in $\mathbf{X}$. In sparse Gaussian processes, we augment the Gaussian process with additional data points $\mathbf{X}_u$ called inducing points, each with a corresponding latent variable $\mathbf{u}$. The inducing points have the following prior distribution, which is the same as the prior for the original latent variables $\mathbf{f}$.

$$
p(\mathbf{u} | \mathbf{X}_u) = \mathcal{N}(\mathbf{u} |0, \mathbf{K}_{uu}) \tag{1}
$$

The joint distribution over the latent variables $\mathbf{f}$ and $\mathbf{u}$, which correspond to the training points $\mathbf{X}$, and inducing points $\mathbf{X}_u$ respectively, is given by the following

$$
p(\mathbf{f}, \mathbf{u} | \mathbf{X}, \mathbf{X}_u) = \mathcal{N}\left(
    \begin{bmatrix}
        \mathbf{f} \\
        \mathbf{u}
    \end{bmatrix} |
    \mathbf{0}, 
    \begin{bmatrix}
        \mathbf{K}_{ff} & \mathbf{K}_{fu}\\
        \mathbf{K}_{uf} & \mathbf{K}_{uu}
    \end{bmatrix}
\right) \tag{2}
$$

From the above joint distribution, we can compute the conditional distribution over $\mathbf{f}$ given $\mathbf{u}$, and the training data points $\mathbf{X}$ as follows 

$$
p(\mathbf{f} | \mathbf{u}, \mathbf{X}) = \mathcal{N}(\mathbf{f} | \mathbf{a}, \tilde{\mathbf{K}}) \tag{3} \\
\mathbf{a} = \mathbf{K}_{fu}\mathbf{K}_{uu}^{-1}\mathbf{u} \\ 
\tilde{\mathbf{K}} = \mathbf{K}_{ff} - \mathbf{K}_{fu} \mathbf{K}_{uu}^{-1} \mathbf{K}_{uf} \\
$$

The above conditional follows from the standard Gaussian conditioning operation used in [Gaussian processes](https://kdkalvik.github.io/gp-tutorial). I will drop the explicit conditioning on the inputs $\mathbf{X}$ from here on to keep the notation clean.

Alright, so why did we introduce the inducing points, and how will that help us reduce the computation cost of Gaussian processes? 

First, assuming we know the distribution of $\mathbf{u}$, our above formulation allows us to marginalize the inducing variables $\mathbf{u}$ and recover the original distribution over only the variables $\mathbf{f}$, which is used for exact inference in Gaussian processes.

$$
\begin{aligned}
p(\mathbf{f}) &= \int p(\mathbf{f}|\mathbf{u}) p(\mathbf{u}) d\mathbf{u} \\
&= \int \underbrace{\mathcal{N}(\mathbf{f} | \mathbf{K}_{fu}\mathbf{K}_{uu}^{-1}\mathbf{u}, \mathbf{K}_{ff} - \mathbf{K}_{fu} \mathbf{K}_{uu}^{-1} \mathbf{K}_{uf})}_{\text{Equation (3)}} \times \underbrace{\mathcal{N}(\mathbf{u} |0, \mathbf{K}_{uu})}_{\text{Equation (1)}} d\mathbf{u}\\
&= \underbrace{\mathcal{N}(\mathbf{f}|\mathbf{K}_{fu}\mathbf{K}_{uu}^{-1}\mathbf{0}, \mathbf{K}_{ff} - \mathbf{K}_{fu} \mathbf{K}_{uu}^{-1} \mathbf{K}_{uf} + \mathbf{K}_{fu} \mathbf{K}_{uu}^{-1} \mathbf{K}_{uf})}_{\text{Equation (A.1) in the appendix}} \\
&= \mathcal{N}(\mathbf{f}| \mathbf{0}, \mathbf{K}_{ff}) 
\end{aligned}
$$

And second, we can factorize the marginal distribution of the training set labels as follows 

$$
p(\mathbf{y}, \mathbf{f}, \mathbf{u}) = p(\mathbf{y}|\mathbf{f})p(\mathbf{f}|\mathbf{u})p(\mathbf{u})
$$


# Appendix

### Gaussian margnial and condititonal distributions

If we have a marginal Gaussian distribution for $\mathbf{u}$ and a conditional Gaussian distribution for $\mathbf{f}$ given $\mathbf{u}$ as shown below, the marginal distribution of $\mathbf{f}$ is given as follows 

$$
\begin{aligned}
p(\mathbf{u}) &= \mathcal{N}(\mathbf{u} | \mathbf{\mu}_u, \mathbf{\Sigma}_u) \\
p(\mathbf{f|u}) &= \mathcal{N}(\mathbf{f} | \mathbf{Mu+m}, \mathbf{\Sigma}_f) \\
p(\mathbf{f}) &= \mathcal{N}(\mathbf{f} | \mathbf{M}\mathbf{\mu}_u + \mathbf{m}, \mathbf{\Sigma}_f + \mathbf{M}\mathbf{\Sigma}_u \mathbf{M}^\top)
\end{aligned} \tag{A.1}
$$

### Woodbury matrix identity
Given an invertable matrix $\mathbf{A}$ of size $n \times n$, matrices $\mathbf{U}$ and $\mathbf{V}$ of size $n \times m$, and an invertable matrix $\mathbf{W}$ of size $m \times m$

$$
(\mathbf{A} + \mathbf{U}\mathbf{W}\mathbf{V}^\top)^{-1} = \mathbf{A}^{-1} - \mathbf{A}^{-1}\mathbf{U}(\mathbf{W}^{-1} + \mathbf{V}^\top \mathbf{A}^{-1} \mathbf{U})^{-1} \mathbf{V}^\top \mathbf{A}^{-1} \tag{A.2}
$$

### Matrix determinant lemma

$$
|\mathbf{A} + \mathbf{U} \mathbf{W} \mathbf{V}^\top| = |\mathbf{W}^{-1} + \mathbf{V}^\top \mathbf{A}^{-1} \mathbf{U}| |\mathbf{W}| |\mathbf{A}| \tag{A.3}
$$

where $\|.\|$ denotes the determinant of a matrix.
