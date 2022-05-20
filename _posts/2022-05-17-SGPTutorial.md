---
layout: post
title: "Tutorial on sparse Gaussian processes (work in progress)"
author: "Kalvik Jakkala"
categories: journal
tags: [documentation, sample]
abstract: "My notes on the derivation of the variational sparse Gaussian process approach [Titsias 2009]."
---

# Gaussian processes

[Gaussian processes](https://kdkalvik.github.io/gp-tutorial) are one of the most, if not the most, mathematically beautiful and elegant machine learning methods in history. We can use them for classification, regression, or generative problems. Also, the best part, they are probabilistic, so we can quantify the uncertainty in our predictions and have a lower risk of overfitting. 

Given a regression task's training set $$\mathcal{D} = \{(\mathbf{x}_i, y_i), i = 1,...,n\}$$ with $n$ data samples consisting of inputs $$\mathbf{x}_i \in \mathbb{R}^d$$ and noisy outputs $y_i \in \mathbb{R}$, we can use Gaussian processes to predict the noise free outputs $$f_*$$ (or noisy $$y_*$$) at test locations $$\mathbf{x}_*$$. The approach assumes that the relationship between the inputs  $$\mathbf{x}_i$$ and outputs  $$y_i$$ is given by

$$
y_i = f(\mathbf{x}_i) + \epsilon_i \quad \quad \text{where} \ \ \epsilon_i \sim \mathcal{N}(0, \sigma^2_{\text{noise}}) \\
$$

Here $$\sigma^2_{\text{noise}}$$ is the variance of the independent additive Gaussian noise in the observed outputs $$y_i$$. The latent function $$f(\mathbf{x})$$ models the noise free function of interest that explains the regression dataset. 

Gaussian processes (GP) model datasets formulated as shown above by assuming a GP prior over the space of functions that could be used to explain the dataset, i.e., assumes a priori that the function values behave according to 

$$
p(\mathbf{f} | \mathbf{X}) = \mathcal{N}(0, \mathbf{K}) \\
$$

where $$\mathbf{f} = [f_1, f_2,...,f_n]^\top$$ is a vector of latent function values, $$f_i = f(\mathbf{x_i})$$, $$\mathbf{X} = [\mathbf{x}_1, \mathbf{x}_2,...,\mathbf{x}_n]^\top$$ is a vector (or matrix) of inputs. And $\mathbf{K} \in \mathbb{R}^{n \times n}$ is a covariance matrix, whose entries $$\mathbf{K}_{ij}$$ are given by the kernel function $$k(x_i, x_j)$$. 

GPs learn the latent function that explains the training data and use the kernel function to index and order the inputs $$\mathbf{x_i}$$ so that points closer to each other  (i.e., have high covariance value from the kernel function) have similar labels and vice versa. Inference in GPs to get the output predictions $$\mathbf{y}$$ for the training set input samples $$\mathbf{X}$$ entails marginalizing the latent function values $$\mathbf{f}$$  


$$
p(\mathbf{y, f} | \mathbf{X}) = p(\mathbf{y} | \mathbf{f}) p(\mathbf{f} | \mathbf{X}) \\
p(\mathbf{y} | \mathbf{X}) = \int p(\mathbf{y, f} | \mathbf{X}) d\mathbf{f}
$$

I will drop the explicit conditioning on the inputs $\mathbf{X}$ from here on to reduce the notational complexity and assume that the corresponding inputs are always in the conditioning set.

Inference on test points $$\mathbf{X_*}$$ to get the noise free predictions $$\mathbf{f}_*$$ (or noisy $$\mathbf{y}_*$$) can be done by considering the joint distribution over the training and test latent values, $$\mathbf{f}$$ and $$\mathbf{f}_*$$, and using Gaussian conditioning to marginalizing the training set latent variables as shown below

$$
\begin{aligned}
p(\mathbf{f}, \mathbf{f}_* | \mathbf{y}) &= \frac{p(\mathbf{f}, \mathbf{f}_*)p(\mathbf{y} | \mathbf{f})}{p(\mathbf{y})} \\
p(\mathbf{f}_* | \mathbf{y}) &= \int \frac{p(\mathbf{f}, \mathbf{f}_*)p(\mathbf{y} | \mathbf{f})}{p(\mathbf{y})} d\mathbf{f} \\
&= \mathcal{N}(\mathbf{K}_{*f}(\mathbf{K}_{ff} + \sigma_{\text{noise}}^{2}I)^{-1}\mathbf{y}, \
              \mathbf{K}_{**}-\mathbf{K}_{*f}(\mathbf{K}_{ff} + \sigma_{\text{noise}}^{2}I)^{-1}\mathbf{K}_{f*})
\end{aligned}
$$

The problem with this approach is that it requires an inversion of a matrix of size $n \times n$, which is a $\mathcal{O}(n^3)$ operation, where $n$ is the number of training set samples. Thus this method can handle at most a few thousand training samples. Checkout my [tutorial on Gaussian processes](https://kdkalvik.github.io/gp-tutorial) for a comprehensive explanation. 

---

# Sparse Gaussian processes


In sparse Gaussian processes, we augment the Gaussian process with additional data points $\mathbf{X}_u$ called inducing points, each with a corresponding latent variable $\mathbf{u}$. The inducing points have the following prior distribution, which is the same as the prior for the original latent variables $\mathbf{f}$.

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
p(\mathbf{f} | \mathbf{X}, \mathbf{X}_u, \mathbf{u}) = \mathcal{N}(\mathbf{f} | \mathbf{a}, \tilde{\mathbf{K}}) \tag{3} \\
\mathbf{a} = \mathbf{K}_{fu}\mathbf{K}_{uu}^{-1}\mathbf{u} \\ 
\tilde{\mathbf{K}} = \mathbf{K}_{ff} - \mathbf{K}_{fu} \mathbf{K}_{uu}^{-1} \mathbf{K}_{uf} \\
$$

The above conditional follows from the standard Gaussian conditioning operation used in [Gaussian processes](https://kdkalvik.github.io/gp-tutorial). 

Alright, so why did we introduce the inducing points, and how will that help us reduce the computation cost of Gaussian processes? 

$$\mathbf{f} = \{ \mathbf{u}, \mathbf{f}_{\neq u} \}$$

From Jensen's inequality

$$
\begin{aligned}
\log p(\mathbf{y}) &= \log \int p(\mathbf{y}, \mathbf{f}) d\mathbf{f} \\
&\geq \int q(f) \log \frac{p(\mathbf{y, f})}{q(\mathbf{f})} d\mathbf{f} \\
&= \mathcal{F}(q)
\end{aligned}
$$


This bound it tight when 

$$
q(\mathbf{f}) = p(\mathbf{f|y})
$$

The true posterior for such a factorization is shown below

$$
\begin{aligned}
p(\mathbf{f}|\mathbf{y}) = p(\mathbf{f}_{\neq u} | \mathbf{u}) p(\mathbf{u|y})
\end{aligned}
$$

But such a bound is intractable. Therefore we instead use the distribution below

$$
\begin{aligned}
q(\mathbf{f}) &= q(\mathbf{f}_{\neq u}, \mathbf{u}) \\
              &= p(\mathbf{f}_{\neq u} | \mathbf{u}) q(\mathbf{u})
\end{aligned}
$$

$$
\begin{aligned}
\mathcal{F}(q) = \int p(\mathbf{f|u}) q(\mathbf{u}) \log \frac{p(\mathbf{y|f}) p(\mathbf{u})}{q(\mathbf{u})} d\mathbf{f}
\end{aligned}
$$

But in we don't want to optimize for $q$ directly and intergrate it instead as it would avoid overfitting, giving us the following

$$
\begin{aligned}
\mathcal{F}(q) &= \int p(\mathbf{f|u}) q(\mathbf{u}) \log \frac{p(\mathbf{y|f}) p(\mathbf{u})}{q(\mathbf{u})} d\mathbf{f} d\mathbf{u} \\
&= \int q(\mathbf{u}) \left( \int p(\mathbf{f|u})  \log \frac{p(\mathbf{y|f}) p(\mathbf{u})}{q(\mathbf{u})} d\mathbf{f} \right) d\mathbf{u} \\
&= \int q(\mathbf{u}) \left( \underbrace{\int p(\mathbf{f|u})  \log p(\mathbf{y|f}) d\mathbf{f}}_{G(\mathbf{u, y})} + \log \frac{p(\mathbf{u})}{q(\mathbf{u})} \right) d\mathbf{u} 
\end{aligned}
$$


$$
\begin{aligned}
G(\mathbf{u, y}) &= \int p(\mathbf{f|u})  \log p(\mathbf{y|f}) d\mathbf{f} \\
&= \int p(\mathbf{f|u}) \left( -\frac{n}{2} \log(2\pi \sigma^2) - \frac{1}{2 \sigma^2} Tr \left[ \mathbf{y}\mathbf{y}^\top - 2 \mathbf{y} \mathbf{f}^\top + \mathbf{f}\mathbf{f}^\top \right] \right) d\mathbf{f} \\
&= -\frac{n}{2} \log(2\pi \sigma^2) - \frac{1}{2 \sigma^2} Tr \left[ \mathbf{y}\mathbf{y}^\top - 2 \mathbf{y} \mathbf{\alpha}^\top + \mathbf{\alpha}\mathbf{\alpha}^\top + \mathbf{K}_{ff} - \mathbf{Q} \right]  \\
&= \log [\mathcal{N}(\mathbf{y} | \mathbf{\alpha}, \sigma^2I)]  - \frac{1}{2 \sigma^2} Tr (\mathbf{K}_{ff} - \mathbf{Q})  \\
\end{aligned}
$$


<details>
  <summary>Follows from this equation</summary>  
  $$
  cov[\mathbf{f}|\mathbf{u}] = \mathbb{E}[\mathbf{f}\mathbf{f}^\top|\mathbf{u}] - \mathbb{E}[\mathbf{f}|\mathbf{u}]\mathbb{E}[\mathbf{f}|\mathbf{u}]^\top \\
  \mathbb{E}[\mathbf{f}\mathbf{f}^\top|\mathbf{u}] = \mathbb{E}[\mathbf{f}|\mathbf{u}]\mathbb{E}[\mathbf{f}|\mathbf{u}]^\top cov[\mathbf{f}|\mathbf{u}]
  $$
  Also, because the trace operation is linear.
</details>

$$
\begin{aligned}
\mathcal{F}(q) &= \int q(\mathbf{u}) \log \frac{\mathcal{N}(\mathbf{y} | \mathbf{\alpha}, \sigma^2I) p(\mathbf{u})}{q(\mathbf{u})} d\mathbf{u} - \frac{1}{2 \sigma^2} Tr (\mathbf{K}_{ff} - \mathbf{Q}) \\
\end{aligned}
$$

Now if we reverse the Jensen's inequality, we get the opitmal bound, which can be achieved with the optimal variational distribution $q^*(\mathbf{u})$

$$
\begin{aligned}
\mathcal{F}(q) &= \log \int q(\mathbf{u}) \frac{\mathcal{N}(\mathbf{y} | \mathbf{\alpha}, \sigma^2I) p(\mathbf{u})}{q(\mathbf{u})} d\mathbf{u} - \frac{1}{2 \sigma^2} Tr (\mathbf{K}_{ff} - \mathbf{Q}) \\
&= \log \int \mathcal{N}(\mathbf{y} | \mathbf{\alpha}, \sigma^2I) p(\mathbf{u}) d\mathbf{u} - \frac{1}{2 \sigma^2} Tr (\mathbf{K}_{ff} - \mathbf{Q}) \\
&= \log [\mathcal{N}(\mathbf{y} | 0, \sigma^2I + Q)] - \frac{1}{2 \sigma^2} Tr (\mathbf{K}_{ff} - \mathbf{Q}) \\
\end{aligned}
$$

The optimal distribution $q^*(\mathbf{u})$ that gives rise to this bound is given by 
$$
\begin{aligned}
q^*(\mathbf{u}) &\propto \mathcal{N}(\mathbf{y} | \mathbf{\alpha}, \sigma^2I) p(\mathbf{u}) \\
&= c \exp{ \left( \frac{1}{2} \mathbf{u}^\top (\mathbf{K}_{uu}^{-1} + \frac{1}{2 \sigma^2} \mathbf{K}_{uu}^{-1}\mathbf{K}_{uf}\mathbf{K}_{fu}\mathbf{K}_{uu}^{-1}  ) \mathbf{u} + \frac{1}{2 \sigma^2} \mathbf{y}^\top \mathbf{K}_{uf} \mathbf{K}_{uu}^{-1} \mathbf{u} \right)}
\end{aligned}
$$

where $c$ is a constant. Completing the quadratic form we recognize the Gaussian

$$
q^*(\mathbf{u}) = \mathcal{N}(\mathbf{u} | \sigma^{-2} \mathbf{K}_{uu} \mathbf{\Sigma}^{-1} \mathbf{K}_{uf} \mathbf{y}, \mathbf{K}_{uu} \mathbf{\Sigma}^{-1} \mathbf{K}_{uu})
$$

Where $$\mathbf{\Sigma} = \mathbf{K}_{uu} + \sigma^{-2}\mathbf{K}_{uf}\mathbf{K}_{fu}$$

First, assuming we know the distribution of $\mathbf{u}$, our above formulation allows us to marginalize the inducing variables $\mathbf{u}$ and recover the original distribution over only the variables $\mathbf{f}$, which is used for exact inference in Gaussian processes.

$$
\begin{aligned}
p(\mathbf{f}) &= \int p(\mathbf{f}|\mathbf{u}) p(\mathbf{u}) d\mathbf{u} \\
&= \int \underbrace{\mathcal{N}(\mathbf{f} | \mathbf{K}_{fu}\mathbf{K}_{uu}^{-1}\mathbf{u}, \mathbf{K}_{ff} - \mathbf{K}_{fu} \mathbf{K}_{uu}^{-1} \mathbf{K}_{uf})}_{\text{Equation (3)}} \times \underbrace{\mathcal{N}(\mathbf{u} |0, \mathbf{K}_{uu})}_{\text{Equation (1)}} d\mathbf{u}\\
&= \underbrace{\mathcal{N}(\mathbf{f}|\mathbf{K}_{fu}\mathbf{K}_{uu}^{-1}\mathbf{0}, \mathbf{K}_{ff} - \mathbf{K}_{fu} \mathbf{K}_{uu}^{-1} \mathbf{K}_{uf} + \mathbf{K}_{fu} \mathbf{K}_{uu}^{-1} \mathbf{K}_{uf})}_{\text{Equation (A.1) in the appendix}} \\
&= \mathcal{N}(\mathbf{f}| \mathbf{0}, \mathbf{K}_{ff}) 
\end{aligned}
$$

And second, the above $\mathbf{f}$ is still a Gaussian distibution with a covariance matrix of size $n \times n$ and thereby cost $\mathcal{O}(n^3)$ to invert and get the predictions of any test samples. The bottleneck is from the conditional $p(\mathbf{f}\|\mathbf{u})$ which models the relationship between the training and inducing latent variable. 

$$
p(\mathbf{f}_* | y) = \int p(\mathbf{f}_* | \mathbf{u}, \mathbf{f}) p(\mathbf{f} | \mathbf{u}, y) p(\mathbf{u} | y) d\mathbf{f} d\mathbf{u} \\
$$

Now assuming that the inducing variables $\mathbf{u}$ are a sufficient statistic for the training variables $\mathbf{f}$, i.e., $p(\mathbf{f} \| \mathbf{u}, y) = p(\mathbf{f} \| \mathbf{u})$, it would also imply 

$$p(\mathbf{f}_* | \mathbf{u}, \mathbf{f}) = p(\mathbf{f}_* | \mathbf{u})$$


$$
\begin{aligned}
p(\mathbf{f}_* | y) &= \int p(\mathbf{f}_* | \mathbf{u}) p(\mathbf{f} | \mathbf{u}) p(\mathbf{u} | y) d\mathbf{f} d\mathbf{u} \\
&= \int p(\mathbf{f}_* | \mathbf{u}) p(\mathbf{u} | y) d\mathbf{u} \int p(\mathbf{f} | \mathbf{u}) d\mathbf{f} \\
&= \int p(\mathbf{f}_* | \mathbf{u}) p(\mathbf{u} | y) d\mathbf{u}
\end{aligned}
$$

$$
m(x) = \mathbf{K}_{xu} \mathbf{K}_{uu}^{-1} \mu \\
k(x, x^\prime) = k(x, x^\prime) - \mathbf{K}_{xu} \mathbf{K}_{uu}^{-1} \mathbf{K}_{ux^\prime} + \mathbf{K}_{xu} \mathbf{K}_{uu}^{-1} \mathbf{A} \mathbf{K}_{uu}^{-1} \mathbf{K}_{ux^\prime} \\
$$

---

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
