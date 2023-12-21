---
layout: post
title: "Bayes-Newton Methods for Approximate Bayesian Inference"
author: "Kalvik Jakkala"
categories: journal
tags: [machine learning, Bayesian learning]
abstract: "Tutorial on Bayes-Newton Methods for Approximate Bayesian Inference with PSD Guarantees."
---

## Outline

* [ ] Problem
* [x] Formulation
* [ ] BLR
* [ ] Bonnet and Price's Theorem
* [x] Newton's Method & Laplace Approx.
* [ ] Bayes-Newton (VI)
* [ ] Limitations
* [ ] Bayes-Gauss-Newton (VI)
* [ ] Bayes-Quasi-Newton (VI)
* [ ] PSD constraints via Riemannian Gradients (VI)

## Taylor Series

If we know the value of $f(\mathbf{a})$ and it's derivatives  $f'(\mathbf{a}), f''(\mathbf{a}), f'''(\mathbf{a}), \cdots$, we can approximate the value of $f(\mathbf{x})$ as follows using the Taylor series:

$$
\begin{aligned}
f(\mathbf{x}) &= \sum^\infty_{n=0} \frac{f^n(\mathbf{a})}{n!} (\mathbf{x-a})^n \\
&= f(\mathbf{a}) + \frac{f'(\mathbf{a})}{1!} (\mathbf{x-a}) + \frac{f''(\mathbf{a})}{2!} (\mathbf{x-a})^2 + \frac{f'''(\mathbf{a})}{3!} (\mathbf{x-a})^3 + \cdots
\end{aligned}
$$

## Newton's Method

Newton's method leverages the Taylor series to find solutions, i.e., the stationary points of functions. We do this by considering the derivative of the second order Taylor series and equating it to zero:

$$
\begin{aligned}
0 &= \frac{d}{d (\mathbf{x}_{n+1}-\mathbf{x}_n)} \left[f(\mathbf{x}_n) + f'(\mathbf{x}_n)(\mathbf{x}_{n+1}-\mathbf{x}_n) + \frac{1}{2} f''(\mathbf{x}_n) (\mathbf{x}_{n+1}-\mathbf{x}_n)^2 \right] \\
0 &= 0 + f'(\mathbf{x}_n) + f''(\mathbf{x}_n)(\mathbf{x}_{n+1}-\mathbf{x}_n) \\
&\mathbf{x}_{n+1}-\mathbf{x}_n = -\frac{f'(\mathbf{x}_n)}{f''(\mathbf{x}_n)} \\
&\mathbf{x}_{n+1} = \mathbf{x}_n-\frac{f'(\mathbf{x}_n)}{f''(\mathbf{x}_n)}
\end{aligned}
$$

In the multivariate scenario $\frac{1}{f''(\mathbf{x}_n)} = \mathbf{H}(f(\mathbf{x}_n))^{-1}$, i.e., it is the inverse of the hessian. Unlike first order methods such as gradient descent, Newton's method uses a local quadratic approximation to find the descent direction. 

## Laplace's approximation

The Laplace's approximation is used to approximate computationally intractitble posterior probability distributions $p(\mathbf{y\mid x})$ by fitting a Gaussian distribution $q(\theta)$ with the mean $\mu$ equal to the maximum a posteriori (MAP) solution $\hat{\theta}$ and precision (inverse of the covariance $\Sigma$) equal to the Fisher information matrix $S^{-1}$:

$$
\begin{aligned}
p(\mathbf{y|x}) &\approx q(\theta) \\
&= \mathcal{N}(\mu=\hat{\theta}, \Sigma = S) \\
\text{where, } \hat{\theta} &= \text{arg} \max_\theta \log p(\mathbf{y|x; \theta}) \\
S^{-1} &= - \nabla^2_\theta \log p(\mathbf{y|x; \theta})|_{\theta=\hat{\theta}}
\end{aligned}
$$

## Problem Formulation

$$
\begin{aligned}
\mathbf{f} &\sim p(\mathbf{f}) = \mathcal{N}(\mathbf{f|\mu, K}) \text{,} \ \ \quad \quad \quad \text{(prior)} \\
\mathbf{y|f} &\sim p(\mathbf{y|f}) = \prod^N_{n=1} p(\mathbf{y}_n|\mathbf{f}_n) \text{,} \quad \quad \text{(likelihood)} \\
q(\mathbf{f}) &\approx p(\mathbf{f|y}) \propto p(\mathbf{f}) \prod^N_{n=1} p(\mathbf{y}_n|\mathbf{f}_n) \\
q(\mathbf{f}) &\propto p(\mathbf{f}) \prod^N_{n=1} t(\mathbf{f}_n) \\
\end{aligned}
$$

$$
\begin{alignat*}{3}
\text{Prior:} \quad &p(\mathbf{f}) = \mathcal{N}(\mathbf{f|\mu, K}), \quad &\lambda_\text{prior}^{(1)}=\mathbf{K}^{-1}\mu, \quad &\lambda_\text{prior}^{(2)}=-\frac{1}{2}\mathbf{K}^{-1},\\
\text{Approx. likelihood:} \quad &t(\mathbf{f}) = \mathcal{N}(\mathbf{f|}\bar{\mathbf{m}}, \bar{\mathbf{C}}), \quad &\bar{\lambda}^{(1)}=\bar{\mathbf{C}}^{-1}\bar{\mathbf{m}}, \quad &\bar{\lambda}^{(2)}=-\frac{1}{2}\bar{\mathbf{C}}^{-1},\\
\text{Approx. posterior:} \quad &q(\mathbf{f}) = \mathcal{N}(\mathbf{f|}\mathbf{m}, \mathbf{C}), \quad &\lambda^{(1)}=\mathbf{C}^{-1}\mathbf{m}, \quad &\lambda^{(2)}=-\frac{1}{2}\mathbf{C}^{-1},\\
\end{alignat*}
$$

$$
\begin{aligned}
\lambda^{(1)} &= \lambda^{(1)}_\text{prior} + \bar{\lambda}^{(1)} \\
\lambda^{(2)} &= \lambda^{(2)}_\text{prior} + \bar{\lambda}^{(2)}
\end{aligned}
$$

## Newton's Method and Laplace's approximation for Bayesian Inference

$$
\begin{aligned}
\mathcal{L}(\mathbf{f}) &= \log p(\mathbf{f|y}) =  \log p(\mathbf{y|f}) + \log p(\mathbf{f}) - \log p(\mathbf{y}) \text{,}\\
\nabla_\mathbf{f} \mathcal{L}(\mathbf{f}) &= \nabla_\mathbf{f} \log p(\mathbf{y|f}) - \mathbf{K}^{-1}(\mathbf{f-\mu}) \text{,}\\
\nabla^2_\mathbf{f} \mathcal{L}(\mathbf{f}) &= \nabla^2_\mathbf{f} \log p(\mathbf{y|f}) - \mathbf{K}^{-1} \text{.} \\
\end{aligned}
$$

Full step:

$$
\begin{aligned}
\mathbf{C}_{k+1}^{-1} &= -\nabla_\mathbf{f}^2\mathcal{L}(\mathbf{m}_k) \\
\mathbf{m}_{k+1} &= \mathbf{m}_{k} - (\nabla_\mathbf{f}^2\mathcal{L}(\mathbf{m}_k))^{-1} \nabla_\mathbf{f} \mathcal{L}(\mathbf{m}_k) \\
&= \mathbf{m}_{k} + \mathbf{C}_{k+1} \nabla_\mathbf{f} \mathcal{L}(\mathbf{m}_k) \\
\end{aligned}
$$

Partial step:

$$
\begin{aligned}
\mathbf{C}_{k+1}^{-1} &= (1-\rho) \mathbf{C}_{k}^{-1} - \rho \nabla^2_\mathbf{f} \mathcal{L}(\mathbf{m}_k) \\
\mathbf{m}_{k+1} &= \mathbf{m}_{k} + \rho \mathbf{C}_{k+1} \nabla_\mathbf{f} \mathcal{L}(\mathbf{m}_k) \\
\end{aligned}
$$

Posterior:

$$
\require{cancel}
\begin{aligned}
\lambda^{(1)}_{k+1} :=& \mathbf{C}^{-1}_{k+1} \mathbf{m}_{k+1}  \hspace{200cm} \\
\end{aligned}
$$

<details markdown=1>
  <summary>Click to see the full derivation</summary>

$$
\require{cancel}
\begin{aligned}
=& \mathbf{C}^{-1}_{k+1}[\mathbf{m}_{k} + \rho \mathbf{C}_{k+1} \nabla_\mathbf{f} \mathcal{L}(\mathbf{m}_k)] \\
=& \mathbf{C}^{-1}_{k+1}\mathbf{m}_{k} + \rho \mathbf{C}^{-1}_{k+1}\mathbf{C}_{k+1} \nabla_\mathbf{f} \mathcal{L}(\mathbf{m}_k) \\
=& \mathbf{C}^{-1}_{k+1}\mathbf{m}_{k} + \rho \nabla_\mathbf{f} \log p(\mathbf{y|\mathbf{m}_{k}}) - \rho\mathbf{K}^{-1}(\mathbf{\mathbf{m}_{k}-\mu}) \\
=& [(1-\rho) \mathbf{C}_{k}^{-1} - \rho \nabla^2_\mathbf{f} \mathcal{L}(\mathbf{m}_k)]\mathbf{m}_{k} + \rho \nabla_\mathbf{f} \log p(\mathbf{y|\mathbf{m}_{k}}) - \rho\mathbf{K}^{-1}(\mathbf{\mathbf{m}_{k}-\mu}) \\
=& (1-\rho) \mathbf{C}_{k}^{-1} \mathbf{m}_{k} - \rho \nabla^2_\mathbf{f} \mathcal{L}(\mathbf{m}_k)\mathbf{m}_{k} + \rho \nabla_\mathbf{f} \log p(\mathbf{y|\mathbf{m}_{k}}) - \rho\mathbf{K}^{-1}(\mathbf{\mathbf{m}_{k}-\mu}) \\
=& (1-\rho) \mathbf{C}_{k}^{-1} \mathbf{m}_{k} - \rho \left[\nabla^2_\mathbf{f} \log p(\mathbf{y|\mathbf{m}_{k}}) - \mathbf{K}^{-1}\right] \mathbf{m}_{k} + \rho \nabla_\mathbf{f} \log p(\mathbf{y|\mathbf{m}_{k}}) - \rho\mathbf{K}^{-1}(\mathbf{\mathbf{m}_{k}-\mu}) \\
=& (1-\rho) \mathbf{C}_{k}^{-1} \mathbf{m}_{k} - \rho \nabla^2_\mathbf{f} \log p(\mathbf{y|\mathbf{m}_{k}}) \mathbf{m}_{k} + \cancel{\rho \mathbf{K}^{-1} \mathbf{m}_{k}} + \rho \nabla_\mathbf{f} \log p(\mathbf{y|\mathbf{m}_{k}}) - \cancel{\rho\mathbf{K}^{-1}\mathbf{\mathbf{m}_{k}}}+\rho\mathbf{K}^{-1}\mu \\
=& (1-\rho) \mathbf{C}_{k}^{-1} \mathbf{m}_{k}  +\rho\mathbf{K}^{-1}\mu - \rho \nabla^2_\mathbf{f} \log p(\mathbf{y|\mathbf{m}_{k}}) \mathbf{m}_{k} + \rho \nabla_\mathbf{f} \log p(\mathbf{y|\mathbf{m}_{k}}) \\
=& (1-\rho) \bar{\lambda}^{(1)}_{k}  + \rho \lambda^{(1)}_\text{prior} + \rho \left[ \nabla_\mathbf{f} \log p(\mathbf{y|\mathbf{m}_{k}}) - \nabla^2_\mathbf{f} \log p(\mathbf{y|\mathbf{m}_{k}}) \mathbf{m}_{k} \right] \\
\end{aligned}
$$
</details>

$$
\require{cancel}
\begin{aligned}
=& \rho \lambda^{(1)}_\text{prior} + \underbrace{(1-\rho) \bar{\lambda}^{(1)}_{k} + \rho \left[ \nabla_\mathbf{f} \log p(\mathbf{y|\mathbf{m}_{k}}) - \nabla^2_\mathbf{f} \log p(\mathbf{y|\mathbf{m}_{k}}) \mathbf{m}_{k} \right]}_{\bar{\lambda}^{(1)}_{k+1}} \hspace{200cm} \\
\end{aligned}
$$

$$
\require{cancel}
\begin{aligned}
\lambda^{(2)}_{k+1} :=& -\frac{1}{2}\mathbf{C}^{-1}_{k+1} \hspace{200cm} \\
\end{aligned}
$$

<details markdown=1>
  <summary>Click to see the full derivation</summary>
$$
\require{cancel}
\begin{aligned}
=& -\frac{1}{2} \left[ (1-\rho) \mathbf{C}_{k}^{-1} - \rho \nabla^2_\mathbf{f} \mathcal{L}(\mathbf{m}_k) \right] \\
=& -\frac{1}{2} (1-\rho) \mathbf{C}_{k}^{-1} +\frac{1}{2} \rho \left[ \nabla^2_\mathbf{f} \log p(\mathbf{y|\mathbf{m}_{k}}) - \mathbf{K}^{-1} \right] \\ 
=& -\frac{1}{2} (1-\rho) \left[ \bar{\mathbf{C}}_{k}^{-1} + \mathbf{K}^{-1}  \right] +\frac{1}{2} \rho \left[ \nabla^2_\mathbf{f} \log p(\mathbf{y|\mathbf{m}_{k}}) - \mathbf{K}^{-1} \right] \\ 
=& -\frac{1}{2} \bar{\mathbf{C}}_{k}^{-1} (1-\rho) -\frac{1}{2} \mathbf{K}^{-1}  (1-\rho) + \frac{1}{2} \rho \nabla^2_\mathbf{f} \log p(\mathbf{y|\mathbf{m}_{k}}) - \frac{1}{2} \rho\mathbf{K}^{-1}\\ 
=& -\frac{1}{2} \bar{\mathbf{C}}_{k}^{-1} (1-\rho) -\frac{1}{2} (1-\rho) \mathbf{K}^{-1} - \frac{1}{2} \rho\mathbf{K}^{-1} + \frac{1}{2} \rho \nabla^2_\mathbf{f} \log p(\mathbf{y|\mathbf{m}_{k}}) \\ 
=& -\frac{1}{2} \bar{\mathbf{C}}_{k}^{-1} (1-\rho) -\frac{1}{2} \mathbf{K}^{-1} + \frac{1}{2} \rho \nabla^2_\mathbf{f} \log p(\mathbf{y|\mathbf{m}_{k}}) \\ 
=& (1-\rho) \bar{\lambda}^{(2)}_{k} + \lambda^{(2)}_\text{prior}  + \rho \frac{1}{2}\nabla^2_\mathbf{f} \log p(\mathbf{y|\mathbf{m}_{k}}) \\ 
\end{aligned}
$$
</details>

$$
\require{cancel}
\begin{aligned}
=& \lambda^{(2)}_\text{prior} + \underbrace{(1-\rho) \bar{\lambda}^{(2)}_{k} + \rho \frac{1}{2}\nabla^2_\mathbf{f} \log p(\mathbf{y|\mathbf{m}_{k}})}_{\bar{\lambda}^{(2)}_{k+1}}  \hspace{200cm} \\ 
\end{aligned}
$$

## Bayes-Newton (Variational Inference)

Bayesian inference methods such as variational inference, power expectation propagation, and posterior linearisation can be view as Bayesian generalisations of Newton's method. 

Variational inference with $q(\mathbf{f})$ parameterized by $\lambda$:

$$
q^*(\mathbf{f}) = \text{arg} \min_{q(\mathbf{f})} \text{KL}(q(\mathbf{f}) || p(\mathbf{f} | \mathbf{y})) 
$$

The variational free energy (VFE), i.e., the negative of the evidence lower bound (ELBO):

$$
\begin{aligned}
\text{VFE}(q(\mathbf{f})) &= \mathbb{E}_{q(\mathbf{f})} [\ln q(\mathbf{f})] - \mathbb{E}_{q(\mathbf{f})} [\ln p(\mathbf{f}, \mathbf{y})] \\
\end{aligned}
$$

### Bayesian Learning Rule (BLR)

We can efficiently optimize the variational distribution using the BLR. The BLR updates use natural-gradients $$\tilde{\nabla}_\lambda$$, which scale the vanilla gradients $\nabla_\lambda$ by the Fisher information matrix $\mathbf{F}$ for faster convergence:

$$
\begin{aligned}
\lambda_{k+1} &= \lambda_{k} - \rho \tilde{\nabla}_\lambda \text{VFE}(q(\mathbf{f}))\rvert_{\lambda=\lambda_k} \\
&=  \lambda_k - \rho \underbrace{[\mathbf{F}(\lambda_k)]^{-1} \nabla_\lambda}_\text{natural gradient $\tilde{\nabla}_\lambda$} \text{VFE}(q(\mathbf{f}))\rvert_{\lambda=\lambda_k} \\
&=  \lambda_k - \rho \nabla_\eta \text{VFE}(q(\mathbf{f}))\rvert_{\eta=\eta_k}
\end{aligned}
$$

However, the method leverages the properties of the natural parametrization $\lambda$ of the exponential-family distributions to avoid having to explicitly computing the Fisher information matrix $\mathbf{F}$ by computing the gradients with respect to the expectation parameters $\eta$. 

We can derive ...:

$$
\begin{aligned}
\lambda_{k+1} &= \lambda_k - \rho \nabla_\eta \text{VFE}(q(\mathbf{f})) \\
&= \lambda_k - \rho \left( \nabla_\eta \mathbb{E}_{q(\mathbf{f})} [\ln q(\mathbf{f})] - \nabla_\eta\mathbb{E}_{q(\mathbf{f})} [\ln p(\mathbf{f}, \mathbf{y})] \right)  \\
&= \lambda_k - \rho \left( \lambda_k - \nabla_\eta\mathbb{E}_{q(\mathbf{f})} [\ln p(\mathbf{f}, \mathbf{y})] \right)  \\
&= (1-\rho) \lambda_k + \rho \nabla_\eta\mathbb{E}_{q(\mathbf{f})} [\ln p(\mathbf{f}, \mathbf{y})] \\
\end{aligned}
$$

Refer to my tutorial on the [foundations of BLR](http://127.0.0.1:4000/CVI#natural-param-derivative) for proof of $\nabla_\eta \mathbb{E}_{q(\mathbf{f})} [\ln q(\mathbf{f})] = \lambda_k$.

<details markdown=1>
  <summary>Alternate Parameterizations and their Derivatives (click to see the details)</summary>

### Alternate Parameterizations and their Derivatives
The exponential-family distributions such as the Gaussian distribution can be parametrized by the standard parameters $\theta$, i.e., the mean and covariance, and also the expectation parameters $\eta$ which can be related to the standard parameters $\theta$ as follows:

$$
\begin{alignat*}{3}
\eta^{(1)} &= \mathbf{m} \quad & \eta^{(2)} &= \mathbf{C} + \mathbf{mm^\top} \\
\implies \mathbf{m} &= \eta^{(1)} \quad \quad & \mathbf{C} &= \eta^{(2)} - \eta^{(1)} \left( \eta^{(1)} \right)^\top
\end{alignat*}
$$

We can use the above to derive the following derivatives:

$$
\begin{aligned}
\frac{d}{d\eta^{(1)}} f(\mathbf{m}, \mathbf{C}) &= \frac{df(\mathbf{m}, \mathbf{C})}{d\mathbf{m}} \frac{d\mathbf{m}}{d\eta^{(1)}} + \frac{d f(\mathbf{m}, \mathbf{C})}{d\mathbf{C}} \frac{d\mathbf{C}}{d\eta^{(1)}} \\
&= \frac{df(\mathbf{m}, \mathbf{C})}{d\mathbf{m}} + \frac{d f(\mathbf{m}, \mathbf{C})}{d\mathbf{C}} (-2\eta^{(1)}) \\
&= \frac{df(\mathbf{m}, \mathbf{C})}{d\mathbf{m}} -2 \frac{d f(\mathbf{m}, \mathbf{C})}{d\mathbf{C}} \mathbf{m} \\
\implies \nabla_\eta^{(1)} f(\mathbf{m}, \mathbf{C}) &= \nabla_\mathbf{m} f(\mathbf{m}, \mathbf{C}) - 2 \nabla_\mathbf{C} f(\mathbf{m}, \mathbf{C}) \mathbf{m} \\
\\
\\
\frac{d}{d\eta^{(2)}} f(\mathbf{m}, \mathbf{C}) &= \frac{df(\mathbf{m}, \mathbf{C})}{d\mathbf{m}} \frac{d\mathbf{m}}{d\eta^{(2)}} + \frac{d f(\mathbf{m}, \mathbf{C})}{d\mathbf{C}} \frac{d\mathbf{C}}{d\eta^{(2)}} \\
&= 0 + \frac{d f(\mathbf{m}, \mathbf{C})}{d\mathbf{C}} \\
\implies \nabla_\eta^{(2)} f(\mathbf{m}, \mathbf{C}) &= \nabla_\mathbf{C} f(\mathbf{m}, \mathbf{C}) \\
\end{aligned}
$$
</details>

<details markdown=1>
  <summary>Bonnet's and Price's Theorems (click to see the details)</summary>

We can compute the gradients with respect to the expectation parameters $\eta$.

### Bonnet's and Price's Theorems

Let $f(\mathbf{x})$ be an arbitrary function of $\mathbf{x}$. Suppose that the elements $x_n$ of $\mathbf{x}$ are stochastic variables with joint probability density function $p(\mathbf{x})$. Then, for the special case that $\mathbf{x} \sim \mathcal{N}(\mathbf{m}, \mathbf{C})$, i.e., the variables $\mathbf{x}$ have a Gaussian distribution, the theorems give us the following results:

* [Mean formulation](https://link.springer.com/article/10.1007/BF02994793): 

$$
\begin{aligned}
\frac{\partial}{\partial m_n} \int^\infty_{-\infty} p(\mathbf{x}) f(\mathbf{x}) d\mathbf{x} &= \int^\infty_{-\infty} p(\mathbf{x}) \left[ \frac{\partial}{\partial x_n} f(\mathbf{x}) \right] d\mathbf{x} \\
\frac{\partial}{\partial m_n} \mathbb{E}_{p(\mathbf{x})}[f(\mathbf{x})] &= \mathbb{E}_{p(\mathbf{x})}\left[ \frac{\partial}{\partial x_n} f(\mathbf{x}) \right] \\
\nabla_\mathbf{m} \mathbb{E}_{p(\mathbf{x})}[f(\mathbf{x})] &= \mathbb{E}_{p(\mathbf{x})}\left[ \nabla_\mathbf{x} f(\mathbf{x}) \right] \\
\end{aligned}
$$

* [Covariance formulation](https://arxiv.org/pdf/1710.03576.pdf): 

$$
\begin{aligned}
\frac{\partial}{\partial \mathbf{C}_{ij}} \int^\infty_{-\infty} p(\mathbf{x}) f(\mathbf{x}) d\mathbf{x} &= c_{ij} \int^\infty_{-\infty} p(\mathbf{x}) \left[ \frac{\partial^2}{\partial x_i \partial x_j} f(\mathbf{x}) \right] d\mathbf{x} \\
\frac{\partial}{\partial \mathbf{C}_{ij}} \mathbb{E}_{p(\mathbf{x})}[f(\mathbf{x})] &= c_{ij} \mathbb{E}_{p(\mathbf{x})}\left[ \frac{\partial^2}{\partial x_i \partial x_j} f(\mathbf{x}) \right] \\
\nabla_\mathbf{C} \mathbb{E}_{p(\mathbf{x})}[f(\mathbf{x})] &= \frac{1}{2} \mathbb{E}_{p(\mathbf{x})}\left[ \nabla^2_\mathbf{x} f(\mathbf{x}) \right] \\
\end{aligned}
$$

where $c_{ij} = 1/2$ if $i=j$ and $c_{ij} = 1$ if $i \neq j$ (since $\mathbf{C}$ is symmetric).
</details>

---

$$
\begin{aligned}
\end{aligned}
$$


# References

* [Amari's book—Information Geometry and Its Applications](https://link.springer.com/book/10.1007/978-4-431-55978-8)
* [Amari's paper on natural gradients](https://direct.mit.edu/neco/article-abstract/10/2/251/6143/Natural-Gradient-Works-Efficiently-in-Learning?redirectedFrom=fulltext)
* [Khan and Lin's paper on Conjugate-computation Variational Inference (CVI)](https://arxiv.org/abs/1703.04265)
* [Khan and Nielsen's paper on natural gradient descent](https://arxiv.org/abs/1807.04489)
* [Khan's slides on CVI](https://bigdata.nii.ac.jp/eratokansyasai4/wp-content/uploads/2017/09/4_Emitiyaz-Khan.pdf)
* [David Blei's tutorial on Variational Inference](https://arxiv.org/abs/1601.00670)
* [Khan and Rue's paper on the bayesian learning rule](https://www.jmlr.org/papers/volume24/22-0291/22-0291.pdf)
* [Wilkinson et al. Bayes-Newton Methods for Approximate Bayesian Inference with PSD Guarantees](https://www.jmlr.org/papers/volume24/21-1298/21-1298.pdf)