---
layout: post
title: "Variational Gaussian approximation with only $O(N)$ free parameters"
author: "Kalvik Jakkala"
categories: journal
tags: [documentation, sample]
image: variational-inference.png
abstract: "Tutorial on variational Gaussian approximation and why using Gaussian priors and factorizing likelihoods leads to only $O(N)$ instead of $O(N^2)$ variational parameters, $N$ being the number of random variables."
---

## Variational Gaussian Approximation

It is often the case that there is no closed-form solution to the posterior $$p(\mathbf{x} \mid \mathbf{y})$$:

$$ p(\mathbf{x}|\mathbf{y}) = \frac{p(\mathbf{y}|\mathbf{x}) p(\mathbf{x})}{p(\mathbf{y})} $$

For example, when the likelihood $$p(\mathbf{y} \mid \mathbf{x})$$ is non-Gaussian and the prior $$p(\mathbf{x})$$ is Gaussian, one has to resort to approximations of the posterior. The variational approximation is a popular method for this task. Given the mathematically elegant properties of [Gaussian distributions](https://itskalvik.github.io/GPs), practitioners often choose to approximate the posterior as a Gaussian, even when the true posterior is not Gaussian.

But a Gaussian distribution $\mathcal{N}(\mu, \mathbf{\Sigma})$ over $N$ random variables with mean $\mu$ and covariance $\mathbf{\Sigma}$, has $\mathcal{O}(N^2)$ paramters ($N$ mean paramters and $N^2$ covariance paramters) that need to be optimized, referred to as variational parameters. Consequently, the variational Gaussian approximation does not scale well. However, [Opper and Archambeau, 2009](https://direct.mit.edu/neco/article-abstract/21/3/786/7385/The-Variational-Gaussian-Approximation-Revisited?redirectedFrom=fulltext) showed that if the prior distribution $p(\mathbf{x})$ is a Gaussian and the likelihood $p(\mathbf{y} \mid \mathbf{x})$ is factorized, the number of free variational parameters is only $\mathcal{O}(N)$, i.e., only $\mathcal{O}(N)$ of the $\mathcal{O}(N^2)$ variational parameters need to be optimized. 

I found the original derivation of [Opper and Archambeau, 2009](https://direct.mit.edu/neco/article-abstract/21/3/786/7385/The-Variational-Gaussian-Approximation-Revisited?redirectedFrom=fulltext) to be terse and difficult to follow initially. Therefore, this article will provide a detailed derivation of their key results.

## Variational Free Energy (VFE)

<img src="{{ site.github.url }}/assets/img/variational-inference.png" width="100%" style="vertical-align:middle"/>

The variational approach aims to approximate the posterior $p(\mathbf{x} \mid \mathbf{y})$ with the variational distribution $q(\mathbf{x})$, where the distribution's parameters determine its density. The optimal $q^*(\mathbf{x})$ is obtained by minimizing the KL divergence:

$$
q^*(\mathbf{x}) = \text{arg} \min_{q(\mathbf{x})} \text{KL}(q(\mathbf{x}) || p(\mathbf{x} | \mathbf{y})) 
$$

The KL can be reformulated as follows:

$$
\begin{aligned}
\text{KL}(q(\mathbf{x}) || p(\mathbf{x} | \mathbf{y})) &= \mathbb{E}_{q(\mathbf{x})} [\ln q(\mathbf{x})] - \mathbb{E}_{q(\mathbf{x})} [\ln p(\mathbf{x}|\mathbf{y})] \\
&= \mathbb{E}_{q(\mathbf{x})} [\ln q(\mathbf{x})] - \mathbb{E}_{q(\mathbf{x})} \left[ \ln \frac{p(\mathbf{x}, \mathbf{y})}{p(\mathbf{y})} \right] \\
&= \mathbb{E}_{q(\mathbf{x})} [\ln q(\mathbf{x})] - \mathbb{E}_{q(\mathbf{x})} [\ln p(\mathbf{x}, \mathbf{y})] + \mathbb{E}_{q(\mathbf{x})}[\ln p(\mathbf{y})] \\
&= \mathbb{E}_{q(\mathbf{x})} [\ln q(\mathbf{x})] - \mathbb{E}_{q(\mathbf{x})} [\ln p(\mathbf{x}, \mathbf{y})] +  \ln p(\mathbf{y}) \\
\end{aligned}
$$

In the last equation above, $\ln p(\mathbf{y})$ is independent of $q(\mathbf{x})$. Therefore, we drop it to get the variational free energy $\mathcal{F}$ used to optimize the variational distribution:

$$
\begin{aligned}
\mathcal{F} &= \underbrace{\mathbb{E}_{q(\mathbf{x})} [\ln q(\mathbf{x})]}_{-\text{entropy of } q(\mathbf{x})} - \mathbb{E}_{q(\mathbf{x})} [\ln p(\mathbf{x}, \mathbf{y})] \\
&= -\frac{1}{2} \ln |\mathbf{\Sigma}| -\frac{N}{2} \ln(2\pi e) - \langle \ln p(\mathbf{x}, \mathbf{y}) \rangle_{q(\mathbf{x})}
\end{aligned}
$$

When the variational distribution $q(\mathbf{x})$ is a Gaussian $\mathcal{N}(\mu, \mathbf{\Sigma})$, the solution $q^*(\mathbf{x})$ can be obtained by taking the derivative of the variational free energy $\mathcal{F}$ with respect to the parameters $\mu$ and $\mathbf{\Sigma}$ of the variational distribution $q(\mathbf{x})$ and setting them equal to zero:

$$
\nabla_{\mu} \mathcal{F} =  - \nabla_{\mu} \langle \ln p(\mathbf{x}, \mathbf{y}) \rangle_{q(\mathbf{x})}
$$

$$
\begin{aligned}
\nabla_{\mathbf{\Sigma}} \mathcal{F} &=  - \frac{1}{2} \mathbf{\Sigma}^{-\top} - \nabla_{\mathbf{\Sigma}} \langle \ln p(\mathbf{x}, \mathbf{y}) \rangle_{q(\mathbf{x})} \\
\implies \mathbf{\Sigma}^{-\top} &= - 2\nabla_{\mathbf{\Sigma}} \langle \ln p(\mathbf{x}, \mathbf{y}) \rangle_{q(\mathbf{x})}
\end{aligned}
$$

## VFE for Gaussian Prior and Factorized Likelihood

So far, the only assumption we made was that the variational distribution $q(\mathbf{x})$ is Gaussian. As such, the above variational approach does not scale well (i.e., $\mathcal{O}(N^2)$ variational parameters). However, with two additional assumptions:

$$
\begin{aligned}
p(\mathbf{x}) &\sim \mathcal{N}(0, \mathbf{K}) \\
p(\mathbf{y}|\mathbf{x}) &= \prod_n p(y_n|x_n) 
\end{aligned}
$$

i.e., the prior distribution $p(\mathbf{x})$ is also a Gaussian (which can also have a non-zero mean), and if the likelihood of the data $p(\mathbf{y}\mid\mathbf{x})$ factorizes, it can be shown that the number of free variational parameters is only $\mathcal{O}(N)$. Opper and Archambeau demonstrated this by plugging the joint distribution $p(\mathbf{x}, \mathbf{y})$ with the above assumptions into the variational free energy $\mathcal{F}$. To see this, we first compute $\ln p(\mathbf{x}, \mathbf{y})$:

$$
\begin{aligned}
p(\mathbf{x}, \mathbf{y}) &= p(\mathbf{x}) p(\mathbf{y}|\mathbf{x}) \\
&= \frac{1}{(2\pi)^\frac{N}{2}|\mathbf{K}|^\frac{1}{2}} \exp \left\{ -\frac{1}{2}({\mathbf{x}^\top \mathbf{K}^{-1} \mathbf{x}}) \right\} \exp \left\{\ln \prod_n p(y_n|x_n) \right\} \\
&= \frac{1}{Z_0} \exp \left\{ -\frac{1}{2}({\mathbf{x}^\top K^{-1} \mathbf{x}}) \right\} \exp \left\{- \sum_n - \ln p(y_n|x_n) \right\} \\
\implies & \ln p(\mathbf{x}, \mathbf{y}) = -\ln Z_0 -\frac{1}{2}({\mathbf{x}^\top \mathbf{K}^{-1} \mathbf{x}}) - \sum_n -\ln \left[p(y_n|x_n) \right] \\
\end{aligned}
$$

In the above, note that the likelihoods $p(y_n \mid x_n)$ can have any distribution and are not limited to Gaussians. Although the likelihoods appear in an exponential form, the natural log cancels the exponent, yielding the original product of likelihoods $\prod_n p(y_n \mid x_n)$. We also collect the normalization constants, including $\mid \mathbf{K}\mid $ in $Z_0$. Now, we can compute the expectation of $\ln p(\mathbf{x}, \mathbf{y})$ with respect to the variational distribution $q(\mathbf{x})$, which will be used to compute the variational free energy $\mathcal{F}$:

$$
\begin{aligned}
\langle & \ln p(\mathbf{x}, \mathbf{y}) \rangle_{q(\mathbf{x})} = \langle -\frac{1}{2}({\mathbf{x}^\top \mathbf{K}^{-1} \mathbf{x}}) - \sum_n -\ln \left[p(y_n|x_n) \right] - \ln Z_0 \rangle_{q(\mathbf{x})} \\
&= -\langle \frac{1}{2}({\mathbf{x}^\top \mathbf{K}^{-1} \mathbf{x}}) \rangle_{q(\mathbf{x})} - \sum_n \langle -\ln \left[p(y_n|x_n) \right] \rangle_{q(x_n)} - \langle \ln Z_0 \rangle_{q(\mathbf{x})} \\
&= -\langle \frac{1}{2}\text{Tr}(\mathbf{K}^{-1} \mathbf{x}\mathbf{x}^\top) \rangle_{q(\mathbf{x})} - \sum_n \langle -\ln \left[p(y_n|x_n) \right] \rangle_{q(x_n)} - \ln Z_0 \\
&= - \frac{1}{2}\text{Tr}(\mathbf{K}^{-1} (\mathbf{\Sigma} + \mu\mu^\top)) - \sum_n -\langle \ln \left[p(y_n|x_n) \right] \rangle_{q(x_n)} - \ln Z_0 \\
&= - \frac{1}{2}\text{Tr}(\mathbf{K}^{-1}\mathbf{\Sigma}) - \frac{1}{2}\text{Tr}(K^{-1}\mu\mu^\top) - \sum_n \langle -\ln \left[p(y_n|x_n) \right] \rangle_{q(x_n)} - \ln Z_0 \\
&= - \frac{1}{2}\text{Tr}(\mathbf{K}^{-1}\mathbf{\Sigma}) - \frac{1}{2}\mu^\top K^{-1}\mu - \sum_n \langle -\ln \left[p(y_n|x_n) \right] \rangle_{q(x_n)} - \ln Z_0 \\
\end{aligned}
$$

Now we can plug the above expectation into the variational free energy $\mathcal{F}$ to get the following:

$$
\begin{aligned}
\mathcal{F} &= \mathbb{E}_{q(\mathbf{x})} [\ln q(\mathbf{x})] - \mathbb{E}_{q(\mathbf{x})} [\ln p(\mathbf{x, y})] \\
&= -\frac{1}{2} \ln |\mathbf{\Sigma}| -\frac{N}{2} \ln(2\pi e) + \frac{1}{2}\text{Tr}(\mathbf{K}^{-1}\mathbf{\Sigma}) + \\
& \quad \quad \frac{1}{2}\mu^\top \mathbf{K}^{-1}\mu + \sum_n \langle -\ln \left[p(y_n|x_n) \right] \rangle_{q(x_n)} + \ln Z_0 \\
\end{aligned}
$$

We can compute the solution covariance matrix $\mathbf{\Sigma}$ of the variational distribution $q(\mathbf{x})$ by taking the derivative w.r.t $\mathbf{\Sigma}$ and setting it equal to zero:

$$
\begin{aligned}
\nabla_{\mathbf{\Sigma}} \mathcal{F} &= \nabla_{\mathbf{\Sigma}} \left[ -\frac{1}{2} \ln |\mathbf{\Sigma}| -\frac{N}{2} \ln(2\pi e) + \frac{1}{2}\text{Tr}(\mathbf{K}^{-1}\mathbf{\Sigma}) \right. \\
& \quad \quad \quad \quad \left. + \frac{1}{2}\mu^\top \mathbf{K}^{-1}\mu + \sum_n \langle -\ln \left[p(y_n|x_n) \right] \rangle_{q(x_n)} + \ln Z_0 \right] \\
&= -\nabla_{\mathbf{\Sigma}} \frac{1}{2} \ln |\mathbf{\Sigma}| + \nabla_{\mathbf{\Sigma}} \frac{1}{2}\text{Tr}(\mathbf{K}^{-1}\mathbf{\Sigma}) + \nabla_{\mathbf{\Sigma}} \sum_n \langle -\ln \left[p(y_n|x_n) \right] \rangle_{q(x_n)} \\
&= - \frac{1}{2} \mathbf{\Sigma}^{-\top} + \frac{1}{2}\mathbf{K}^{-1} + \sum_n \nabla_{\mathbf{\Sigma}} \langle -\ln \left[p(y_n|x_n) \right] \rangle_{q(x_n)} \\
\implies& \frac{1}{2} \mathbf{\Sigma}^{-\top} = \frac{1}{2}\mathbf{K}^{-1} + \sum_n \nabla_{\mathbf{\Sigma}} \langle -\ln \left[p(y_n|x_n) \right] \rangle_{q(x_n)} \\
\implies& \mathbf{\Sigma}^{-\top} = \mathbf{K}^{-1} + 2\sum_n \nabla_{\mathbf{\Sigma}} \langle -\ln \underbrace{\left[p(y_n|x_n) \right]}_\text{likelihood} \rangle_{q(x_n)} \\
\implies& \mathbf{\Sigma}^{-\top} = \left( \mathbf{K}^{-1} + \mathbf{\Lambda} \right) \\
\end{aligned}
$$

Due to our factorization assumption for the data likelihood, i.e., the likelihood of each $y_n$ depends only on $x_n$, the expectation of each likelihood $p(y_n\mid x_n)$ with respect to the variational distribution depends solely on $q(x_n)$—a univariate variational Gaussian distribution over $x_n$. Similarly, the derivative of each likelihood with respect to the variational distribution depends only on the $n^\text{th}$ mean term $\mu_n$ and covariance term $\Sigma_{nn}$ (an element of the diagonal). 

Therefore, we can conclude that the solution Gaussian variational distribution $q^*(\mathbf{x})$ will have a precision matrix $\mathbf{\Sigma}^{-1}$ whose off-diagonal terms are the same as the prior precision matrix $\mathbf{K}^{-1}$, and the diagonal terms will have the additional updates from the diagonal matrix $\mathbf{\Lambda}$, obtained from the derivatives of the likelihood.

This implies that our Gaussian variational distribution $q^*(\mathbf{x})$ has $N$ mean parameters and only $N$ effective covariance parameters, resulting in $\mathcal{O}(N)$ parameters. This represents a significant reduction from the conventional $\mathcal{O}(N^2)$ parameters required when optimizing the full covariance matrix instead of a diagonal matrix, making the approach scalable for variational approximations involving a large number of variables.

## New Efficient Parameterization

Based on the above results, Opper and Archambeau suggested defining the mean parameters $\mu$ of the variational distribution as follows, where $\mathcal{v}$ is a vector consisting of the new mean parameters:

$$
\mu = \mathbf{K}\mathcal{v}
$$

This simplifies the variational free energy $\mathcal{F}$ into the following which does not require us to invert the prior covariance $\mathbf{K}$:

$$
\begin{aligned}
\mathcal{F} =& -\frac{1}{2} \ln |\mathbf{\Sigma}| -\frac{N}{2} \ln(2\pi e) + \frac{1}{2}\text{Tr}(\mathbf{K}^{-1}\mathbf{\Sigma}) + \\
             & \quad \frac{1}{2} \mu^\top \mathbf{K}^{-1}\mu + \sum_n \langle -\ln \left[p(y_n|x_n) \right] \rangle_{q(x_n)} + \ln Z_0  \\
            =& -\frac{1}{2} \ln |\mathbf{\Sigma}| -\frac{N}{2} \ln(2\pi e) + \frac{1}{2}\text{Tr}(\mathbf{K}^{-1}\mathbf{\Sigma}) + \\
             & \quad \frac{1}{2} \mathcal{v}^\top\mathbf{K}^\top\mathbf{K}^{-1}\mathbf{K}\mathcal{v} + \sum_n \langle -\ln \left[p(y_n|x_n) \right] \rangle_{q(x_n)} + \ln Z_0 \\
            =& -\frac{1}{2} \ln |\mathbf{\Sigma}| -\frac{N}{2} \ln(2\pi e) + \frac{1}{2}\text{Tr}(\mathbf{K}^{-1}\mathbf{\Sigma}) + \\
             & \quad \frac{1}{2} \mathcal{v}^\top\mathbf{K}^\top\mathcal{v} + \sum_n \langle -\ln \left[p(y_n|x_n) \right] \rangle_{q(x_n)} + \ln Z_0 \\
\end{aligned}
$$

The above gives us the following gradients with respect to the new parameterization $\mathcal{v}$:

$$
\begin{aligned}
\frac{\partial\mathcal{F}}{\partial\mathcal{v}} &= \frac{\partial\mu}{\partial\mathcal{v}} \frac{\partial\mathcal{F}}{\partial\mathcal{\mu}} \\
&= \frac{\partial}{\partial\mathcal{v}}\mathbf{K}\mathcal{v} \left[ \frac{\partial}{\partial\mathcal{\mu}} \frac{1}{2}\mu^\top \mathbf{K}^{-1}\mu - \underbrace{-\frac{\partial}{\partial\mathcal{\mu}}\sum_n \langle -\ln \left[p(y_n|x_n) \right] \rangle_{q(x_n)}}_{\bar{\mathcal{v}}} \right] \\
&= \mathbf{K} \left[ \mathbf{K}^{-1}\mu - \bar{\mathcal{v}} \right] \\
&= \mathbf{K} \left[ \mathbf{K}^{-1}\mathbf{K}\mathcal{v} - \bar{\mathcal{v}} \right] \\
&= \mathbf{K} \left[\mathcal{v} - \bar{\mathcal{v}} \right] \\
\end{aligned}
$$

Finally, Opper and Archambeau suggested defining the covariance parameters with the vector $\mathbf{\lambda}$, which is the diagonal of the diagonal matrix $\mathbf{\Lambda}$, giving us the following gradients:

$$
\begin{aligned}
\frac{\partial\mathcal{F}}{\partial\mathbf{\lambda}} &= \frac{\partial\mathbf{\Sigma}}{\partial\mathbf{\lambda}} \frac{\partial\mathcal{F}}{\partial\mathcal{\mathbf{\Sigma}}} \\
&= \left[ \frac{\partial}{\partial\mathbf{\lambda}} \left( \mathbf{K}^{-1} + \mathbf{\lambda} \right)^{-\top} \right] \frac{1}{2} \Biggl[ \underbrace{- \mathbf{\Sigma}^{-\top} + \mathbf{K}^{-1}}_{-\mathbf{\lambda}} \\
& \quad \quad \quad \quad \quad \quad \quad \quad \quad + \underbrace{2 \sum_n \frac{\partial}{\partial\mathbf{\Sigma}} \langle -\ln \left[p(y_n|x_n) \right] \rangle_{q(x_n)}}_{\bar{\mathbf{\lambda}}} \Biggr] \\
&= \frac{1}{2}\frac{\partial}{\partial\mathbf{\lambda}} \left( \mathbf{K}^{-1} + \mathbf{\lambda} \right)^{-\top}[-\mathbf{\lambda}+\bar{\mathbf{\lambda}}]\\
&= \frac{1}{2}-\left( \mathbf{K}^{-1} + \mathbf{\lambda} \right)^{-\top} \frac{\partial \left( \mathbf{K}^{-1} + \mathbf{\lambda} \right)}{\partial\mathbf{\lambda}} \left( \mathbf{K}^{-1} + \mathbf{\lambda} \right)^{-\top} [-\mathbf{\lambda}+\bar{\mathbf{\lambda}}] \\
&= \frac{1}{2}[\mathbf{\Sigma} \mathbf{I} \mathbf{\Sigma}] [\mathbf{\lambda}-\bar{\mathbf{\lambda}}]\\
&= \frac{1}{2}[\mathbf{\Sigma} \circ \mathbf{\Sigma}] [\mathbf{\lambda}-\bar{\mathbf{\lambda}}]\\
&= \frac{1}{2}[\mathbf{\Sigma} \circ \mathbf{\Sigma}] [\mathbf{\lambda}-\bar{\mathbf{\lambda}}]\\
\end{aligned}
$$

## Conclusion

In short, we resort to approximate inference methods when computing the original distributions becomes challenging. In Variational Gaussian approximation, a variational Gaussian distribution is employed to approximate the distribution of interest. However, this method entails optimizing $N$ mean parameters and $N^2$ covariance parameters for the solution variational Gaussian distribution over $N$ random variables.

[Opper and Archambeau, 2009](https://direct.mit.edu/neco/article-abstract/21/3/786/7385/The-Variational-Gaussian-Approximation-Revisited?redirectedFrom=fulltext) demonstrated that assuming a Gaussian prior and a factorized likelihood allows for the optimization of $N$ mean parameters and only $N$ covariance parameters. This significantly enhances the scalability of the method for larger problems. Furthermore, they proposed a new parameterization for the variational Gaussian distribution to achieve these improved results. Notably, their proposed parameterization is akin to the [natural parametrization of Gaussian distributions](https://itskalvik.github.io/NaturalParams), as recently shown by [Khan and Lin 2017](https://arxiv.org/abs/1703.04265) to also result in only $\mathcal{O}(N)$ free variables in this problem context. 

Finally, it's worth noting that Opper and Archambeau considered a full approximation, i.e., they used an $N$-dimensional variational Gaussian distribution to approximate the true posterior distribution of $N$ random variables. However, if we are interested in a [sparse approximation](https://itskalvik.github.io/VFE), considering only an $M$-dimensional variational Gaussian distribution where $M\ll N$, then we will have to optimize the full covariance matrix of the variational distribution. Although in such cases, the difference between $\mathcal{O}(M^2)$ and $\mathcal{O}(N)$ will be significant enough that the full covariance matrix of the sparse approximation would still have fewer variational parameters than a full approximation with a diagonal covariance matrix.

## Relevant Identities 

* $\mathbb{E}[\mathbf{x}\mathbf{x}^\top] = \mathbb{E}[\mathbf{x}]\mathbb{E}[\mathbf{x}]^\top + \text{cov}[\mathbf{x}]$

* $\nabla_{\mathbf{X}} \text{Tr}(\mathbf{A}\mathbf{X}) = \mathbf{A}$

* $\nabla_{\mathbf{X}} \ln \mid\mathbf{X}\mid = \mathbf{X}^{-\top}$

* $\nabla_{\mathbf{X}} \mathbf{X}^{-1} = -\mathbf{X}^{-1} [\nabla_{\mathbf{X}}\mathbf{X}] \mathbf{X}^{-1}$
