# Role-of-Multiplicative-Noise

Multiplicative noise (or parametric noise) is ubiquitous in real world systems and (in contrast to additive noise) is known to play an important role in state transitions, including tipping points where hopping is induced from one attractor to another. The code here is for the study of the effects of parametric (multiplicative) noise in simple models. Scale parameters are used to study the behaviour analytically. It is based on the paper "The stabilizing role of multiplicative noise in non-confining potentials", E. T. Phillips, B. Lindner, H. Kantz, 2025.

I one dimension the general Langevin stochastic differential equation is
$$\dot{x} = f(x) + \sqrt{2D}g(x)\xi(t),$$
where $\xi(t)$ represents Gaussian white noise with mean $\langle \xi(t) \rangle = 0$ and correlation $\langle \xi(t)\xi(t') \rangle = \delta(t-t')$. $f(x)$ and $g(x)$ represent the drift and diffusion terms respectively. When $g(x) \neq$ const. the noise is said to be multiplicative. 

The code here studies a variety of stochastic differential equations, ranging from some of the simplest systems with multiplicative noise to double well potentials and stochastic oscillator problems. 
