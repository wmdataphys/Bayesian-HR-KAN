# [Bayesian (Higher Order) ReLU KAN](https://arxiv.org/abs/2410.01687)

# Abstract 

We introduce the first method of uncertainty quantification in the domain of Kolmogorov-Arnold Networks, specifically focusing on (Higher Order) ReLUKANs to enhance computational efficiency given the computational demands of Bayesian methods. The method we propose is general in nature, providing access to both epistemic and aleatoric uncertainties. It is also capable of generalization to other various basis functions. We validate our method through a series of closure tests, including simple one-dimensional functions and application to the domain of (Stochastic) Partial Differential Equations. Referring to the latter, we demonstrate the method's ability to correctly identify functional dependencies introduced through the inclusion of a stochastic term.

# Dependencies

Code is built on top of pre-existing works referenced below:

[ReLU KAN](https://github.com/quiqi/relu_kan)

[HRKAN](https://github.com/kelvinhkcs/HRKAN)

[torch-mnf](https://github.com/janosh/torch-mnf)

We also referenced [Jax KAN](https://github.com/srigas/jaxKAN) in the building of code for solving Partial Differential Equations (PDEs).

Noteable dependencies for reproducibility:

- Pytorch:    2.4.0
- CUDA:       12.1

Dependencies can be emulated by installing the provided conda environment:

```
$ conda env create -f rapids.yml
```


# Usage 

## 1D Functions

1D functions can by fit using the following command, where the --lkd argument can be one of "Gauss" or "Student":

```
  $ python fit_functions.py --lkd <Likelihood_type>
```

## Partial Differential Equations

All code for solving of PDEs is in the respective notebooks.

