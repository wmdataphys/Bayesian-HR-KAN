# [Bayesian (Higher Order) ReLU KAN]()

# Abstract 

Here goes the abstract from the paper...

# Dependencies

Code is built on top of pre-existing works referenced below:

[ReLU KAN](https://github.com/quiqi/relu_kan)
[HRKAN](https://github.com/kelvinhkcs/HRKAN)

We also referenced [Jax KAN](https://github.com/srigas/jaxKAN) in the building of code for solving Partial Differential Equations (PDEs).

Noteable dependencies for reproducibility:

- Pytorch:    Pytorch 2.4.0
- CUDA:       12.1

Dependencies can be emulated by installing the provided conda environemnt:

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

