# [Bayesian (Higher Order) ReLU KAN]()

# Abstract 

Here goes the abstract from the paper...

# Dependencies

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

All code for solving of Partial Differential Equations (PDEs) is in the respective notebooks.

