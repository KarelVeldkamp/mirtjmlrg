# mirtjmlrg

Regularized Joint Maximum Likelihood Estimation for High-Dimensional Item Factor Analysis

## Description

This fork contains an adaptation of the [mirtjml code](https://github.com/cran/mirtjml) that allows you to add a regularisation term to the likelihood during training.

The package works exactly the same as the original package with the exception of two new parameters: lambda1 and lambda2. 
These paramters detemine the size of the L1 and L2 penalty terms. When both terms are set to zero, the model will be equivalent to standard mirtjml. 

In thems of code, the important adaptations with respect to the original package consist of updated formulas for the negative log likelihood as well as updated formulas for the gradients of the negative log likelihood with regards to both the person and item parameters. 
