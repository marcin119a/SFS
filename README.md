---
title: "SFS package"
author: "Ragnhild Laursen"
date: "09/11/2020"
output: markdown::html_vignette
vignette: >
  %\VignetteIndexEntry{introduction.Rmd}
  %\VignetteEngine{knitr::rmarkdown}
  \usepackage[utf8]{inputenc}
---
# Introduction
This package is developed for a new sampling algorithm to find the set of feasible solutions(SFS) from an initial solution of non-negative matrix factorization(NMF). Remember, non-negative matrix factorization takes a non-negative matrix $M(K \times G)$ and approximates it by two other non-negative matrices $P(K \times N)$ and $E(N \times G)$ such that
\begin{equation*}
M \approx PE.
\end{equation*}
Other solutions with the same approximation could be construct with an invertible matrix $A(N \times N)$ such that 
\begin{equation*}
    \tilde{P} = PA \geq 0 \quad \tilde{E} = A^{-1}E \geq 0,
\end{equation*}
are new solutions. There exist trivial ambiguities where $A$ is either a diagonal matrix or a permutation matrix, but besides these trivial ambiguities others could exist as well. The scaling ambiguity is removed by assuming the columns of $P$ sum to one. The goal of the main function \texttt{sampleSFS()} in this package is to approximate the whole SFS that exist for $P$ and $E$ besides the ambiguities. The advantage of this algorithm is that is has a simple implementation and can be applied for an arbitrary dimension of $N$. A further desciption can be found in the corresponding paper \textit{R. Laursen and A. Hobolth, A sampling algorithm to compute the set of feasible solutions for non-negative matrix factorization for an arbitrary rank.}.

The package includes the following functions:

- `sampleSFS` is the main function that can find the SFS from an initial NMF
- `NMFPois` can find an initial NMF solution from a data matrix
- `gkl.dev` is an internal function for `NMFPois`, that calculates the generalized Kullback-Leibler
- `plotSFS` will plot the SFS  
- `samplesToSVD` will transform SFS solutions from `sampleSFS` relative to SVD solution

# Workflow of the package

## Installation 

The following packages are used in the package \textbf{SFS} and do therefore need to be installed.
```r
install.packages("devtools")
install.packages("SQUAREM")
install.packages("ggplot2")
install.packages("Rcpp")
install.packages("RcppArmadillo")
devtools::install_github("ragnhildlaursen/SFS")
```

The most simple way to install the package is using the package \textbf{devtools}.

```r
library(devtools)
library(SQUAREM)
library(ggplot2)
library(Rcpp)
library(RcppArmadillo)
library(SFS)
```

## Example of how to use functions
To illustrate the functions let us assume we have given a matrix of data $M (4 \times 6)$
```r

M = matrix(c(20, 3, 24, 19,  2, 15, 
             9, 14, 25, 30, 15, 10,
             30, 6, 12, 10, 11,  7,
             9, 27, 5, 11, 19, 15),
           nrow = 4, ncol = 6)
```

First, we need to create an initial NMF solution which is made using the function \texttt{NMFPois}. The input for this function is a matrix $M$ and a rank $N$, that we here choose to be $3$.

```r
initial.fit = NMFPois(M,3)
initial.fit$P
initial.fit$E
initial.fit$P%*%initial.fit$E #approximation of M
```

Now, as an initial solution has been constructed one can find the $SFS$ with the function \texttt{sampleSFS}. Here, we just need the initial solutions of $P$ and $E$. 

```r
sfs.result = sampleSFS(initial.fit$P,initial.fit$E) 

```

The results of the SFS can now be illustated by the function \texttt{plotSFS} function by setting the whole output from sampleSFS as input.

```r
plots = plotSFS(sfs.result)
plots$Pplot
plots$Eplot
```











