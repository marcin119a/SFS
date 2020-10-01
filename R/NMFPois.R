#' @import SQUAREM

library(SQUAREM)

#' Function for calculating the generalised Kullback Leibler divergence
#' 
#' Internal function used in NMFPois
#' 
#' @param y Observation
#' @param mu Estimate
#' 
#' @return Generalized Kullback-Leibler
gkl.dev <- function(y, mu){
  r <- mu
  p <- which(y > 0)
  r[p] <- (y * (log(y)- log(mu)) - y + mu)[p]
  return(sum(r))
}

#' @title Non-negative matrix factorization algorithm for Poisson data
#' 
#' Factorizing M into two matrices P and E of
#' dimension ncol(M) x N and N x nrow(M) with the acceleration of SQUAREM.
#' The objective function is the generalized Kullback-Leibler divergence(GKLD).
#' 
#' @param M Non-negative data matrix of size
#' @param N Small dimension of the two new matrices
#' @param seed  Vector of random seeds to initialize the matrices
#' @param arrange Arranging columns in P and rows in E after largest row sums of E 
#' @param tol Maximum change of P and E when stopping 
#'
#' @return A list of the matrices derived by the factorization and the corresponding generalized Kullback-Leibler
#'  \itemize{
#'  \item P   - Non-negative matrix of dimension ncol(V) x K, with columns summing to one
#'  \item E   - Non-negative matrix of dimension K x nrow(V), where rows sum to one 
#'  \item gkl - Smallest Value of the Generalized Kullback-Leibler
#'  }
NMFPois = function(M,N,seed = sample(1:100,3), arrange = TRUE, tol = 1e-5){
  K <- dim(M)[1]  # mutations
  G <- dim(M)[2]  # patients
  
  div <- rep(0,length(seed)) # vector of different GKLD values
  Plist <- list()            # list of P matrices
  Elist <- list()            # list of E matrices
  reslist <- list()
  
  poisson.em = function(x){
    x = exp(x)
    P = matrix(x[1:(K*N)], nrow = K, ncol = N)
    E = matrix(x[-c(1:(K*N))], nrow = N, ncol = G)
    
    PE <- P%*%E
    P <- P * ((M/PE) %*% t(E))      # update of signatures
    P <- P %*% diag(1/rowSums(E))   
    
    PE <- P%*%E
    E <- E * (t(P) %*% (M/PE))      # update of exposures
    E <- diag(1/colSums(P)) %*% E
    
    par = c(as.vector(P),as.vector(E))
    par[par <= 0] = 1e-10
    return(log(par))
  }
  
  gklobj = function(x){
    x = exp(x)
    P = matrix(x[1:(K*N)], nrow = K, ncol = N)
    E = matrix(x[-c(1:(K*N))], nrow = N, ncol = G)
    
    GKL <- gkl.dev(as.vector(M),as.vector(P%*%E)) # GKLD value
    
    return(GKL)
  }
  
  for(i in 1:length(seed)){ 
    set.seed(seed[i])
    
    P <- matrix(stats::runif(K*N), nrow = K, ncol = N)  # Initialize P
    E <- matrix(stats::runif(N*G), nrow = N, ncol = G)  # Initialize E
    
    init = log(c(as.vector(P),as.vector(E)))
    sres = squarem(init, fixptfn = poisson.em, objfn = gklobj, control = list(tol = tol))
    
    P = matrix(exp(sres$par[1:(K*N)]), nrow = K, ncol = N)
    E = matrix(exp(sres$par[-c(1:(K*N))]), nrow = N, ncol = G)
    E = diag(colSums(P)) %*% E # normalizing 
    P = P %*% diag(1/colSums(P))
    
    Plist[[i]] <- P # signatures
    Elist[[i]] <- E # exposures
    div[i] <- gklobj(sres$par)   # final generalized Kullback-Leibler divergence
    reslist[[i]] = sres
  }
  
  best <- which.min(div) # Smallest GKLD value
  P = Plist[[best]]
  E = Elist[[best]]
  
  if(arrange == TRUE){
    idx = order(rowSums(E),decreasing = TRUE)
    P = P[,idx]
    E = E[idx,]
  }
  
  Output <- list()
  Output$P <-  P
  Output$E <-  E
  Output$gkl <- div[best]
  
  return(Output)
}