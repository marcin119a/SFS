#' Finding SVD representation from solutions of matrices P and E
#'
#' @param Presults Matrix of results of P transposed stacked on top of each other. Dimension is (N*results x nrow(P)).
#' @param Eresults Matrix of results of E stacked on top of each other. Dimension is (N*results x ncol(E))
#' @param Mfit The initial factorization of P and E to use as a reference for the eigenvectors. 
#'  Default is the factorization of the first matrix in Presults and Eresults.
#' @param N The rank of the factorization
#' 
#' @return The SVD representation of the set of feasible solutions
#' \itemize{
#'  \item P.points - Matrix of P results as SVD solution (results x (N-1)).
#'  \item E.points - Matrix of E results as SVD solution (results x (N-1)).
#'  }
samplesToSVD = function(Presults, Eresults, N, Mfit = t(Presults[1:N,])%*%Eresults[1:N,]){
  svd.Mfit = svd(t(Mfit), nu = N, nv = N)
  svdV = svd.Mfit$v
  svdU = svd.Mfit$u
  P.points = matrix(0, nrow = nrow(Presults), ncol = N-1)
  E.points = matrix(0, nrow = nrow(Eresults), ncol = N-1)
  for(i in 1:(nrow(Presults)/N)){
    p = Presults[(i*N-(N-1)):(i*N),]
    Tmat = p%*%svdV  # find Tmat
    Tmat = Tmat/Tmat[,1]
    P.points[(i*N-(N-1)):(i*N),] = Tmat[,c(2:N)]
    e = Eresults[(i*N-(N-1)):(i*N),]
    Tmat = e%*%svdU  # find Tmat
    Tmat = Tmat/Tmat[,1]
    E.points[(i*N-(N-1)):(i*N),] = Tmat[,c(2:N)]
  }
  
  Output = list()
  Output$P.points = P.points
  Output$E.points = E.points
  return(Output)
}
