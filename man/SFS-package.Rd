\name{SFS-package}
\alias{SFS-package}
\alias{SFS}
\docType{package}
\title{
\packageTitle{SFS}
}
\description{
\packageDescription{SFS}
}
\author{
\packageAuthor{SFS}

Maintainer: \packageMaintainer{SFS}
}
\references{
Ragnhild Laursen and Asger Hobolth. A sampling algorithm to compute the set of feasible solutions for non-negative matrix factorization for a general number of components.
}
\examples{
x = matrix(rpois(24,c(10:34)),4,6)
res = NMFPois(x,3)           # Finding an intial NMF solution
sfs = sampleSFS(res$P,res$E) # Finding the SFS
}
