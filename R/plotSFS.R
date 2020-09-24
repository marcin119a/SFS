#' @import ggplot2
#' @import ggpubr
#' @import stats
require(scales)
library(ggplot2)
library(ggpubr)

#' Plotting the set of feasible solutions
#' 
#' @param sampleSFSOutput The output from the sampleSFS function
#' 
#' @return Plot of the SFS for P and E
plotSFS = function(sampleSFSOutput){
  prob.min = sampleSFSOutput$Pminimum
  prob.max = sampleSFSOutput$Pmaximum
  N = nrow(prob.min)    # the rank
  K = ncol(prob.min)    # the first dimension of the data
  
  # Converting into right data frame
  sig = rep(c(1:N), sampleSFSOutput$totalIter)
  dat1 = data.frame(m = 1:K, S = t(prob.min))
  datmin = reshape(dat1, varying = colnames(dat1)[-1], direction = "long", v.names = "min")
  dat1 = data.frame(m = 1:K, S = t(prob.max))
  datmax = reshape(dat1, varying = colnames(dat1)[-1], direction = "long", v.names = "max")
  data2 = merge(datmin,datmax, by = c("m","time","id"))
  data2$time = factor(data2$time)
  
  equal_breaks <- function(n = 3, s = 0.05, ...){
    function(x){
      # rescaling
      d <- s * diff(range(x)) / (1+2*s)
      seq(min(x)+d, max(x)-d, length=n)
    }
  }
  # Plotting of SFS for P
  g1 = ggplot(data2, aes(x = m, y = min))+
    geom_bar(aes(x = m, y = max), stat = "identity", width = 0.78, fill = "tomato2")+
    geom_bar(stat = "identity", width = 0.8, fill = "black")+
    facet_grid(rows = vars(time), scales = "free", switch = "x")+ 
    theme_bw()+
    theme(text = element_text(size=12, face = "bold"), axis.text.x=element_blank(),axis.text.y = element_text(size = 8),axis.ticks = element_blank(), 
          legend.position = "none",strip.text.y.right = element_text(angle = 0, size = 15), panel.spacing.x = unit(0,"line"),
          strip.background.x = element_rect(color="black", fill="white",linetype="blank"),
          strip.text.x = element_blank())+ 
    ylab("Probability")+xlab(" ")+ggtitle("Entries of P")+
    scale_y_continuous(labels = scales::percent_format(accuracy = 1), breaks=equal_breaks(n=3, s=0.2), 
                       expand = c(0.05, 0))
  
  # Converting into right data frame
  expos.min = sampleSFSOutput$Eminimum%*%diag(1/colSums(sampleSFSOutput$E_lastCheckResults[1:N,]))
  expos.max = sampleSFSOutput$Emaximum%*%diag(1/colSums(sampleSFSOutput$E_lastCheckResults[1:N,]))
  G = ncol(expos.min)
  m = c(1:G)
  dat1 = data.frame(m, S = t(expos.min))
  datmin = reshape(dat1, varying = colnames(dat1)[-1], direction = "long", v.names = "min")
  dat1 = data.frame(m , S = t(expos.max))
  datmax = reshape(dat1, varying = colnames(dat1)[-1], direction = "long", v.names = "max")
  
  dat2 = merge(datmin,datmax, by = c("m","time","id"))
  dat2$time = factor(dat2$time)
  
  # Plotting the SFS for E
  g2 = ggplot(dat2, aes(x = m, y = min))+
    geom_bar(aes(x = m, y = max), stat = "identity", width = 0.78, fill = "tomato2")+
    geom_bar(stat = "identity", width = 0.8, fill = "black")+
    facet_grid(cols = vars(time))+
    theme_bw()+
    theme(text = element_text(size=12, face = "bold"), axis.text.x=element_text(angle = 315, size = 5, hjust = 0.7, vjust = -0.6), axis.text.y = element_blank(),
          axis.ticks.y = element_blank(), legend.position = "none",strip.text.y.right = element_text(angle = 0))+ 
    ylab("Probability")+xlab(" ")+ggtitle("   Normalized entries of E")+
    scale_y_continuous(labels = scales::percent_format(accuracy = 1), breaks = c(0,0.5,1))+
    coord_flip()
  finalplot = ggarrange(g1,g2, labels = c("(A)","(B)"), widths = c(K/2, N*7)) 
  return(finalplot)
  
}