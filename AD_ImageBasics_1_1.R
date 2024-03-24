#=========== Loading packages and image, re-sizing ===========================
library(easypackages)
libraries("imager","tidyverse")
getwd()
setwd("~/Desktop/Imager_assignment")
temp1=load.image("Duck.jpeg")%>%plot
temp2=resize(temp1,round(width(temp1)/10),round(height(temp1)/10))%>%plot
#=========== Plot the three colors separately ===============================
R(temp1)%>%plot
G(temp1)%>%plot
B(temp1)%>%plot
#=========== Convert to dataframe and plot the values for each color =======
bdf <- as.data.frame(temp1)
bdf <- mutate(bdf,channel=factor(cc,labels=c('R','G','B')))
ggplot(bdf,aes(value,col=channel))+
  geom_histogram(bins=15)+
  facet_wrap(~ channel)
#=========== Create the gradient of an image with respect to both axis =====
gr=imgradient(temp1,"xy")
gr.sq <- gr %>% map_il(~ .^2)
imgradient(temp1,"xy") %>% enorm %>% plot(main="Gradient magnitude")
plot(sqrt(gr.sq$x),main="Gradient magnitude along x")
plot(sqrt(gr.sq$y),main="Gradient magnitude along y")
#========== A similar computation for the gradient =========================
gr.sq <- add(gr.sq) #Add (d/dx)^2 and (d/dy)^2
plot(sqrt(gr.sq))
#========== Computing edges ================================================
edges <- imsplit(gr.sq,"c") %>% add
plot(sqrt(edges),main="Detected edges")
#========== Edges with blurring function ===================================
detect.edges <- function(im,sigma=1)
{
  isoblur(im,sigma) %>% imgradient("xy") %>% enorm %>% imsplit("c") %>% add
}
#========= Computing edges, plotting and creating a "priority map" ========
edges <- detect.edges(temp1,3) %>% sqrt 
plot(edges)
pmap <- 1/(1+edges) #Priority inv. proportional to gradient magnitude
plot(pmap,main="Priority map") #Nice metal plate effect! 
#========= Creating "masks" using watershed ===============================
seeds <- imfill(dim=dim(pmap))%>%plot #Empty image
seeds[200,200,1,1] <- 1 #Background pixel 
seeds[400,400,1,1] <- 2 #Foreground pixel
wt <- watershed(seeds,pmap)
plot(wt,main="Watershed segmentation")
mask <- add.colour(wt) #We copy along the three colour channels
#======== Applying the mask to separate forground from background =========
plot(temp1*(mask==1),main="Background")
plot(temp1*(mask==2),main="Foreground")
temp2=temp1*(mask==2)
#======================== Slight improvement using the blue color only ===
bdf1 <- as.data.frame(B(temp1))
B(temp1)%>%plot
bdf1$value[bdf1$value<0.42]=0
temp3=as.cimg(bdf1)%>% plot
edges <- detect.edges(temp3,2) %>% sqrt 
plot(edges)
pmap <- 1/(1+edges) #Priority inv. proportional to gradient magnitude
plot(pmap,main="Priority map") #Nice metal plate effect! 
seeds <- imfill(dim=dim(pmap)) #Empty image
seeds[200,150,1,1] <- 1 #Background pixel 
seeds[400,250,1,1] <- 2 #Foreground pixel
wt <- watershed(seeds,pmap)
plot(wt,main="Watershed segmentation")
mask <- add.colour(wt) #We copy along the three colour channels
plot(temp1*(mask==1),main="Background")
plot(temp1*(mask==2),main="Foreground")


