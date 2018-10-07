<<<<<<< HEAD
###Types of Functions
#1. Input/Output
#2. Dummies and ID Lists
#3. Feature Functions + Classification
#4. Tables and Charts
#5. Plotting

#######################################################################
###Input/Output Functions - Copying, Transfers, and Working with Folders
#######################################################################

#Sort IDs from smallest to largest by Feature fname
sort.copy <- function(ids,fname,savedir,write.txt=TRUE) {
  setwd("C:/Users/jstwa/Desktop/ava/")
  
  ids1 <- ids[which(ids %in% high.ids)]
  if(length(ids1) < length(ids)) { 
    print(paste0("Notice: ",length(ids)-length(ids1)," rows were removed to conform to High IDs."))
  }
  ids <- ids1
  
  ft <- get.rows(ids)[,fname]
  ids.ordered <- ids[order(ft)]; ft.ordered <- ft[order(ft)]
  
  create.dir(savedir)
  for(i in 1:length(ids.ordered)) {
    curr.id <- ids.ordered[i]
    old <- paste0("Renumbered Data/high/",curr.id,".jpg")
    new <- paste0(savedir,"/R",i,"-",curr.id,".jpg")
    file.copy(old,new,overwrite = FALSE, recursive = FALSE, copy.mode = TRUE)
  }
  
  if(write.txt) {
    flist <- mapply(function(i,x,y) { paste0(i,"   ",x,"   ",y) },1:length(ids.ordered),ids.ordered,ft.ordered)
    flist <- c(c("Rank  ID  Feature"),flist)
    fconn <- file(paste0(savedir,"list.txt"))
    writeLines(flist,fconn)
    close(fconn)
  }
}

#Order IDs by feature, supplement to sort.copy()
order.ft <- function(ids,fname) {
  setwd("C:/Users/jstwa/Desktop/ava/")
  ft <- get.rows(ids)[,fname]
  ids.ordered <- ids[order(ft)]; ft.ordered <- ft[order(ft)]
  ids.ordered
}

#Copy entire index list to targetdir
copy.ava <- function(indexlist,targetdir,origindir="C:/Users/jstwa/Desktop/ava/Renumbered Data/high/") {
  setwd("C:/Users/jstwa/Desktop/ava/")
  create.dir(targetdir)
  filesToCopy <- sapply(indexlist,function(x) { paste0("/",x,".jpg") })
  for(i in 1:length(filesToCopy)) {
    currImg <- filesToCopy[i]
    file.copy(paste0(origindir,currImg),paste0(targetdir,currImg),overwrite = FALSE, recursive = FALSE, copy.mode = TRUE)
  }
}

copy.nbhd <- function(indexlist,destdir) {
  
  copy.one <- function(index) {
    ninfo <- read.table(paste0("Features Data/highhistnbhd/",toString(index),".txt"))
    hnbrs <- ninfo$V1[which(ninfo$V2 == 1)]; lnbrs <- ninfo$V1[which(ninfo$V2 == 0)] 
    numdir <- paste0(destdir,toString(index),"/")
    HDir <- paste0(numdir,"H/"); LDir <- paste0(numdir,"L/")
    dir.create(numdir); dir.create(HDir); dir.create(LDir)
    copy.ava(c(index),numdir,"Renumbered Data/high/")
    copy.ava(hnbrs,HDir,"Renumbered Data/high/")
    copy.ava(lnbrs,LDir,"Renumbered Data/low/")
  }
  
  avadir <- "C:/Users/jstwa/Desktop/ava/"; setwd(avadir)
  create.dir(destdir)
  
  for(i in 1:length(indexlist)) { 
    currnum <- indexlist[i]
    copy.one(currnum)
  }
}

#Get numeric vector of all ID #s contained in a folder of pictures
list.folder <- function(dir) {
  setwd("C:/Users/jstwa/Desktop/ava/")
  flist <- list.files(dir)
  is.picture <- unname(sapply(flist,grepl,pattern=".jpg"))
  flist <- flist[-which(is.picture==0)]
  numlist <- sort(strtoi(unlist(strsplit(flist,".jpg"))))
  numlist
}

#Copy all bottom percentages to folders
copy.multiple <- function(folderids,foldernames) {
  setwd("C:/Users/jstwa/Desktop/ava/")
  for(i in 1:length(foldernames)) { 
    targetdir <- file.path(basedir,foldernames[i])
    copy.ava(folderids[[i]],targetdir)
  }
}

create.dir <- function(dir,overwrite=FALSE) { 
  if(!dir.exists(dir)) { 
    dir.create(dir) 
    cat("New folder created: ",dir,"\n")
  } else {
    cat("Note: Directory already exists.","\n")
  }
}



#######################################################################
###Working with Dummies and ID Lists
#######################################################################

dum.to.ids <- function(dummies) { high.ids[which(dummies == 1)]  } #Convert dummy list to their IDs
rmv <- function(idlist,rm) { idlist[-which(idlist %in% rm)] } #Remove all IDs in rm from idlist
ids.to.idx <- function(ids) { which(high.ids %in% ids) } #Get row indices corresponding to IDs
get.rows <- function(ids) { highData[ids.to.idx(ids),] } #Get rows corresponding to an ID list
cross <- function(id1,id2) { id1[which(id1 %in% id2)] } #Get all IDs in C1 that are also C2
prop <- function(id1,id2) { length(cross(id1,id2))/length(id1) } #What proportion of C1s also exhibit C2?
ptest <- function(a1,a2,b1,b2) #Does the proportion of a1 in a2 differ from b1 in b2?
{ x <- c(length(a1),length(b1)); n <- c(length(a2),length(b2)); prop.test(x,n) } 
compare.ft <- function(id1,id2,fname,parametric=TRUE,plot=TRUE)  #t test of Feature fname among pictures of id1 and id2
{
  rets <- list()
  f1 <- get.rows(id1)[,fname]; f2 <- get.rows(id2)[,fname]
  if(plot) { 
    par(mfrow=c(2,2)); kdplot(f1,f2); boxplot(f1,f2,main="Boxplots",names=c("Pop 1","Pop 2")) 
    hist(f1,main="Hist Pop 1",breaks=20); hist(f2,main="Hist Pop 2",breaks=20)
  }
  rets[[1]] <- summary(f1); rets[[2]] <- summary(f2)
  if(parametric) { rets[[3]] <- t.test(f1,f2) }
  else { rets[[3]] <- wilcox.test(f1,f2) }
  rets
}



#######################################################################
###Feature Functions - Working with the Features, Classification, etc.
#######################################################################

assign.stars <- function(x) {
  if(x < 0.001) {
    return(1111)
  } else if(x >= 0.001 & x < 0.01 ) {
    return(111)
  } else if(x >= 0.01 & x < 0.05) {
    return(11)
  } else if(x >= 0.05 & x < 0.1) {
    return(1)
  } else {
    return(0)
  }
}

bottom.per <- function(x,per=0.1,direction='positive') {
  if(direction != 'positive') { x = -x }
  which(x <= quantile(x,probs=per))
}

cross.validate <- function(data,labels,K=5) {
  
  #Setup: Construct indices for Cross-Validation
  library(randomForest); library(e1071)
  n <- dim(data)[1]; iters <- 1:K
  cv.size <- floor(n/K); stops <- cv.size*iters; stops[length(stops)] <- n+1; stops <- c(1,stops)
  lr.output.list = list()
  lr.preds <- rep(-1,n); rf.preds <- rep(-1,n); svm.preds <- rep(-1,n)
  lr.errors <- rep(-1,K); rf.errors <- rep(-1,K); svm.errors <- rep(-1,K)
  
  #Loop over K pieces
  for(k in iters) {
    
    #Create Current Epoch Train and Test Data
    cat("Epoch ",k,"\n")
    start <- stops[k]; stop <- stops[k+1]-1
    indices.test <- start:stop; indices.train <- (1:n)[-indices.test]
    data.train <- data[indices.train,]; labels.train <- labels[indices.train]
    data.test <- data[indices.test,]; labels.test <- labels[indices.test]
    
    #Train and Test Logistic Regression
    ctrain <- data.frame(cbind(labels.train,data.train)); colnames(ctrain) <- c("labels.train",colnames(data.train))
    lr <- glm(as.factor(labels.train)~.,data=ctrain,family=binomial(link='logit'))
    lr.probs <- predict(lr,newdata=as.data.frame(data.test),type='response')
    lr.pred <- ifelse(lr.probs > 0.5, 1, 0)
    lr.errors[k] <- sum(abs(lr.pred-labels.test))/length(labels.test)
    lr.preds[indices.test] <- lr.pred
    
    #Store LogReg results
    lr.output.list[[k]] <- summary(lr)$coefficients
    
    #Train and Test Random Forest
    rf <- randomForest(as.factor(labels.train)~.,data=data.train)
    rf.pred <- as.numeric(as.character(predict(rf,data.test)))
    rf.errors[k] <- sum(abs(rf.pred-labels.test))/length(labels.test)
    rf.preds[indices.test] <- rf.pred
    
    #Train and Test SVM
    ctrain <- data.frame(cbind(labels.train,data.train))
    ctest <- data.frame(cbind(labels.test,data.test))
    svm1 <- svm(labels.train~.,ctrain)
    svm.probs <- predict(svm1,ctest)
    svm.pred <- unname(ifelse(svm.probs > 0.5, 1, 0))
    svm.errors[k] <- sum(abs(svm.pred-labels.test))/length(labels.test)
    svm.preds[indices.test] <- svm.pred
  }
  
  #Combine Logisic Regression Output
  combined <- (Reduce("+",lr.output.list))/K
  Signif <- unname(sapply(combined[,4],assign.stars))
  lr.output <- round(cbind(combined,Signif),digits=6)
  
  errors <- rbind(lr.errors,rf.errors,svm.errors)
  predictions <- list(lr=lr.preds,rf=rf.preds,svm=svm.preds)
  list(errors=errors,predictions=predictions,lr.output=lr.output)
}

rework.aspect <- function(data) {
  a.indx <- which(names(data) %in% c("Aspect"))
  stopifnot(length(a.indx) >= 1)
  
  cutoffs <- c(7/8,7/6,17/12,7/4)
  Aspect.34 <- ifelse(data$Aspect <= cutoffs[1],1,0)
  Aspect.square <- ifelse(data$Aspect > cutoffs[1] & data$Aspect <= cutoffs[2],1,0)
  Aspect.43 <- ifelse(data$Aspect > cutoffs[2] & data$Aspect <= cutoffs[3],1,0)
  Aspect.32 <- ifelse(data$Aspect > cutoffs[3],1,0)
  data <- cbind(data[,-a.indx],Aspect.34,Aspect.square,Aspect.32)
  data
}

to.percentile <- function(x)  { 
  round(trunc(rank(x))/length(x),3)
}


#######################################################################
###Tables and Charts of Results
#######################################################################

rtable <- function(sub1,sub2,round.place=8) {
  ftlist <- colnames(sub1)
  
  results.table <- matrix(0,length(ftlist),3)
  colnames(results.table) <- c("t statistic","p value","Significance")
  rownames(results.table) <- ftlist
  for(i in 1:length(ftlist)) {
    ftname <- ftlist[i]
    f1 <- sub1[,ftname]; f2 <- sub2[,ftname]
    ttest <- t.test(f1,f2); p.value <- ttest$p.value
    results.table[i,1] <- round(ttest$statistic,round.place)
    results.table[i,2] <- round(p.value,round.place)
    
    sg <- assign.stars(p.value)
    results.table[i,3] <- sg
  }
  results.table
}

rtable.ids <- function(ids,idpool=high.ids) {
  posdata <- get.rows(ids); negdata <- get.rows(rmv(idpool,ids))
  rtable(posdata,negdata)
}

rtable.comp.ids <- function(id1,id2) {
  data1 <- get.rows(id1); data2 <- get.rows(id2)
  rtable(data1,data2)
}

per.cross <- function(bad.data,good.data,property) {
  origHighIDs <- strtoi(rownames(highData))
  prop.ids <- origHighIDs[which(property==1)]
  bad.ids <- strtoi(rownames(bad.data)); good.ids <- strtoi(rownames(good.data))
  bad.cross.idx <- which(bad.ids %in% prop.ids); good.cross.idx <- which(good.ids %in% prop.ids)
  x <- c(length(bad.cross.idx),length(good.cross.idx)); n <- c(length(bad.ids),length(good.ids))
  ptest <- prop.test(x,n)
  direction <- ifelse(ptest$estimate[2] < ptest$estimate[1],1,-1)
  test.results <- c(ptest$estimate,ptest$p.value,direction)
  names(test.results) <- c("Bad Prop","Good Prop","p-value","direction")
  
  bad.cross.ids <- bad.ids[bad.cross.idx]
  list(bad.cross.ids=bad.cross.ids,test.results=test.results)
}

props.aspect <- function(aspectvar) {
  cutoffs <- c(0,5/8,7/8,7/6,17/12,7/4,5); iters <- length(cutoffs)-1
  props <- rep(-1,iters)
  
  for(i in 1:iters) {
    lower.bound <- cutoffs[i]; upper.bound <- cutoffs[i+1]
    props[i] <- round(length(which(lower.bound <= aspectvar & aspectvar < upper.bound))/length(aspectvar),3)
  }
  names(props) <- c("Tall","3:4","Square","4:3","3:2","Wide")
  props
}

sp.aspect <- function(aspectvar,value,epsilon=0.05) {
  length(which(aspectvar >= 1.5-epsilon & aspectvar <= 1.5+epsilon))/length(aspectvar)
}


#######################################################################
###Plotting Functions
#######################################################################

plot.aspect <- function(f1,f2) {
  if(!is.null(dim(f1)) & !is.null(dim(f2))) {
    f1 <- f1$Aspect; f2 <- f2$Aspect
  }
  
  
  d1 <- density(f1); d2 <- density(f2)
  max1 <- max(d1$y); max2 <- max(d2$y)
  if(max1 > max2) { 
    plot(d1,col=2,xlim=c(0,3))
    lines(d2,col=4)
  }
  else {
    plot(d2,col=4,xlim=c(0,3))
    lines(d1,col=2)
  }

  cutoffs <- c(5/8,7/8,7/6,17/12,7/4); color.choices <- c(1:4,6)
  for(i in 1:length(cutoffs)) { abline(v=cutoffs[i],lty=4,col=color.choices[i]) }
  legend(2,2.5,c("5/8","7/8","7/6","17/12","7/4"),col=color.choices,lty=2)
}

kdplot <- function(f1,f2) {
  d1 <- density(f1); d2 <- density(f2)
  max.y <- max(max(d1$y),max(d2$y)); max.x <- max(f1,f2)
  plot(d1,col=2,ylim=c(0,max.y),xlim=c(0,max.x),main="Densities, Red = Pop 1"); lines(d2,col=4)
=======
###Types of Functions
#1. Input/Output
#2. Dummies and ID Lists
#3. Feature Functions + Classification
#4. Tables and Charts
#5. Plotting

#######################################################################
###Input/Output Functions - Copying, Transfers, and Working with Folders
#######################################################################

#Sort IDs from smallest to largest by Feature fname
sort.copy <- function(ids,fname,savedir,write.txt=TRUE) {
  setwd("C:/Users/jstwa/Desktop/ava/")
  
  ids1 <- ids[which(ids %in% high.ids)]
  if(length(ids1) < length(ids)) { 
    print(paste0("Notice: ",length(ids)-length(ids1)," rows were removed to conform to High IDs."))
  }
  ids <- ids1
  
  ft <- get.rows(ids)[,fname]
  ids.ordered <- ids[order(ft)]; ft.ordered <- ft[order(ft)]
  
  create.dir(savedir)
  for(i in 1:length(ids.ordered)) {
    curr.id <- ids.ordered[i]
    old <- paste0("Renumbered Data/high/",curr.id,".jpg")
    new <- paste0(savedir,"/R",i,"-",curr.id,".jpg")
    file.copy(old,new,overwrite = FALSE, recursive = FALSE, copy.mode = TRUE)
  }
  
  if(write.txt) {
    flist <- mapply(function(i,x,y) { paste0(i,"   ",x,"   ",y) },1:length(ids.ordered),ids.ordered,ft.ordered)
    flist <- c(c("Rank  ID  Feature"),flist)
    fconn <- file(paste0(savedir,"list.txt"))
    writeLines(flist,fconn)
    close(fconn)
  }
}

#Order IDs by feature, supplement to sort.copy()
order.ft <- function(ids,fname) {
  setwd("C:/Users/jstwa/Desktop/ava/")
  ft <- get.rows(ids)[,fname]
  ids.ordered <- ids[order(ft)]; ft.ordered <- ft[order(ft)]
  ids.ordered
}

#Copy entire index list to targetdir
copy.ava <- function(indexlist,targetdir,origindir="C:/Users/jstwa/Desktop/ava/Renumbered Data/high/") {
  setwd("C:/Users/jstwa/Desktop/ava/")
  create.dir(targetdir)
  filesToCopy <- sapply(indexlist,function(x) { paste0("/",x,".jpg") })
  for(i in 1:length(filesToCopy)) {
    currImg <- filesToCopy[i]
    file.copy(paste0(origindir,currImg),paste0(targetdir,currImg),overwrite = FALSE, recursive = FALSE, copy.mode = TRUE)
  }
}

copy.nbhd <- function(indexlist,destdir) {
  
  copy.one <- function(index) {
    ninfo <- read.table(paste0("Features Data/highhistnbhd/",toString(index),".txt"))
    hnbrs <- ninfo$V1[which(ninfo$V2 == 1)]; lnbrs <- ninfo$V1[which(ninfo$V2 == 0)] 
    numdir <- paste0(destdir,toString(index),"/")
    HDir <- paste0(numdir,"H/"); LDir <- paste0(numdir,"L/")
    dir.create(numdir); dir.create(HDir); dir.create(LDir)
    copy.ava(c(index),numdir,"Renumbered Data/high/")
    copy.ava(hnbrs,HDir,"Renumbered Data/high/")
    copy.ava(lnbrs,LDir,"Renumbered Data/low/")
  }
  
  avadir <- "C:/Users/jstwa/Desktop/ava/"; setwd(avadir)
  create.dir(destdir)
  
  for(i in 1:length(indexlist)) { 
    currnum <- indexlist[i]
    copy.one(currnum)
  }
}

#Get numeric vector of all ID #s contained in a folder of pictures
list.folder <- function(dir) {
  setwd("C:/Users/jstwa/Desktop/ava/")
  flist <- list.files(dir)
  is.picture <- unname(sapply(flist,grepl,pattern=".jpg"))
  flist <- flist[-which(is.picture==0)]
  numlist <- sort(strtoi(unlist(strsplit(flist,".jpg"))))
  numlist
}

#Copy all bottom percentages to folders
copy.multiple <- function(folderids,foldernames) {
  setwd("C:/Users/jstwa/Desktop/ava/")
  for(i in 1:length(foldernames)) { 
    targetdir <- file.path(basedir,foldernames[i])
    copy.ava(folderids[[i]],targetdir)
  }
}

create.dir <- function(dir,overwrite=FALSE) { 
  if(!dir.exists(dir)) { 
    dir.create(dir) 
    cat("New folder created: ",dir,"\n")
  } else {
    cat("Note: Directory already exists.","\n")
  }
}



#######################################################################
###Working with Dummies and ID Lists
#######################################################################

dum.to.ids <- function(dummies) { high.ids[which(dummies == 1)]  } #Convert dummy list to their IDs
rmv <- function(idlist,rm) { idlist[-which(idlist %in% rm)] } #Remove all IDs in rm from idlist
ids.to.idx <- function(ids) { which(high.ids %in% ids) } #Get row indices corresponding to IDs
get.rows <- function(ids) { highData[ids.to.idx(ids),] } #Get rows corresponding to an ID list
cross <- function(id1,id2) { id1[which(id1 %in% id2)] } #Get all IDs in C1 that are also C2
prop <- function(id1,id2) { length(cross(id1,id2))/length(id1) } #What proportion of C1s also exhibit C2?
ptest <- function(a1,a2,b1,b2) #Does the proportion of a1 in a2 differ from b1 in b2?
{ x <- c(length(a1),length(b1)); n <- c(length(a2),length(b2)); prop.test(x,n) } 
compare.ft <- function(id1,id2,fname,parametric=TRUE,plot=TRUE)  #t test of Feature fname among pictures of id1 and id2
{
  rets <- list()
  f1 <- get.rows(id1)[,fname]; f2 <- get.rows(id2)[,fname]
  if(plot) { 
    par(mfrow=c(2,2)); kdplot(f1,f2); boxplot(f1,f2,main="Boxplots",names=c("Pop 1","Pop 2")) 
    hist(f1,main="Hist Pop 1",breaks=20); hist(f2,main="Hist Pop 2",breaks=20)
  }
  rets[[1]] <- summary(f1); rets[[2]] <- summary(f2)
  if(parametric) { rets[[3]] <- t.test(f1,f2) }
  else { rets[[3]] <- wilcox.test(f1,f2) }
  rets
}



#######################################################################
###Feature Functions - Working with the Features, Classification, etc.
#######################################################################

assign.stars <- function(x) {
  if(x < 0.001) {
    return(1111)
  } else if(x >= 0.001 & x < 0.01 ) {
    return(111)
  } else if(x >= 0.01 & x < 0.05) {
    return(11)
  } else if(x >= 0.05 & x < 0.1) {
    return(1)
  } else {
    return(0)
  }
}

bottom.per <- function(x,per=0.1,direction='positive') {
  if(direction != 'positive') { x = -x }
  which(x <= quantile(x,probs=per))
}

cross.validate <- function(data,labels,K=5) {
  
  #Setup: Construct indices for Cross-Validation
  library(randomForest); library(e1071)
  n <- dim(data)[1]; iters <- 1:K
  cv.size <- floor(n/K); stops <- cv.size*iters; stops[length(stops)] <- n+1; stops <- c(1,stops)
  lr.output.list = list()
  lr.preds <- rep(-1,n); rf.preds <- rep(-1,n); svm.preds <- rep(-1,n)
  lr.errors <- rep(-1,K); rf.errors <- rep(-1,K); svm.errors <- rep(-1,K)
  
  #Loop over K pieces
  for(k in iters) {
    
    #Create Current Epoch Train and Test Data
    cat("Epoch ",k,"\n")
    start <- stops[k]; stop <- stops[k+1]-1
    indices.test <- start:stop; indices.train <- (1:n)[-indices.test]
    data.train <- data[indices.train,]; labels.train <- labels[indices.train]
    data.test <- data[indices.test,]; labels.test <- labels[indices.test]
    
    #Train and Test Logistic Regression
    ctrain <- data.frame(cbind(labels.train,data.train)); colnames(ctrain) <- c("labels.train",colnames(data.train))
    lr <- glm(as.factor(labels.train)~.,data=ctrain,family=binomial(link='logit'))
    lr.probs <- predict(lr,newdata=as.data.frame(data.test),type='response')
    lr.pred <- ifelse(lr.probs > 0.5, 1, 0)
    lr.errors[k] <- sum(abs(lr.pred-labels.test))/length(labels.test)
    lr.preds[indices.test] <- lr.pred
    
    #Store LogReg results
    lr.output.list[[k]] <- summary(lr)$coefficients
    
    #Train and Test Random Forest
    rf <- randomForest(as.factor(labels.train)~.,data=data.train)
    rf.pred <- as.numeric(as.character(predict(rf,data.test)))
    rf.errors[k] <- sum(abs(rf.pred-labels.test))/length(labels.test)
    rf.preds[indices.test] <- rf.pred
    
    #Train and Test SVM
    ctrain <- data.frame(cbind(labels.train,data.train))
    ctest <- data.frame(cbind(labels.test,data.test))
    svm1 <- svm(labels.train~.,ctrain)
    svm.probs <- predict(svm1,ctest)
    svm.pred <- unname(ifelse(svm.probs > 0.5, 1, 0))
    svm.errors[k] <- sum(abs(svm.pred-labels.test))/length(labels.test)
    svm.preds[indices.test] <- svm.pred
  }
  
  #Combine Logisic Regression Output
  combined <- (Reduce("+",lr.output.list))/K
  Signif <- unname(sapply(combined[,4],assign.stars))
  lr.output <- round(cbind(combined,Signif),digits=6)
  
  errors <- rbind(lr.errors,rf.errors,svm.errors)
  predictions <- list(lr=lr.preds,rf=rf.preds,svm=svm.preds)
  list(errors=errors,predictions=predictions,lr.output=lr.output)
}

rework.aspect <- function(data) {
  a.indx <- which(names(data) %in% c("Aspect"))
  stopifnot(length(a.indx) >= 1)
  
  cutoffs <- c(7/8,7/6,17/12,7/4)
  Aspect.34 <- ifelse(data$Aspect <= cutoffs[1],1,0)
  Aspect.square <- ifelse(data$Aspect > cutoffs[1] & data$Aspect <= cutoffs[2],1,0)
  Aspect.43 <- ifelse(data$Aspect > cutoffs[2] & data$Aspect <= cutoffs[3],1,0)
  Aspect.32 <- ifelse(data$Aspect > cutoffs[3],1,0)
  data <- cbind(data[,-a.indx],Aspect.34,Aspect.square,Aspect.32)
  data
}

to.percentile <- function(x)  { 
  round(trunc(rank(x))/length(x),3)
}


#######################################################################
###Tables and Charts of Results
#######################################################################

rtable <- function(sub1,sub2,round.place=8) {
  ftlist <- colnames(sub1)
  
  results.table <- matrix(0,length(ftlist),3)
  colnames(results.table) <- c("t statistic","p value","Significance")
  rownames(results.table) <- ftlist
  for(i in 1:length(ftlist)) {
    ftname <- ftlist[i]
    f1 <- sub1[,ftname]; f2 <- sub2[,ftname]
    ttest <- t.test(f1,f2); p.value <- ttest$p.value
    results.table[i,1] <- round(ttest$statistic,round.place)
    results.table[i,2] <- round(p.value,round.place)
    
    sg <- assign.stars(p.value)
    results.table[i,3] <- sg
  }
  results.table
}

rtable.ids <- function(ids,idpool=high.ids) {
  posdata <- get.rows(ids); negdata <- get.rows(rmv(idpool,ids))
  rtable(posdata,negdata)
}

rtable.comp.ids <- function(id1,id2) {
  data1 <- get.rows(id1); data2 <- get.rows(id2)
  rtable(data1,data2)
}

per.cross <- function(bad.data,good.data,property) {
  origHighIDs <- strtoi(rownames(highData))
  prop.ids <- origHighIDs[which(property==1)]
  bad.ids <- strtoi(rownames(bad.data)); good.ids <- strtoi(rownames(good.data))
  bad.cross.idx <- which(bad.ids %in% prop.ids); good.cross.idx <- which(good.ids %in% prop.ids)
  x <- c(length(bad.cross.idx),length(good.cross.idx)); n <- c(length(bad.ids),length(good.ids))
  ptest <- prop.test(x,n)
  direction <- ifelse(ptest$estimate[2] < ptest$estimate[1],1,-1)
  test.results <- c(ptest$estimate,ptest$p.value,direction)
  names(test.results) <- c("Bad Prop","Good Prop","p-value","direction")
  
  bad.cross.ids <- bad.ids[bad.cross.idx]
  list(bad.cross.ids=bad.cross.ids,test.results=test.results)
}

props.aspect <- function(aspectvar) {
  cutoffs <- c(0,5/8,7/8,7/6,17/12,7/4,5); iters <- length(cutoffs)-1
  props <- rep(-1,iters)
  
  for(i in 1:iters) {
    lower.bound <- cutoffs[i]; upper.bound <- cutoffs[i+1]
    props[i] <- round(length(which(lower.bound <= aspectvar & aspectvar < upper.bound))/length(aspectvar),3)
  }
  names(props) <- c("Tall","3:4","Square","4:3","3:2","Wide")
  props
}

sp.aspect <- function(aspectvar,value,epsilon=0.05) {
  length(which(aspectvar >= 1.5-epsilon & aspectvar <= 1.5+epsilon))/length(aspectvar)
}


#######################################################################
###Plotting Functions
#######################################################################

plot.aspect <- function(f1,f2) {
  if(!is.null(dim(f1)) & !is.null(dim(f2))) {
    f1 <- f1$Aspect; f2 <- f2$Aspect
  }
  
  
  d1 <- density(f1); d2 <- density(f2)
  max1 <- max(d1$y); max2 <- max(d2$y)
  if(max1 > max2) { 
    plot(d1,col=2,xlim=c(0,3))
    lines(d2,col=4)
  }
  else {
    plot(d2,col=4,xlim=c(0,3))
    lines(d1,col=2)
  }

  cutoffs <- c(5/8,7/8,7/6,17/12,7/4); color.choices <- c(1:4,6)
  for(i in 1:length(cutoffs)) { abline(v=cutoffs[i],lty=4,col=color.choices[i]) }
  legend(2,2.5,c("5/8","7/8","7/6","17/12","7/4"),col=color.choices,lty=2)
}

kdplot <- function(f1,f2) {
  d1 <- density(f1); d2 <- density(f2)
  max.y <- max(max(d1$y),max(d2$y)); max.x <- max(f1,f2)
  plot(d1,col=2,ylim=c(0,max.y),xlim=c(0,max.x),main="Densities, Red = Pop 1"); lines(d2,col=4)
>>>>>>> 8cc9dedaa868523276b391673922e48859692bb4
}