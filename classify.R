<<<<<<< HEAD
#Setup
set.seed(12348)
source("util2.R")
lowData <- read.table("low.txt"); highData <- read.table("high.txt")
lowData <- lowData[-which(!is.finite(lowData$Light.Contrast)),]
high.ids <- strtoi(row.names(highData)); low.ids <- strtoi(row.names(lowData))


#Redo 
redo <- function(data,fname,log=FALSE) {
  x <- data[,fname]; m <- mean(x[which(x != -1)]); x[which(x == -1)] <- m
  if(log) { x <- log(x) }
  data[,fname] <- x; data
}

#Create 
data <- rbind(lowData,highData); data <- data[,-2]
labels <- c(rep(0,dim(lowData)[1]),rep(1,dim(highData)[1]))
ids <- strtoi(c(rownames(lowData),rownames(highData)))


#Scramble 
n <- dim(data)[1]
mix <- sample(1:n,n); data <- data[mix,]; labels <- labels[mix]; ids <- ids[mix]

#Rework Aspect 
data1 <- rework.aspect(data)

#Cross-Validation
cv <- cross.validate(data1,labels,K=5)
errors <- apply(cv$errors,1,mean)
lr <- cv$lr.output
=======
#Setup
set.seed(12348)
source("util2.R")
lowData <- read.table("low.txt"); highData <- read.table("high.txt")
lowData <- lowData[-which(!is.finite(lowData$Light.Contrast)),]
high.ids <- strtoi(row.names(highData)); low.ids <- strtoi(row.names(lowData))


#Redo 
redo <- function(data,fname,log=FALSE) {
  x <- data[,fname]; m <- mean(x[which(x != -1)]); x[which(x == -1)] <- m
  if(log) { x <- log(x) }
  data[,fname] <- x; data
}

#Create 
data <- rbind(lowData,highData); data <- data[,-2]
labels <- c(rep(0,dim(lowData)[1]),rep(1,dim(highData)[1]))
ids <- strtoi(c(rownames(lowData),rownames(highData)))


#Scramble 
n <- dim(data)[1]
mix <- sample(1:n,n); data <- data[mix,]; labels <- labels[mix]; ids <- ids[mix]

#Rework Aspect 
data1 <- rework.aspect(data)

#Cross-Validation
cv <- cross.validate(data1,labels,K=5)
errors <- apply(cv$errors,1,mean)
lr <- cv$lr.output
>>>>>>> 8cc9dedaa868523276b391673922e48859692bb4
print(errors); print(lr)