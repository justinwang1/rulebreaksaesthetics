<<<<<<< HEAD
setwd("C:/Users/jstwa/Desktop/ML/")
source("AesthSetup.R")

#Percentile some features - OPTIONAL
highData$Blur <- to.percentile(highData$Blur)
highData$Size <- to.percentile(highData$Size)
highData$Avg..S <- to.percentile(highData$Avg..S)
highData$Avg..V <- to.percentile(highData$Avg..V)
 
#t tests table for High Data vs. Low Data
results.table(highData,lowData)

##Kernel Density Plot for Aspect 
plot.aspect(highData,lowData)

#Obtain RF Predictions
rf.preds <- cv$predictions$rf; rf.mc <- abs(rf.preds - labels)
highmc <- sort(ids[which(rf.mc == 1 & labels == 1)])
lowmc <- sort(ids[which(rf.mc == 1 & labels == 0)])


##Specific Features

#S
s.idx <- bottom.per(highData$Avg..S,per=0.3); s.ids <- high.ids[s.idx]
rtable.ids(s.ids)

#Blur
blur.idx <- bottom.per(highData$Blur); blur.ids <- high.ids[blur.idx]
rtable.ids(blur.ids)
plot.aspect(badblur.data,goodblur.data)

#Size
size.idx <- bottom.per(highData$Size); size.ids <- high.ids[size.idx]
results.table.ids(size.ids)
plot.aspect(badsize.data,goodsize.data)

#Hue Count
hue.idx <- bottom.per(highData$Hue.Count,direction='negative'); hue.ids <- high.ids[hue.idx]
results.table.ids(hue.ids)
plot.aspect(badhc.data,goodhc.data)

#NN
nn.idx <- which(highData$NN.Hist <= 0.2); nn.ids <- high.ids[nn.idx]
badnnhist.data <- highData[nn.idx,]; goodnnhist.data <- highData[-nn.idx,]
rtable(badnnhist.data,goodnnhist.data)
plot.aspect(badnnhist.data,goodnnhist.data)

#Aspect 4/3
aspect43.idx <- which(highData$Aspect >= 7/6 & highData$Aspect < 17/12); aspect43.ids <- high.ids[aspect43.idx]
aspect43.data <- highData[aspect43.idx,]; noaspect43.data <- highData[-aspect43.idx,]
=======
setwd("C:/Users/jstwa/Desktop/ML/")
source("AesthSetup.R")

#Percentile some features - OPTIONAL
highData$Blur <- to.percentile(highData$Blur)
highData$Size <- to.percentile(highData$Size)
highData$Avg..S <- to.percentile(highData$Avg..S)
highData$Avg..V <- to.percentile(highData$Avg..V)
 
#t tests table for High Data vs. Low Data
results.table(highData,lowData)

##Kernel Density Plot for Aspect 
plot.aspect(highData,lowData)

#Obtain RF Predictions
rf.preds <- cv$predictions$rf; rf.mc <- abs(rf.preds - labels)
highmc <- sort(ids[which(rf.mc == 1 & labels == 1)])
lowmc <- sort(ids[which(rf.mc == 1 & labels == 0)])


##Specific Features

#S
s.idx <- bottom.per(highData$Avg..S,per=0.3); s.ids <- high.ids[s.idx]
rtable.ids(s.ids)

#Blur
blur.idx <- bottom.per(highData$Blur); blur.ids <- high.ids[blur.idx]
rtable.ids(blur.ids)
plot.aspect(badblur.data,goodblur.data)

#Size
size.idx <- bottom.per(highData$Size); size.ids <- high.ids[size.idx]
results.table.ids(size.ids)
plot.aspect(badsize.data,goodsize.data)

#Hue Count
hue.idx <- bottom.per(highData$Hue.Count,direction='negative'); hue.ids <- high.ids[hue.idx]
results.table.ids(hue.ids)
plot.aspect(badhc.data,goodhc.data)

#NN
nn.idx <- which(highData$NN.Hist <= 0.2); nn.ids <- high.ids[nn.idx]
badnnhist.data <- highData[nn.idx,]; goodnnhist.data <- highData[-nn.idx,]
rtable(badnnhist.data,goodnnhist.data)
plot.aspect(badnnhist.data,goodnnhist.data)

#Aspect 4/3
aspect43.idx <- which(highData$Aspect >= 7/6 & highData$Aspect < 17/12); aspect43.ids <- high.ids[aspect43.idx]
aspect43.data <- highData[aspect43.idx,]; noaspect43.data <- highData[-aspect43.idx,]
>>>>>>> 8cc9dedaa868523276b391673922e48859692bb4
results.table(aspect43.data,noaspect43.data)