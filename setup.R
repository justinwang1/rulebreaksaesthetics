<<<<<<< HEAD
source("util.R")
lowData <- read.table("low.txt"); highData <- read.table("high.txt")
high.ids <- strtoi(row.names(highData)); low.ids <- strtoi(row.names(lowData))

##Bottom Percentage Features IDs
blur.idx <- bottom.per(highData$Blur); blur.ids <- high.ids[blur.idx]
size.idx <- bottom.per(highData$Size); size.ids <- high.ids[size.idx]
hue.idx <- bottom.per(highData$Hue.Count,direction='negative'); hue.ids <- high.ids[hue.idx]
nn.idx <- which(highData$NN.Hist <= 0.25); nn.ids <- high.ids[nn.idx]

##All the Aspects IDs
aspect43.idx <- which(highData$Aspect >= 7/6 & highData$Aspect < 17/12); aspect43.ids <- high.ids[aspect43.idx]
aspect32.idx <- which(highData$Aspect >= 17/12 & highData$Aspect < 7/4); aspect32.ids <- high.ids[aspect32.idx]
aspectwide.idx <- which(highData$Aspect >= 7/4); aspectwide.ids <- high.ids[aspectwide.idx]
aspectsquare.idx <- which(highData$Aspect >= 7/8 & highData$Aspect < 7/6); aspectsquare.ids <- high.ids[aspectsquare.idx]
aspecttall.idx <- which(highData$Aspect < 7/8); aspecttall.ids <- high.ids[aspecttall.idx]


##Properties - Have every property ID List ready to go
propertynames <- c("animals","sil","bw","people","portrait","body","ppl_mult","ppl_other","bkgd","water","mountain",
                "city","bridge","bkgd_other")

setwd("C:/Users/jstwa/Desktop/ava/Property Text Lists/Attributes/")
for(name in propertynames) {  assign(name,read.table(paste0(name,".txt"))$V1)}


#Do something similar for the monochrome backgrounds
colorpropnames <- c("allcolors","black","blue","grayish","green","misccolors","white")
setwd("C:/Users/jstwa/Desktop/ava/Property Text Lists/Colors/")
for(name in colorpropnames) {  assign(name,read.table(paste0(name,".txt"))$V1)}
=======
source("util.R")
lowData <- read.table("low.txt"); highData <- read.table("high.txt")
high.ids <- strtoi(row.names(highData)); low.ids <- strtoi(row.names(lowData))

##Bottom Percentage Features IDs
blur.idx <- bottom.per(highData$Blur); blur.ids <- high.ids[blur.idx]
size.idx <- bottom.per(highData$Size); size.ids <- high.ids[size.idx]
hue.idx <- bottom.per(highData$Hue.Count,direction='negative'); hue.ids <- high.ids[hue.idx]
nn.idx <- which(highData$NN.Hist <= 0.25); nn.ids <- high.ids[nn.idx]

##All the Aspects IDs
aspect43.idx <- which(highData$Aspect >= 7/6 & highData$Aspect < 17/12); aspect43.ids <- high.ids[aspect43.idx]
aspect32.idx <- which(highData$Aspect >= 17/12 & highData$Aspect < 7/4); aspect32.ids <- high.ids[aspect32.idx]
aspectwide.idx <- which(highData$Aspect >= 7/4); aspectwide.ids <- high.ids[aspectwide.idx]
aspectsquare.idx <- which(highData$Aspect >= 7/8 & highData$Aspect < 7/6); aspectsquare.ids <- high.ids[aspectsquare.idx]
aspecttall.idx <- which(highData$Aspect < 7/8); aspecttall.ids <- high.ids[aspecttall.idx]


##Properties - Have every property ID List ready to go
propertynames <- c("animals","sil","bw","people","portrait","body","ppl_mult","ppl_other","bkgd","water","mountain",
                "city","bridge","bkgd_other")

setwd("C:/Users/jstwa/Desktop/ava/Property Text Lists/Attributes/")
for(name in propertynames) {  assign(name,read.table(paste0(name,".txt"))$V1)}


#Do something similar for the monochrome backgrounds
colorpropnames <- c("allcolors","black","blue","grayish","green","misccolors","white")
setwd("C:/Users/jstwa/Desktop/ava/Property Text Lists/Colors/")
for(name in colorpropnames) {  assign(name,read.table(paste0(name,".txt"))$V1)}
>>>>>>> 8cc9dedaa868523276b391673922e48859692bb4
