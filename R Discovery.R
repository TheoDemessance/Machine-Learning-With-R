setwd("~/GitHub/Machine Learning with R")

# Importation of the packages

## Data Structure
require(data.table) 

## Data Visualization
library(ggplot2) 
require(ggthemes) 
library(naniar)
library(GGally)
library(corrplot) 
require(dummies)

# Data Loading 
DT <- fread("data/data immobilier.csv")

# Identify column names starting with a number and add a letter in front of it

FeatDebNum <- names(DT)[grepl("^[[:digit:]]", names(DT))] 
for (f in FeatDebNum) setnames(DT, f, paste0("F_", f)) # make sure that no variable name starts with a number 

# Data Exploration
str(DT) ; summary(DT)
corrplot(cor(DT[, names(which(sapply(DT, class) != "character")), with=FALSE]), method = "ellipse")

# Logarithm effect
(ggplot(DT, aes(x=SalePrice)) + geom_histogram(bins=50,fill="blue") + theme_classic())
(ggplot(DT,aes(x=log(SalePrice))) + geom_histogram(bins=50,fill="blue") + theme_classic())

# the logarithmic approach makes errors comparable regardless of absolute values
# and also has the effect of bringing the distribution closer to a Gaussian, which is important for linear regression approaches.

DT[,SalePrice:=log(SalePrice)]


# Features type
int<-names(DT)[which(sapply(DT, class) %in% c("integer","numeric"))]
char<-names(DT)[which(sapply(DT, class) == "character")]
level<-sort(sapply(DT[,char,with=FALSE], function(x) length(unique(x))))# identifier le nombre de valeur différentes pour les colonnes string

# Mapping NA values
isna <- sort(sapply(DT, function(x) sum(is.na(x))/length(x))) # Percentage of missing values in each column
isna <- isna[isna>0] # We only keep percentages > 0
isnaDT <- data.table(var=names(isna),txna=isna) # We store it as a data table 
isnaDT[, type:="integer"] ; isnaDT[var %in% char,type:="string"] ; # We store it as an int or a string
isnaDT[, var := factor(var,levels=names(isna))] # to order the display
isnaDT[var %in% char,type:="string"] # to differentiate the color according to the type
(ggplot(isnaDT[txna>0],aes(x=var,y=txna))+geom_bar(stat="identity",aes(fill=type))+theme_classic())

# Or with Naniar package, we can have a better plot with 1 function
gg_miss_var(DT) 
gg_miss_upset(DT)

# Observation of correlations
temp <- copy(DT[, c(char, "SalePrice"), with=FALSE]) # We keep our data table with char-typed columns and the Sale Price
temp <- melt.data.table(temp, id.vars = "SalePrice") # In order to wide-to-long reshape
(ggplot(temp, aes(x=value, y=SalePrice)) + geom_violin(fill="blue") + facet_wrap(~variable, scales = "free_x") + theme_classic())
temp <- copy(DT[, int, with=FALSE]) # We keep our data table with int-typed columns and the Sale Price
temp<-melt.data.table(temp,id.vars = "SalePrice")
(ggplot(temp, aes(x=value, y=SalePrice)) + geom_point(col="blue") + facet_wrap(~variable, scales="free_x") + theme_classic())

# Treatment of NA values
DTfull <- copy(DT)
for (c in intersect(names(isna), char)) DTfull[is.na(get(c)), (c) := "ex.na"] # We fill-in the missing char-values by "ex.na" 
for (c in intersect(names(isna), int)) DTfull[is.na(get(c)), (c) := median(DTfull[[c]], na.rm=TRUE)] # We fill-in the missing int-values by the median of the column

# Treatment of infrequent factors 
for (c in char) for (v in names(which(table(DTfull[[c]]) < 15))) DTfull[get(c) == v, (c) := "Other"] #We fill in the infrequent char values (freq < 15) by "Other" 
for (c in char) if(min(table(DT[[c]])) < 40) {temp <- names(head(sort(table(DTfull[[c]])), 2)) ; for (t in temp) DTfull[get(c) == t, (c) := paste(temp, collapse="_")]} # We replace the values < 40 by pasting the 1st and the 2nd most frequent values

# Preparation of the basics
valid <- sample(nrow(DTfull), floor(nrow(DTfull)/3)) # we split by keeping all the columns and 1/3 of the rows 
DTTrain <- DTfull[-valid] ; DTValid <- DTfull[valid] #Train takes the values not in valid (2/3 of the full DT) and valid 1/3 of the full DT
supp <- names(which(sapply(DTTrain[, char, with=FALSE], function(x) length(unique(x))) == 1)) ; for (c in supp) {DTTrain[, (c) := NULL] ; DTValid[, (c) := NULL]}
# We drop the columns when we only have 1 distinct value in the columns
char <- names(DTTrain)[which(sapply(DTTrain, class) == "character")]

DTFactor <- rbind(DTTrain,DTValid) #rbind allows us to merge 2 Data tables
for (c in char) DTFactor[,(c) := as.factor(get(c))] #we set the char values as a factor
DTTrainRF <- DTFactor[1 : nrow(DTTrain)] 
DTValidRF <- DTFactor[(nrow(DTTrain) + 1) : nrow(DTFactor)] #We split again our 2 DT using different method

DTMatrice <- rbind(DTTrain,DTValid)
DTMatrice <- dummy.data.frame(DTMatrice) #Create a DT of dummies 
DTTrainMat <- as.matrix(DTMatrice[1:nrow(DTTrain),]) #and we split as a Train and Validation matrix
DTValidMat <- as.matrix(DTMatrice[(nrow(DTTrain)+1):nrow(DTMatrice),]) 

