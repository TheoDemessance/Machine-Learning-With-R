setwd("~/GitHub/Machine Learning with R")

# Environnement 

## Data structures
library(data.table) 

## Data Visualization
library(ggplot2) 
library(ggthemes) 
library(corrplot) 
library(GGally)

## Model tuning
library(glmnet) # elastic-net regularization package
library(usdm) # to access vifstep package
library(caret) # it makes model tuning easier

# Implementation of basics metrics functions
RSS <- function(Y,Yhat) sum((Yhat-Y)^2)/length(Y) #Residual sum of Squares
R2 <- function(Y,Yhat) 1 - sum((Yhat-Y)^2) / sum((Y-mean(Y))^2) #R² metric, coefficient of determination

# show that you can have a fabulous R2 and a bad result
set.seed(222) 

Nlig<-100 
Ncol<-95

X <- matrix(rnorm(Nlig*Ncol),ncol=Ncol) 
Y <- rnorm(Nlig) 

summary(lm(Y~X))

# Data loading
DT <- fread("data/boston.csv") 
valid <- sample(nrow(DT), 200) # choose a validation set
Y <- log(DT[valid]$medv) # reference value for calculating residues

# Visualization
summary(DT); str(DT)
ggplot(DT, aes(x=medv)) + geom_histogram(fill="blue") + theme_classic()
corrplot(cor(DT)) # identify the direct correlation
ggpairs(DT) # attention calculation time on dataset

# Data Tables manipulation
DT <- copy(DT) #DT2<-DT will only create a new pointer to DT and any action on D2 will also modify DT.
DT[nox>0.5] # filter on DT2 lines, the DT2 object has not been modified.. I can save this result in a new table if necessary: DT3<-DT2 [nox>0.5& lstat<5].
DT[,.(zn,chas,indus)] # column filter: does not change DT2
DT[,.(dis_moyenne=mean(dis)),by=.(zn==0)] # does not change DT2

# calculate median age by nox for zn = 0
DT[zn==0,.(age_med = median(age)), by =.(nox)]

# successive modelizations

## 0. raw model
mod0 <- lm(medv~., DT[-valid]) 
summary(mod0) ; plot(mod0)
Yhat <- predict(mod0, DT[valid]) ; Yhat[Yhat < 1] <- 1 #this last command is a patch: to be able to log and have the same error metric as in the following modelings  
RSS.0 <- RSS(Y, log(Yhat)) # metric of residual edges on test set. log to compare with the following models
R2.0 <- R2(Y, log(Yhat))
Mod.0 <- c(summary(mod0)$r.squared, summary(mod0)$adj.r.squared, summary(mod0)$f[1], mod0$rank, RSS.0, R2.0) #we store the different metric in a vector shape

## 1. normal error distribution
DT1 <- copy(DT) ; DT1[, medv := log(medv)] # instead of modeling medv we will model its log which has a more "normal" distribition.
mod1 <- lm(medv~., DT1[-valid]) ; summary(mod1) ; plot(mod1)
Yhat <- predict(mod1, DT1[valid])
RSS.1 <- RSS(Y, Yhat)
R2.1 <- R2(Y, Yhat)
Mod.1 <- c(summary(mod1)$r.squared, summary(mod1)$adj.r.squared, summary(mod1)$f[1], mod1$rank, RSS.1, R2.1)

## 2. removal of multi-colinearity - as it is an asumption of the LR
temp <- DT1[-valid] ; temp$medv <- NULL
DT2 <- DT1[, c(as.character(vifstep(temp, th=8)@results$Variables), "medv"), with=FALSE] #vifstep allows us to calculate colinearity for all variables and keep only the ones > th
mod2 <- lm(medv~., DT2[-valid]) ; summary(mod2) ; plot(mod2)
Yhat <- predict(mod2, DT2[valid])
RSS.2 <- RSS(Y, Yhat)
R2.2 <- R2(Y, Yhat)
Mod.2 <- c(summary(mod2)$r.squared, summary(mod2)$adj.r.squared, summary(mod2)$f[1], mod2$rank, RSS.2, R2.2)

## 3. added new non-collinear dimensions
DT3 <- copy(DT2) ; DT3[, rm2 := rm^2] ; DT3[, lstat2 := lstat^2]; DT3[, crim2 := crim^2]; DT3[, crimlstat := crim*lstat] #we add new columns non-collinear to the DT
mod3 <- lm(medv~., DT3[-valid,]) ; summary(mod3) ; plot(mod3)
Yhat <- predict(mod3, DT3[valid])
RSS.3 <- RSS(Y, Yhat)
R2.3 <- R2(Y, Yhat)
Mod.3 <- c(summary(mod3)$r.squared, summary(mod3)$adj.r.squared, summary(mod3)$f[1], mod3$rank, RSS.3, R2.3)

## 4. Feature Engineering
train.control <- trainControl(method = "cv", number = 10) #The caret `trainControl()` function is used to set the parameters of the learning process. Here "cv" stands for cross validation and number is the number of folds
step.model <- train(medv ~., data = DT3, method = "leapForward", tuneGrid = data.frame(nvmax = 1:ncol(DT3)), trControl = train.control) # We create a custom model available with Caret package. We use leap forward method with a number of predictors = to the number of columns of DT3
step.model$results ; step.model$bestTune #observe the evolution of rmse and mae
mod4 <- step.model$finalModel
coef <- coef(mod4, step.model$bestTune[[1]]) 
DT4 <- DT3[, c("medv", names(coef)[-1]), with=FALSE]
mod4 <- lm(medv~., DT4[-valid,])
Yhat <- predict(mod4, DT4[valid])
RSS.4 <- RSS(Y, Yhat)
R2.4 <- R2(Y, Yhat)
Mod.4 <- c(summary(mod4)$r.squared, summary(mod4)$adj.r.squared, summary(mod4)$f[1], mod4$rank, RSS.4, R2.4)

# Display of the different models and metrics
results <- as.data.frame(rbind(Mod.0, Mod.1, Mod.2, Mod.3, Mod.4))
names(results) <- c("R2", "R2a", "Fstat", "nbVar", "RSS", "Rval")
ggplot(results, aes(x = nbVar,y = Rval, label =rownames(results))) +
  geom_point(aes(size =Fstat, col=R2)) + theme_classic() + 
  geom_text(hjust=1, vjust=0) + scale_color_continuous("grey", "black")

## 5. Lasso Regularization
DT5 <- copy(DT3)     
DT5[, random := rnorm(nrow(DT5))] ; DT5[, random2 := rnorm(nrow(DT5))] ; DT5[, random3 := rnorm(nrow(DT5))] ; DT5[, random4 := rnorm(nrow(DT5))]
DT5.Y <- DT5$medv
DT5.X <- copy(DT5) ; DT5.X[, medv := NULL] ; DT5.X <- as.matrix(DT5.X); 

### by default alpha=1 
fit = glmnet(DT5.X[-valid,], DT5.Y[-valid]) ; plot(fit, label=TRUE, xvar="lambda") ; print(fit) ; coef(fit, s = 0.1) ; predict(fit, newx=DT5.X[valid,], s = c(0.1,0.05))

### How to choose lambda : cross validation !
cv.fit = cv.glmnet(DT5.X[-valid,], DT5.Y[-valid]) ; plot(cv.fit) ; print(cv.fit) ; coef(cv.fit, s = cv.fit$lambda.1se) ; predict(fit, newx=DT5.X[valid,], s = c(0.1,0.05))

Yhat <- predict(cv.fit, newx = DT5.X[valid,], s = "lambda.min")
RSS.5 <- RSS(DT5.Y[valid], Yhat)
R2.5 <- R2(DT5.Y[valid], Yhat)
Mod.5 <- c(NA, NA, NA, length(which(as.matrix(coef(cv.fit, s = "lambda.min")) != 0)) - 1, RSS.5, R2.5)

### Display of the new model metrics
temp <- rownames(results) ; temp <- c(temp, "Mod.5")
results <- rbind(results, Mod.5) ; rownames(results) <- temp
ggplot(results, aes(x=nbVar, y=Rval, label = rownames(results))) + geom_point() + theme_classic() + geom_text(hjust=1, vjust=0)

## 6. Elastic-Net Regularization
DT6 <- cbind(DT5.X, DT5.Y) 

### definition of a search grid on alpha and beta
lambda.grid <- 10 ^ seq(-2, -6, length=30) ; alpha.grid <- seq(0,1, length=15) ; 
srchGrid <- expand.grid(.alpha = alpha.grid, .lambda = lambda.grid)

### use of the caret functions
trnCtl <- trainControl(method = "repeatedcv", number=10, repeats = 3) # In 3 repeats of 10 fold CV, we'll perform the average of 3 error terms obtained by performing 10 fold CV 3 times. Important thing to note is that 3 repeats of 10 fold CV is not same as 30 fold CV.

### testing of all grid values
my.train <- train(DT5.Y~., data = DT6, method = "glmnet", tuneGrid = srchGrid, trControl = trnCtl, standardize = TRUE) #it fits a LR with a Elastic Net regularization with values in SrchGrid
plot(my.train) # performance visualization for different alpha and beta values.
ggplot(my.train$results, aes(x = log10(lambda), y = Rsquared)) + geom_line(aes(group = alpha, col = alpha)) + theme_classic() + ylim(0.805, 0.808) + xlim(-5, -2)

### attributes(my.train)
my.train$bestTune
my.glmnet.model <- my.train$finalModel
coef(my.glmnet.model, s = my.train$bestTune$lambda)
Yhat <- predict(my.glmnet.model, newx = DT5.X[valid,], s = my.train$bestTune$lambda, alpha = my.train$bestTune$alpha)
RSS.6 <- RSS(DT5.Y[valid], Yhat)
R2.6 <- R2(DT5.Y[valid], Yhat)
Mod.6 <- c(NA, NA, NA, length(which(as.matrix(coef(cv.fit, s = "lambda.min")) != 0)) - 1, RSS.6, R2.6)

temp <- rownames(results) ; temp <- c(temp, "Mod.6")
results <- rbind(results, Mod.6) ; rownames(results) <- temp
ggplot(results, aes(x = nbVar, y = Rval, label = rownames(results))) + geom_point() + theme_classic() + geom_text(hjust = 1, vjust = 0)