# Code for the Trade Policy class at PSE
# Author: Michal Miktus at michal.miktus@gmail.com
# Date: 23.02.2019


path <- '/Users/miktus/Documents/PSE/Trade policy/Model/'
# path <- 'C:/Repo/Trade/Trade-policy/'

setwd(path)
set.seed(12345)

# Load packages -----------------------------------------------------

list.of.packages <- c("readstata13", "data.table", "gravity", "dplyr", 'stargazer', 'caret')

new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages, repos = "http://cran.us.r-project.org")

invisible(lapply(list.of.packages, library, character.only = TRUE))

# Useful functions

RMSE = function(m, o){
  sqrt(mean((m - o)^2, na.rm=TRUE))
}

# Load the data ----------------------------------------------------

data <- fread(paste0(path,"Data/data_PL.csv"))
names(data) <- make.names(names(data), unique=TRUE)

# Year variable

year <- data[, 'yr']

# Near zero variance variables

near <- nearZeroVar(data)
data <- data[, -near, with = FALSE]

# Remove highly correlated data

corr = cor(data)
hc = findCorrelation(corr, cutoff=0.8) # put any value as a "cutoff" 
hc = sort(hc)
data = data[, -hc, with = FALSE]

# Add year (just for splitting)

data[, yr := year]

# Data split to compare the reults

data_bef2010 <- data[yr <= 2010]
# data_bef2010[, yr := NULL]
data_aft2010 <- data[yr > 2010]
# data_aft2010[, yr := NULL]
data_aft2010[, dist_log := log(distw)]
var <- setdiff(names(data_bef2010), c("Trade_value_total", "distw", "V1", "yr"))

# PPML: Poisson Pseudo Maximum Likelihood

PPML <- ppml(dependent_variable= "Trade_value_total", distance="distw", additional_regressors = var, es = T, robust=TRUE, data = data_bef2010)
summary(PPML)
predictions <- predict(PPML, newdata = data_aft2010)
residuals <- predictions - data_aft2010[,"Trade_value_total"]
MSE <- mean(sum(residuals^2)/length(unlist(residuals)))
max(unlist(residuals))

# Summary to latex

stargazer(PPML)

# FE ----------------------------------------------------

dependent <- c("Trade_value_total")
continous <- c("distw", "pop_o", "pop_d", "gdp_o", "gdp_d", "area_d", "tdiff", "comrelig")
log_variables <- paste("log(",continous, ")", sep = "")
dummies <- setdiff(setdiff(names(data_bef2010), continous), dependent) 

linear_het <- as.formula(paste(paste("log(",dependent, ")", sep = ""),
                               paste(paste(log_variables, collapse = " + "), paste(dummies, collapse = " + "), sep = " + "), sep = " ~ "))

linear_het <- as.formula(paste(dependent,
                               paste(paste(log_variables, collapse = " + "), paste(dummies, collapse = " + "), sep = " + "), sep = " ~ "))

FE <- lm(linear_het, data = data_bef2010)
#FE$coefficients <- lapply(coef(FE), function(x) {ifelse(is.na(x), as.numeric(0), as.numeric(x))})
summary(FE)
MSE_FE_train <- (mean(FE$residuals^2))
MSE_FE_train

predictions <- predict(FE, newdata = data_aft2010)
residuals =  predictions - (data_aft2010[,'Trade_value_total'])
max(residuals)

MSE_FE_test <- (sum(residuals^2)/length(unlist(residuals)))
MSE_FE_test

# Summary to latex

stargazer(FE)

# FE on test

# FE <- lm(linear_het, data = data_aft2010)
# MSE_FE_aft <- mean(sum(FE$residuals^2)/length(FE$residuals))
# MSE_FE_aft


