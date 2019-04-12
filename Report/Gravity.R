# Code for the Trade Policy class at PSE
# Author: Michal Miktus at michal.miktus@gmail.com
# Mateusz Szmidt at mateuszszmidt95@gmail.com
# Date: 23.02.2019


#path <- '/Users/miktus/Documents/PSE/Trade policy/Model/'
path <- 'C:/Repo/Trade/Trade-policy/'

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
distance <- data[, 'distw']
flow <- data[, "Trade_value_total"]
data_bef2010 <- data[yr <= 2010]

# Near zero variance variables

near <- nearZeroVar(data_bef2010, freqCut = 300/1)
data <- data[, -near, with = FALSE]

# Remove highly correlated data

corr = cor(data)
hc = findCorrelation(corr, cutoff=0.30) # put any value as a "cutoff" 
hc = sort(hc)
data = data[, -hc, with = FALSE]

# Add year and other variables which are crucial for the PPML (just for splitting)

data[, yr := year]
data[, distw := distance]
data[, Trade_value_total := flow]
# Data split to compare the reults

data_bef2010 <- data[yr <= 2010]
data_bef2010[, yr := NULL]
data_aft2010 <- data[yr > 2010]
data_aft2010[, yr := NULL]
data_aft2010[, dist_log := log(distw)]

colinear = c("pt3ISO_ABW","yr_2010", "yr_2009","yr_2003", "yr_2008", "flaggsp_o_d_no.gsp.recorded.in.Rose", "legnew_d_uk")
var <- setdiff(names(data_bef2010), c("Trade_value_total", "distw", "V1", colinear))

# PPML: Poisson Pseudo Maximum Likelihood

PPML <- ppml(dependent_variable= "Trade_value_total", distance="distw", additional_regressors = var, robust=TRUE, data = data_bef2010)
summary(PPML)

predictions <- predict(PPML, newdata = data_aft2010, type="response", se.fit=T)

residuals <- predictions$se.fit
MSE <- mean(sum(residuals^2)/length(unlist(residuals)))
(MSE)/var(data$Trade_value_total)

# Summary to latex



(summary(PPML))

# FE ----------------------------------------------------
# Left just in case - to be removed in final version
fe <- F
#
if (fe){
  data <- fread(paste0(path,"Data/data_PL.csv"))
  names(data) <- make.names(names(data), unique=TRUE)
  
  # Year variable
  
  year <- data[, 'yr']
  distance <- data[, 'distw']
  flow <- data[, "Trade_value_total"]
  data_bef2010 <- data[yr <= 2010]
  
  # Near zero variance variables
  
  near <- nearZeroVar(data_bef2010, freqCut = 1000/1)
  data <- data[, -near, with = FALSE]
  
  # Remove highly correlated data
  
  corr = cor(data)
  hc = findCorrelation(corr, cutoff=0.90) # put any value as a "cutoff" 
  hc = sort(hc)
  data = data[, -hc, with = FALSE]
  
  # Add year (just for splitting)
  
  data[, yr := year]
  data[, distw := distance]
  data[, Trade_value_total := flow]
  # Data split to compare the reults
  
  data_bef2010 <- data[yr <= 2010]
  data_bef2010[, yr := NULL]
  data_aft2010 <- data[yr > 2010]
  data_aft2010[, yr := NULL]
  
  
  
  dependent <- c("Trade_value_total")
  continous <- c("distw", "gdp_d", "area_d")
  log_variables <- paste("log(",continous, ")", sep = "")
  colinear = c("pt3ISO_ABW","yr_2010", "yr_2009","yr_2003", "yr_2008", "flaggsp_o_d_no.gsp.recorded.in.Rose", "legnew_d_uk")
  dummies <- setdiff(setdiff(names(data_bef2010), c(continous, colinear)), dependent) 
  
  linear_het <- as.formula(paste(paste("log(",dependent, "+ 1)", sep = ""),
                                 paste(paste(log_variables, collapse = " + "), paste(dummies, collapse = " + "), sep = " + "), sep = " ~ "))
  
  
  FE <- lm(linear_het, data = data_bef2010)
  summary(FE)
  
  data_aft2010[, Trade_value_total := Trade_value_total + 1]
  predictions <- predict(FE, newdata = data_aft2010, type="response")
  residuals =  predictions - (data_aft2010[,'Trade_value_total'])
  max(residuals)
  
  MSE_FE_test <- (sum(residuals^2)/length(unlist(residuals)))
  
  MSE_FE_test/var(data$Trade_value_total)
  
  
  # Summary to latex
  
  stargazer(FE)
}
