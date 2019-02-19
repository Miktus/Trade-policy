
require(readstata13)


# Code for the 1st PS for Machine Learning in Econ class at PSE
# Author: Michal Miktus at michal.miktus@gmail.com
# Date: 03.02.2019


path <- '/Users/miktus/Documents/PSE/Trade policy'

setwd(path)
set.seed(12345)


# Load packages -----------------------------------------------------


list.of.packages <- c("readstata13", "data.table", "gravity")

new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages, repos = "http://cran.us.r-project.org")

invisible(lapply(list.of.packages, library, character.only = TRUE))


# Perform computations or load the data -----------------------------------

data_cepii <- as.data.table(read.dta13(paste0(path,"/gravdata_cepii/gravdata.dta")))
data_cepii_small <- as.data.table(read.dta13(paste0(path,"/gravdata_cepii/col_regfile09.dta")))
trade <- as.data.table(read.dta13(paste0(path,"/gravdata_cepii/Trade_cepii8006_STATA.dta")))


# PPML: Poisson Pseudo Maximum Likelihood

countries_chosen <- c("POL", "GER", "RUS", "SLO", "ITA")
grav_small <- data_cepii_small[iso_o %in% countries_chosen]

PPRML <- ppml(dependent_variable="flow", distance="distw", additional_regressors=c("rta","iso_o","iso_d"), robust=TRUE, data=grav_small)
