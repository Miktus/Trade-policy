
# Code for the Trade Policy class at PSE
# Author: Michal Miktus at michal.miktus@gmail.com
# Date: 23.02.2019


#path <- '/Users/miktus/Documents/PSE/Trade policy/Model/'
#path <- 'C:/Repo/Trade/Trade-policy/'
path <- 'C:/Users/janro/Desktop/FDI/'

setwd(path)
set.seed(12345)


# Load packages -----------------------------------------------------


list.of.packages <- c("readstata13", "data.table")

new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages, repos = "http://cran.us.r-project.org")

invisible(lapply(list.of.packages, library, character.only = TRUE))

# Useful functions

RMSE = function(m, o){
  sqrt(mean((m - o)^2, na.rm=TRUE))
}

# Perform computations or load the data -----------------------------------

data_cepii <- as.data.table(read.dta13(paste0(path,"Data/gravdata.dta")))
data_trade <- fread(paste0(path,"Data/trade_data.csv"))
data_FDI <- fread(paste0(path,"Data/FDI_3.csv"))
data_FDI<- data_FDI[,c("FLOW","PC", "COU", "Value" , "Year") ] 

# Delete cases for which the trading partner is unknown

data_trade <- data_trade[complete.cases(data_trade[,pt3ISO])]

# Convert TradeValues to numeric, with emphasis on scientific notation issues

data_trade[, TradeValue := as.numeric(format(as.numeric(gsub(',', '.', TradeValue)), scientific = FALSE))]
data_trade <- data_trade[, c('yr', 'TradeValue', 'rt3ISO', 'pt3ISO')]
data_trade <- unique(data_trade[, 'Trade_value_total' := sum(TradeValue), by = c("yr", "rt3ISO", "pt3ISO")], by = c("yr", "rt3ISO", "pt3ISO", "Trade_value_total"))
data_trade[, TradeValue := NULL]
data_trade <- data_trade[!data_trade[, pt3ISO == 'WLD']]

# Merge data

# Inner 

data_inner <- merge(data_trade, data_cepii, by.y = c('year', 'iso3_o', 'iso3_d'), by.x = c('yr', 'rt3ISO', 'pt3ISO'))

# table(data[,"yr"])

data_cepii["year" > 1993]

#Left

data_left <- merge(data_trade, data_cepii["year" > 1993], by.y = c('year', 'iso3_o', 'iso3_d'), by.x = c('yr', 'rt3ISO', 'pt3ISO'), all.y = T)

data_left[, Trade_value_total := lapply(data_left[,"Trade_value_total"], function(x) {ifelse(is.na(x), 0, x)})]

data_FDI[, Value := lapply(data_FDI[,"Value"], function(x) {ifelse(is.na(x), 0, x)})]
data_in_FDI<-data_FDI[FLOW=="IN",]
data_out_FDI<-data_FDI[FLOW=="OUT",]
data_FDI_2<-merge(data_in_FDI,data_out_FDI, by.y = c( "Year", "COU", "PC"),  by.x = c( "Year", "COU", "PC"), all.x =T )
colnames(data_FDI_2)[c(5,7)]<-c("FLOW_IN","FLOW_OUT")
data_FDI_2<-data_FDI_2[,c(1:3,5,7)]
data_FDI_2[, FLOW_OUT := lapply(data_FDI_2[,"FLOW_OUT"], function(x) {ifelse(is.na(x), 0, x)})]
data_left<-merge(data_left,data_FDI_2, by.y = c( "Year", "COU", "PC"), by.x = c('yr', 'rt3ISO', 'pt3ISO'))

# Write whole dataset

fwrite(data_left, 'Data/final_data_trade.csv')








