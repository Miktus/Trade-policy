# Data Scrapper for comtrade.un.org
# Authors Michal Miktus & Mateusz Szmidt
# February 2019

# Environment setup

closeAllConnections()
library(rjson)
library(data.table)

######################################################################################
# Defining all functions necessary to scrap the data


# Support function closing all conections (urls) opened during a scrapping process to avoid errors 
#
# It uses a vector of connections defined at the beginning of each process 
# and closes the opened ones when process ends

connections_dropper <- function(vector){
  new_connections <- getAllConnections()
  if(length(vector)<length(new_connections)){
    connections_to_kill <- setdiff(new_connections, vector)
    for(i in 1:length(connections_to_kill)){
      con <- getConnection(i)
      close(con)
    }
  }
}

# Support function Splitting numeric or string vector into vector of n-elements batches with "," separator
# It allows to lower the number of queries

vector_processing <- function(vector, n){
  
  # We consider a case when the set size of a batch is greater than the length of vector
  
  if(length(vector)> n){
    
    list <- split(vector,cut(seq_along(vector),ceiling(length(vector)/n) , labels = F))
    j = 1
    vector <- c()
    
    for(i in list){
      subsample <- NULL
      for(ii in i){
        if(is.null(subsample)){
          subsample <- paste(subsample, ii, sep="")
        }
        else subsample <- paste(subsample, ii, sep=",")
      }
      
      # Self check if the split was performed correctly 
      if(length(i) > n){
        print("Something went wrong!")
      }
      vector[j] <- subsample
      j = j + 1
    }
  }
  else{
    vector_ <- vector
    vector <- c()
    subsample <- NULL
    for(i in 1:length(vector_)){
      if(is.null(subsample)){
        subsample <- paste(subsample, vector_[i], sep="")
      }
      else subsample <- paste(subsample, vector_[i], sep=",")
    }  
    vector[1] <- subsample
  }
  return(vector)
}



# Basic data scrapper for a single query
# Default values of parameters adjusted to download annuall data on trade flows
# Source: https://comtrade.un.org/data/Doc/api/ex/r

get.Comtrade <- function(url="http://comtrade.un.org/api/get?"
                         ,maxrec=50000
                         ,type="C"
                         ,freq="A"
                         ,px="HS"
                         ,ps="now"
                         ,r
                         ,p
                         ,rg="all"
                         ,cc="TOTAL"
                         ,fmt="json"
)
{
  string<- paste(url
                 ,"max=",maxrec,"&" #maximum no. of records returned
                 ,"type=",type,"&" #type of trade (c=commodities)
                 ,"freq=",freq,"&" #frequency
                 ,"px=",px,"&" #classification
                 ,"ps=",ps,"&" #time period
                 ,"r=",r,"&" #reporting area
                 ,"p=",p,"&" #partner country
                 ,"rg=",rg,"&" #trade flow
                 ,"cc=",cc,"&" #classification code
                 ,"fmt=",fmt        #Format
                 ,sep = ""
  )
  
  if(fmt == "csv") {
    raw.data<- read.csv(string,header=TRUE)
    return(list(validation=NULL, data=raw.data))
  } else {
    if(fmt == "json" ) {
      raw.data<- fromJSON(file=string)
      data<- raw.data$dataset
      validation<- unlist(raw.data$validation, recursive=TRUE)
      ndata<- NULL
      if(length(data)> 0) {
        var.names<- names(data[[1]])
        data<- as.data.frame(t( sapply(data,rbind)))
        ndata<- NULL
        for(i in 1:ncol(data)){
          data[sapply(data[,i],is.null),i]<- NA
          ndata<- cbind(ndata, unlist(data[,i]))
        }
        ndata<- as.data.frame(ndata)
        colnames(ndata)<- var.names
      }
      return(list(validation=validation,data =ndata))
    }
  }
}


# Definining an object for an output of basic_scrapper function
output <- setRefClass("scrapper_output", fields = list(data ="ANY", checked = "ANY", hits = "ANY"))


# Function scrapping the data on all possible connections 
# between countries defined in the input <vector> and all the partners available 
# for the years defined as <year> .
#
# To control for the number of queries we use the parameter <hits>.
# It allows to stop the process after 100 hits to not exceed an hourly limit of 100 queries

basic_scrapper <- function(vector, year, hits){
  
  # Console output and definition of an output object 
  print(paste("Trying for vector of", length(vector), "length."))
  current_connections <-getAllConnections()
  data <- NULL
  checked <- NULL
  
  # Looping over all batches of countries in a tryCatch block to avoid a failure of a process
  for(i in 1:length(vector)){
    tryCatch({
      print(i)
      out <- NULL
      unit <- get.Comtrade(r=vector[i], p="all", ps=toString(year), freq="A")
      
      if(is.null(unit$data)){
        checked <- rbind(checked, vector[i])
        print(paste("No data available for year", year, "for" ,vector[i]))
      }
      else{
        checked <- rbind(checked, vector[i])
        out <- unit$data
      }
    }, 
    error = function(e){
      print(paste("Error for", i))
    }
    )
    
    # Stopping the process for 1 hour after 100 hits
    hits = hits + 1
    
    if(hits >= 100){
      Sys.sleep(3600)
      hits = 0
    }
    
    # Output generation
    data <- rbind(data, out)
    
    # Dropping all connections opened during a process
    connections_dropper(current_connections)
  }
  
  out <- output(data = data, checked = checked, hits = hits)
  return(out)
}


# Main scrapping process using basic_scrapper function
# It splits the year range into batches of length 5 to optimize the number of queries.
# It also splits the list of countries into batches with initial length of 5,
# the batches where the error occured are joined and split again into batches of smaller size (up to 1). 

main_scrapper <- function(main_vector, from, to){
  
  # Definition of an output object and years range splitting into batches of 5
  main_data <- NULL
  years <- vector_processing(seq(from, to), 5)
  hits = 0
  
  # Looping over the years
  for(i in 1:length(years)){
    vector <- main_vector
    cond <- TRUE
    data <- NULL
    try <- NULL
    split <- 5
    
    # Scrapping the data for all connections between the countries for a given batch of years
    # It is continued until for none of a countries an error is reported
    while(cond){
      print(paste("Scrapping for years:", years[i]))
      print(paste("The number of countries checked in one hit is", split))
      
      unit <- basic_scrapper(vector, years[i], hits)
      data <- rbind(data, unit$data)
      hits <- unit$hits
      try <- unique(rbind(unlist(try), unique(unlist(unit$checked))))
      
      # Vector of countires for which error is reported and so the queries will be repeated 
      vector <- setdiff(unlist(strsplit(main_vector, "\\,")), unlist(strsplit(try, "\\,")))
      
      #  Checking if data for all countries is scrapped, 
      #  then if not splitting the vector of countries into batches of smaller size.
      if (length(vector) < 1){
        cond = FALSE
      }
      else{
        split <- max(split - 1, 1)
        vector <- vector_processing(vector, split)
      }
    }
    
    # Overriding the state of the scrapping after each finished batch of years
    main_data <- rbind(main_data, data)
    write.csv(file = "trade_data.csv", main_data)
  }
  return(main_data)
}

######################################################################################
# Scrapping the data


# Scrapping the list of the countries listed in the comtrade database

download_reporters <- TRUE
if (download_reporters){
  string <- "http://comtrade.un.org/data/cache/partnerAreas.json"
  reporters <- fromJSON(file=string)
  reporters <- as.data.frame(t(sapply(reporters$results,rbind)))
}

# Adjusting the list of reporters for which the process works (removing "world" and "all")
vector <-vector_processing(unlist(as.numeric(reporters$V1[3:length(reporters$V1)])), 5)

# Data scrapping for the range of dates available in comtrade database
data <- main_scrapper(vector, 1962, 2018)
fwrite(file = "trade_data.csv", data)
