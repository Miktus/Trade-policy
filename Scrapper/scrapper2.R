# Scrapper do comtrade.un.org
# Mo?na jeszcze:
# doda? stop przez u?ytkownika przy wej?ciu w funkcj? w funkcji
# doda? sciaganie 5 lat
#

closeAllConnections()

library(rjson)

download_reporters <- TRUE

if (download_reporters){
  string <- "http://comtrade.un.org/data/cache/partnerAreas.json"
  reporters <- fromJSON(file=string)
  reporters <- as.data.frame(t(sapply(reporters$results,rbind)))
}

# podstawowy scrapper od comtrade
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

output <- setRefClass("scrapper_output", fields = list(data ="ANY", checked = "ANY"))

# Scrapper do lapania bledow, zapetlony

temp <- 0
 
basic_scrapper <- function(vector, year){
  
  print(paste("Trying for vector of", length(vector), "length."))
  
  data <- NULL
  checked <- NULL
  
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
    
    data <- rbind(data, out)
    
    print(showConnections(T))
    
    # print(paste("temp equals ", temp))
    
    if (length(showConnections(T)[,1]) > 3) {
      con <- getConnection(3)
      
      # con <- getConnection(3+temp)
      # temp <- temp+1
      close(con)
    }
  }
  out <- output(data = data, checked = checked)
  return(out)
}

# Scrappowanie dla zakresu lat
main_scrapper <- function( main_vector, from, to){
  main_data <- NULL
  for(i in from:to){
    vector <- main_vector
    cond <- TRUE
    data <- NULL
    try <- NULL
    while(cond){
      print(i)
      Sys.sleep(10)
      unit <- basic_scrapper(vector, i)
      data <- rbind(data, unit$data)
      try <- unique(rbind(try, unlist(unique(as.vector(unit$checked)))))
      vector <- setdiff(unlist(main_vector), try)
      print(length(vector))
      if (length(vector) < 1){
        cond = FALSE
      }
    }
    main_data <- rbind(main_data, data)
  }
}


# Trzeba wywalic pierwszy i drugi indeks ("all", "World"), bo inaczej zapytanie jest zbyt duzo,
# gdyby przeszlo dla "all" to reszta jest niepotrzebna.
vector <-reporters$V1[3:length(reporters$V1)]

#vector <-reporters$V1[3:10] # trial sample

# Scrappowane dla danego roku
#if(length(vector) > 0){
#  data <- basic_scrapper(vector, 2017, 2, 10)
#}

# scrappowane dla przedzialu lat (rok po roku, da sie po 5, max 1962 - 2018)
data <- main_scrapper(vector, 1962, 2018)
write.csv(file = "trade_data.csv", data)
