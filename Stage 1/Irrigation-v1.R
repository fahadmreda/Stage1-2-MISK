#Analysis of the Irrigation dataset
#Fahad Reda
#17.11.2020
#A small case study
#load package
library(tidyverse)

#being with wide "messy" format:
irrigation <- read.csv("data/irrigation_wide.csv")

#Examine the data :
glimpse(irrigation)
summary(irrigation)

#In 2007, what is the total area under irrigation
#for only the Americas

irrigation %>%
 filter (year == 2007) %>%
 select( ends_with("erica")) %>% sum()

###
irrigation %>%
 filter(year==2007) %>%
 select ('N.America', 'S.America')%>% sum()
###
irrigation %>%
 filter(year==2007) %>%
 select(4,5) %>%
 sum()




#how to make tidy data

irrigation_t <- irrigation %>%
 pivot_longer(-year, names_to ="region")
irrigation_t


#what is the total areas under irrigation in each  year?


irrigation_t %>%
 group_by(year) %>%
 summarise(total = sum (value))




irrigation_t %>%
 group_by(region)%>%
 summarise
(diff = value [year==2007] - value[year==1980]) %>%



 ##


 c (0, diff(xx)/xx[-length(xx)])
irrigation_t %>%
 group_by  (region) %>%
 mutate (diff(value))



#what is the rate of change in each region?

irrigation_t %>% arrange(region) %>%
 group_by(region) %>%
 mutate(rate= c (0, diff(value)/ value [-length(value)]))


#where is the lowest and highest?

irrigation_t[which.max(irrigation_t$rate),]
irrigation_t[which.min(irrigation_t$rate),]

#this will give max rate for each region

####

irrigation_t %>%
 slice_max(rate, n=1)

#becaue ... the tibble is still a group_df
#so to get the global answer : ungroup()


#highst
irrigation_t %>%
 ungroup() %>%
 slice_max(rate, n = 1)

#lowest

irrigation_t %>%
 ungroup() %>%
 slice_min(rate, n= 1 )


#standardize aginest 1980 (relative change over 1980) (easier)

#which region increased most from 1980 t0 2007?

irrigation_t %>%
 group_by(region) %>%
 group_split()
summarise(diff = value[year==2007] - value[year==1980])


#plot areas over time for each region?

ggplot(irrigation, aes(x= region, y = area)) +
 geom_point()
