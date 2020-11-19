#Analysis of the Diamonds dataset
#Fahad Reda
#18.11.2020
# A small case study for EDA and Stat

# load package
library(tidyverse)

jems <- read_csv("data/diamonds.csv")



#Exercise 8.2 (Examine structure) ----
# Get familiar with dataset
summary(jems)
names(jems)
glimpse(jems)

# Attributes
attributes(jems)
typeof(jems)

jems <- as_tibble(jems)
jems
# Exercise 8.3 (Counting individual groups) -----
# - How many diamonds with a clarity of category “IF” are present in the data-set?

count_if <- jems %>%
  filter(clarity=='IF') %>%
  count()
IF_count <- count_if$n
IF_count

# - What fraction of the total do they represent?
count_total <- jems %>%
  count()
Total_count <- count_total$n

fractionOfIF <- IF_count/Total_count

fractionOfIF

# Exercise 8.4 (Summarizing proportions) ----
# What proportion of the whole is made up of each category of clarity?
count_clr <- jems %>%
  count(clarity)
p <- count_clr$n/Total_count
p

proportion <- data.frame(count_clr,p)
proportion


# Exercise 8.5 (Find specific diamonds prices)
# What is the cheapest diamond price overall?
cheapestPrice <- jems %>%
  summarise(min(price))
cheapestPrice

# What is the range of diamond prices?
rangePrice <- jems %>%
  summarise(range(price))
rangePrice


# What is the average diamond price in each category of cut and color?
jems %>%
  group_by(cut,color)%>%
  summarise(avg=mean(price))

# Exercise 8.6 (Basic plotting)----
# Make a scatter plot that shows the price of a diamond as described
# by another continous variable, like the carat.
ggplot(jems,aes(carat,price)) +
  geom_point()


#produce modules

jems_lm <- lm(price)



#Exercise 8.7 ----
#(Interpreting plots)
#**What can you say about the relationship between these two variables?
##Do you think that you can use the carat weight of
##a diamond to predict its price?


#Exercise 8.8 (Applying transformations) ----
#**Using the functions we discuss earlier, and in class,
#apply a log10 transformation to both the price and carat.
#You can save these as new columns in the data set called`price_log10`and`
#carat_log10`.
carat_log10 <- log10(jems$carat)
carat_log10
price_log10 <- log10(jems$price)
price_log10
jems2 <- data.frame(jems,carat_log10,price_log10)
jems2

#Exercise 8.9 (Basic plotting) ----
#**Make a scatter plot that shows the price of a diamond as described by
#*another continous variable, like the carat.
ggplot(jems2,aes(carat_log10,price_log10)) +
  geom_point()

#Exercise 8.10 (Viewing models) ----
#describes the relationship shown in the plot?
jems_lm <- lm(price_log10 ~ carat_log10,data=jems2)
jems_lm

#Exercise 8.11 (Displaying models) ----
#Now that we’ve described the diamond price given a single variable,
#can you display that on the plot?
ggplot(jems_lm,aes(carat_log10,price_log10)) +
  geom_point()+
  geom_smooth(aes(price_log10 ~ carat_log10))
# Extra Ex ----
jems %>%
  filter(clarity=='VVS2' & cut=='Good')
