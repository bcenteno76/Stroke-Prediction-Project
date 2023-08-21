### Loading Necessary libraries
library(readxl)
library(tidyverse)
library(dplyr)
library(caret)

### Downloading the dataset
urlfile <- 'https://raw.githubusercontent.com/bcenteno76/StrokeProject/main/strokedata.csv'
stroke_df <- read_csv(url(urlfile))


### Data Exploration, cleaning and Visualization

## Exploring the structure of the dataset
str(stroke_df)

## Checking the presence of NAs
any(is.na(stroke_df))

## Checking the presence of duplicates
any(duplicated(stroke_df))

## Adjusting the dataset: char and Int variables parsed to Factor
stroke_df <- stroke_df %>% select(- id)%>% mutate(stroke = as.factor(stroke),# id is not a predictor
                                                  bmi = as.numeric(bmi),
                                                  gender = as.factor(gender),
                                                  ever_married = as.factor(ever_married),
                                                  smoking_status = as.factor(smoking_status),
                                                  hypertension = as.factor(hypertension),
                                                  heart_disease = as.factor(heart_disease),
                                                  work_type = as.factor(work_type),
                                                  Residence_type = as.factor(Residence_type))

## Imputing NAs:
library(mice)
datos_imputados <- mice(stroke_df, method = "pmm", m = 5, seed = 123)
stroke_df <- complete(datos_imputados)

## Stroke proportion
stroke_df %>%mutate(stroke = ifelse(stroke == 1, 'Yes', 'No'))%>%  # Bar graph: Stroke Proportion
  count(stroke) %>%                              
  mutate(porcentaje = n / sum(n) * 100) %>%   
  ggplot(aes(x = stroke, y = porcentaje,fill= stroke)) +      
  geom_bar(stat = "identity") + 
  labs(title = "Stroke Porportion", x = "Stroke", y = "%") +
  scale_y_continuous(labels = scales::percent_format(scale = 1))+
  geom_text(aes(label = paste(round(porcentaje, 1), "%"), y = porcentaje), vjust = -0.5, size = 4)+
  theme(legend.position = "none")

## Gender
stroke_df %>%  # Bar graph: Stroke Proportion
  count(gender) %>%                              
  mutate(porcentaje = n / sum(n) * 100) %>%   
  ggplot(aes(x = gender, y = porcentaje,fill= gender)) +      
  geom_bar(stat = "identity") + 
  labs(title = "Gender Porportions", x = "Gender", y = "%") +
  scale_y_continuous(labels = scales::percent_format(scale = 1))+
  geom_text(aes(label = paste(round(porcentaje, 1), "%"), y = porcentaje), vjust = -0.5, size = 4)+
  theme(legend.position = "none")

options(digits = 2)
contingency_table <- stroke_df%>% mutate(stroke = ifelse(stroke == 1, 'yes','no'))%>%
  select(stroke, gender)%>%table()
prop.table(contingency_table, margin = 2) * 100 # table of stroke percentages in each gender stratum

## Exploring and visualizing continuous variables: Age, bmi and average glucose level
stroke_df %>% select(age,bmi,avg_glucose_level)%>% summary()
stroke_df%>% ggplot(aes(y=age))+ geom_boxplot()+ggtitle('Age Distribution')
stroke_df%>% ggplot(aes(y=bmi))+ geom_boxplot()+ggtitle('BMI Distribution')
stroke_df%>% ggplot(aes(y=avg_glucose_level))+ geom_boxplot()+ggtitle('Average Glucose Distribution')

# Comparing the distribution of each variable in patients with and without stroke
stroke_df%>% mutate(stroke = ifelse(stroke == 1, 'yes','no'))%>% ggplot(aes(stroke,age))+ geom_boxplot()
stroke_df%>% mutate(stroke = ifelse(stroke == 1, 'yes','no'))%>% ggplot(aes(stroke,bmi))+ geom_boxplot()
stroke_df%>% mutate(stroke = ifelse(stroke == 1, 'yes','no'))%>% ggplot(aes(stroke,avg_glucose_level))+ geom_boxplot()

## Exploring the different proportions of stroke in each category of the predictors hypertension, heart-disease, smoking_status, work_type, ever_married, Residence_type
stroke_df%>% mutate(hypertension = ifelse(hypertension == '1','yes','no'))%>%
  group_by(hypertension)%>% summarise(Stroke = (sum(stroke == 1)/n()) * 100)%>%
  arrange(desc(Stroke))%>% rename('Stroke %' = Stroke, 'Hypertension' = hypertension)
stroke_df%>% mutate(heart_disease = ifelse(heart_disease == '1','yes','no'))%>%
  group_by(heart_disease)%>% summarise(Stroke = (sum(stroke == 1)/n()) * 100)%>%
  arrange(desc(Stroke))%>% rename('Stroke %' = Stroke, 'Heart Disease' = heart_disease)
stroke_df%>% group_by(smoking_status)%>% summarise(Stroke = (sum(stroke == 1)/n()) * 100)%>%arrange(desc(Stroke))%>% rename('Stroke %' = Stroke, 'Smoking Status' = smoking_status)
stroke_df%>% group_by(work_type)%>% summarise(Stroke = (sum(stroke == 1)/n()) * 100)%>%
  arrange(desc(Stroke))%>% rename('Stroke %' = Stroke, 'Work Type' = work_type)
stroke_df%>% group_by(ever_married)%>% summarise(Stroke = (sum(stroke == 1)/n()) * 100)%>%arrange(desc(Stroke))%>% rename('Stroke %' = Stroke, 'Ever Married' = ever_married)
stroke_df%>% group_by(Residence_type)%>% summarise(Stroke = (sum(stroke == 1)/n()) * 100)%>%arrange(desc(Stroke))%>% rename('Stroke %' = Stroke, 'Residence Type' = Residence_type)

### Data Splitting
set.seed(1) # seed set to 1 to get reproducible results
test_index <- createDataPartition(stroke_df$stroke,times = 1, p = 0.2, list = FALSE)
train_set <- stroke_df[-test_index,]
test_set <- stroke_df[test_index,]
## Oversampling method to address imbalanced data
train_set <- upSample(x = train_set[, -11], y = train_set$stroke, yname = "stroke")

### Model training and testing

## Model 1 : Logistic Regression

fit_glm <- train(stroke~., method= 'glm', data = train_set)# Training the model
y_hat <- predict(fit_glm,train_set) # predicting stroke (o = no, 1 = yes)
result1 <-confusionMatrix(y_hat,train_set$stroke)$overall["Accuracy"] # Testing the model on the train_set
result1

## Model 2: knn algorithm
fit_knn <- train(stroke~.,method = 'knn', data = train_set, # Training the model
                 tuneGrid = data.frame(k = seq(7, 51, 2)))
y_hat1 <- predict(fit_knn, train_set) # predicting stroke (o = no, 1 = yes)
result2 <-confusionMatrix(y_hat1,train_set$stroke)$overall["Accuracy"] # Testing the model on the train_set
result2

## Model 3: random forest algorithm
fit_rf <- train(stroke~.,method = 'rf', data = train_set)  # Training the model
y_hat2 <- predict(fit_rf, train_set) # predicting stroke (o = no, 1 = yes)
result3 <-confusionMatrix(y_hat2,train_set$stroke)$overall["Accuracy"] # Testing the model on the train_set
result3

### Model validation
y_hat_rf <- predict(fit_rf, test_set)
result4 <-confusionMatrix(y_hat_rf,test_set$stroke)$overall["Accuracy"]
result4

### Results

results <- tibble(Method = c("glm",'knn',
                                  'rf','Final model validation (rf)'),
                       Accuracy = c(result1, result2, result3,result4))
results # summary table showing the accuracy obtained for each model
