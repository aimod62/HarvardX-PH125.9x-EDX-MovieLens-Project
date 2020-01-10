#R_script
# code given by prompt
# Note: this process could take a couple of minutes
# Loading Libraries

library(tidyverse)
library(caret)
library(data.table) 
library(recommenderlab)
library(Matrix) 
library(gridExtra) 
library(kableExtra)
library(ggplot2)
knitr::opts_chunk$set(cache = TRUE)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip


dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))



movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data

set.seed(1, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set

validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)
```


dim(edx)
dim(validation)

#Data Preparation
ratings_m <- sparseMatrix(i = as.integer(as.factor(edx$userId)),
                          j = as.integer(as.factor(edx$movieId)),
                          x = edx$rating)

dimnames(ratings_m) <- list(
  user=paste('u', 1:length(unique(edx$userId)), sep=''),
  item=paste('m', 1:length(unique(edx$movieId)), sep=''))

class(ratings_m)
rRM <- as(ratings_m, "realRatingMatrix")
class(rRM)
#Data Exploration
slotNames(rRM)
dim(rRM@data)
rRM_mod <- rRM[rowCounts(rRM) > 50, colCounts(rRM) > 100]
rRM_mod
min_movies1 <- quantile(rowCounts(rRM), 0.90) 
print(min_movies1)
min_users1 <- quantile(colCounts(rRM), 0.90)
rRM_mod2 <- rRM[rowCounts(rRM) > min_movies1,
                colCounts(rRM) > min_users1]
rRM_mod2
#Normalizing the data
rRM_mod2_norm <- normalize(rRM_mod2)
sum(rowMeans(rRM_mod2_norm) > 0.00001) #Testing for Normality
par(mfrow = c(1, 2))
im_1 <- image(rRM_mod2, main = "Heatmap of the top users 
and movies")
im_2 <- image(rRM_mod2_norm, main = "Heatmap of the top users 
and movies")
# Understanding Ratings
v_ratings <- as.vector(rRM_mod2@data) 
unique(v_ratings) 
#count the occurrences 
t_ratings <- table(v_ratings) 
kable(t_ratings)
#0 Means no rating, turn into factors to plot
v_ratings <- v_ratings[v_ratings != 0]
v_ratings <- factor(v_ratings)
head(v_ratings)
str(v_ratings)
#plot
pl_v_ratings <- ggplot(as_tibble(v_ratings), aes(x=value, fill = ..x..))+
  geom_histogram(stat = "count")+
  ggtitle("Ratings Distribution")+
  theme(plot.title = element_text(hjust = 0.5))+
  scale_fill_gradient2(low = "green", high = "blue")+
  guides(fill=FALSE)
pl_v_ratings
#Ratings Distribution
avg_ratings_per_user <- as_tibble(rowMeans(rRM_mod2))
head(avg_ratings_per_user)
pl_avg_ratings_per_user <- ggplot(avg_ratings_per_user, aes(x=value, fill= ..x..))+ 
  geom_histogram(binwidth = 0.10)+
  ggtitle("Distribution of the Average Rating per User") + 
  theme(plot.title = element_text(hjust = 0.5))+
  xlab("Average Rating per Users")+
  scale_fill_gradient("Average Ratings", low = "green", high = "blue")+
  geom_vline(xintercept=mean(avg_ratings_per_user$value), color="black", size=2)
pl_avg_ratings_per_user
kable(summary(avg_ratings_per_user))
# Cosine similarity
similarity_cosine <- similarity(rRM_mod2[1:5, ], method = "cosine", which = "users")
class(similarity_cosine)
kable(as.matrix(similarity_cosine))
similarity_pearson <- similarity(rRM_mod2_norm[1:5, ], method = "pearson", which = "users")
class(similarity_pearson)
kable(as.matrix(similarity_pearson))
par(mfrow = c(1, 2))
im_cosine <- image(as.matrix(similarity_cosine), main = "User Similarity - method = Cosine")
im_pearson <- image(as.matrix(similarity_pearson), main = "User Similarity - method = Pearson")
#Constructing the Model
#Defining Parameters
percentage_training <- 0.8
min(rowCounts(rRM_mod2))#37
items_to_keep <- 25
rating_threshold <- 3
n_eval <- 1
#Splitting the Data
# split
eval_sets_split <- evaluationScheme(data = rRM_mod2, method = "split", 
                              train = percentage_training, given = items_to_keep, 
#Checking eval_sets
getData(eval_sets_split, "train")
`#same number of users
getData(eval_sets_split, "known")#1396 x 1068 rating matrix of class 'realRatingMatrix' with 34900 ratings.
getData(eval_sets_split, "unknown")#1396 x 1068 rating matrix of class 'realRatingMatrix' with 427043 ratings.
#There should be about 20 percent of data in the test set:
#Let's see how many items we have for each user in the known set. It should be equal to items_to_keep
unique(rowCounts(getData(eval_sets_split, "known")))#25
dat_unknown <- as.tibble(rowCounts(getData(eval_sets_split, "unknown"))) 
head(dat_unknown)
pl_unknown <- ggplot(dat_unknown, aes(x=value, fill=..x..))+
  geom_histogram(binwidth = 10) +
  theme(plot.title = element_text(hjust = 0.5))+
  ggtitle("unknown items by the users")+
  xlab("Unknown Items")+
  scale_fill_gradient("Items", low = "green", high = "blue")+
  geom_vline(xintercept=mean(dat_unknown$value), color="black", size=2)
pl_unknown
#k-fold
n_fold <- 4 
eval_sets_kfold <- evaluationScheme(data = rRM_mod2, method = "cross-validation", 
                                    k = n_fold, given = items_to_keep, goodRating = rating_threshold)
#Setting the Model
##single model
model_parameter <- NULL
# Creating the Recommender
eval_recommender <- Recommender(data = getData(eval_sets_kfold, "train"), 
                                method = "IBCF", parameter = model_parameter)
items_to_recommend <- 10
# Predicting 
eval_prediction <- predict(object = eval_recommender, newdata = getData(eval_sets_kfold, "known"),
                           n = items_to_recommend, type = "ratings") 
class(eval_prediction)
#Distribution of Predictions
d_prediction <- as.tibble(rowCounts(eval_prediction))
kable(head(d_prediction))
pl_d_prediction <- ggplot(d_prediction, aes(x =value, fill = ..x..))+
  geom_histogram(binwidth = 10) + 
  ggtitle("Distribution of Predicted Ratings per User")+
  theme(plot.title = element_text(hjust = 0.5))+
  scale_fill_gradient("Items", low = "green", high = "blue")+
  geom_vline(xintercept=mean(d_prediction$value), color="black", size=2)
pl_d_prediction
#Accuracy
#Accuracy per User
eval_accuracy <- calcPredictionAccuracy( 
  x = eval_prediction, data = getData(eval_sets_kfold, "unknown"), byUser = 
    TRUE) 
#head(eval_accuracy)
tail(eval_accuracy)
#Distribution of RMSE
d_accuracy <- as.tibble(eval_accuracy[, "RMSE"])
kable(head(d_accuracy, 10))
pl_d_accuracy <- ggplot(d_accuracy, aes(x=value, fill = ..x..))+ 
  geom_histogram(binwidth = 0.1)+
  ggtitle("Distribution of the RMSE by user")+
  theme(plot.title = element_text(hjust = 0.5))+
  scale_fill_gradient("RMSE", low = "green", high = "blue")+
  geom_vline(xintercept=mean(d_accuracy$value), color="black", size=2)+
  scale_x_continuous(breaks = seq(0, 4, 0.25))
pl_d_accuracy
# Accuracy per Model
#Accuracy per model
eval_accuracy_model <- calcPredictionAccuracy(x = eval_prediction, data = getData(eval_sets_kfold, "unknown"),
                                              byUser = FALSE) 
eval_accuracy_model
#Evaluate the Recommendations
#Evaluate the recommendations
results <- evaluate(x = eval_sets_kfold, method = "IBCF", n = seq(10, 100, 10))
#class(results)
kable(head(getConfusionMatrix(results)[[1]]))
#Aggregated Values
col_to_sum <- c("TP", "FP", "FN", "TN") 
aggregated_values <- Reduce("+", getConfusionMatrix(results))[, col_to_sum] 
kable(head(aggregated_values))
#ROC CURVE
plot(results, annotate = TRUE, main = "ROC curve")
# Precision versus Recall
plot(results, "prec/rec", annotate = TRUE, main = "Precision-recall")
# Recommender Package
Available_Models <- recommenderRegistry$get_entries(dataType = "realRatingMatrix")
kable(names(Available_Models))
kable(lapply(Available_Models, "[[", "description"))
kable(Available_Models$UBCF_realRatingMatrix$parameters)
#Selecting the Most Suitable Model
m_list <- list( 
  IBCF_cos = list(name = "IBCF", param = list(method = "cosine")),
  IBCF_cor = list(name = "IBCF", param = list(method = "pearson")),
  UBCF_cos = list(name = "UBCF", param = list(method = "cosine")),
  UBCF_cor = list(name = "UBCF", param = list(method = "pearson")),
  random = list(name = "RANDOM", param=NULL))
n_recommendations <- c(1, 5, seq(10, 100, 10))
list_results <- evaluate(x = eval_sets_kfold, method = m_list, n = n_recommendations) 
#class(list_results)
avg_matrices <- lapply(list_results, avg)
head(avg_matrices$UBCF_cos[, 5:8])
plot(list_results, annotate = 1, legend = "topleft") 
title("ROC curve")
plot(list_results, "prec/rec", annotate = c(1,2), legend = "bottomright")
title("Precision-Recall")
#Optimizing Numeric Parameters
vector_k <- c(5, 10, 20, 30, 40)
models_to_evaluate_1 <- lapply(vector_k, function(k){ 
  list(name = "UBCF", param = list(method = "pearson", k = k))})
names(models_to_evaluate_1) <- paste0("UBCF_k_", vector_k)
n_recommendations <- 20
list_results_k <- evaluate(x = eval_sets_kfold, method = models_to_evaluate_1, 
                           n = n_recommendations)
plot(list_results, annotate = 1,legend = "topleft") 
title("ROC curve")
plot(list_results, "prec/rec", annotate =1, legend = "bottomright") 
title("Precision-recall")
#Validation Set
ratings_m_val <- sparseMatrix(i = as.integer(as.factor(validation$userId)),
                              j = as.integer(as.factor(validation$movieId)),
                              x = validation$rating)

dimnames(ratings_m_val) <- list(
  user=paste('u', 1:length(unique(validation$userId)), sep=''),
  item=paste('m', 1:length(unique(validation$movieId)), sep=''))

class(ratings_m_val)
rRM_val <- as(ratings_m_val, "realRatingMatrix")
class(rRM_val)
dim(rRM_val)
rRM_mod2_val <- rRM[rowCounts(rRM_val) > 50, colCounts(rRM) > 100]
rRM_mod2
min(rowCounts(rRM_mod2_val))
n_fold <- 40
items_to_keep <- 10
rating_threshold <- 3
n_eval <- 1
set.seed(13)
eval_sets_kfold_val <- evaluationScheme(data = rRM_mod2_val, method = "cross-validation", 
                                        k = n_fold, given = items_to_keep, goodRating = rating_threshold)
eval_recommender_val <- Recommender(data = getData(eval_sets_kfold_val, "train"), 
                                    method = "UBCF", parameter = "pearson")
model_detail$description
set.seed(13)
eval_prediction_val <- predict(object = eval_recommender_val, newdata = getData(eval_sets_kfold_val, "known"),
                               n = items_to_recommend, type = "ratings") 
class(eval_prediction_val)
set.seed(13)
eval_accuracy_val <- calcPredictionAccuracy( 
  x = eval_prediction_val, data = getData(eval_sets_kfold_val, "unknown"))
#head(eval_accuracy)
eval_accuracy_val
#Why do I not achieve a more desirable RMSE?
data("MovieLense")
dim(MovieLense)
class(MovieLense)
slotNames(MovieLense)
min_movies_ML <- quantile(rowCounts(MovieLense), 0.90) 
print(min_movies_ML)
min_users_ML <- quantile(colCounts(MovieLense), 0.90)
print(min_users_ML)
RM <- MovieLense[rowCounts(MovieLense) > min_movies_ML,
                 colCounts(MovieLense) > min_users_ML]
RM
min(rowCounts(RM))
n_fold <- 40
items_to_keep <- 50
rating_threshold <- 3
n_eval <- 1
set.seed(13)
eval_sets_RM <- evaluationScheme(data = RM, method = "cross-validation", 
                                 k = n_fold, given = items_to_keep, goodRating = rating_threshold)
set.seed(13)
eval_recommender_RM <- Recommender(data = getData(eval_sets_RM, "train"), 
                                   method = "UBCF", parameter = "pearson")

set.seed(13)
eval_prediction_RM <- predict(object = eval_recommender_RM, newdata = getData(eval_sets_RM, "known"),
                              n = items_to_recommend, type = "ratings") 
class(eval_prediction_RM)
set.seed(13)
eval_accuracy_RM <- calcPredictionAccuracy(x = eval_prediction_RM,
                                           data = getData(eval_sets_RM, "unknown"),
                                           byUser = FALSE) 
eval_accuracy_RM











