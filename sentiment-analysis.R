setwd(dirname(rstudioapi::getSourceEditorContext()$path))
library(tensorflow)
library(keras)
library(here)
library(utf8)
library(stringr)
library(dplyr)
library(reticulate)
library(purrr)
library(fastDummies)
library(quanteda)
library(quanteda.textmodels)
library(caret)
source("functions.R")

#install_miniconda()
#conda_install(c('keras','tensorflow'), pip=TRUE)

# Define parameters
method <- "MNB"              # "MNB" for Multinomial Naïve Bayes, "LSTM" for neural model
use_reviews <- TRUE           # if TRUE uses reviews as input text, otherwise uses summaries
perc_test <- 0.2              # fraction of instances for test split

# Overall neural model parameters
max_features <- 5000          # number of words used in dictionary
max_length <- 2500            # considered length for reviews
use_own_embedding <- TRUE     # if TRUE, embedding is trained together with LSTM

# Embedding parameters
embedding_size <- 32          # dimension of embedding vector
skip_window <- 8              # number of skip-gram
num_sampled <- 5              # number of negative samples for model tuning
steps_embedding <- 128        # number of steps per epoch in word embedding
epochs_embedding <- 30        # number of epochs in word embedding
batch_size_embedding <- 
  12000%/%steps_embedding     # batch size for model

# LSTM parameters
dropout_lstm <- 0.3           # dropout for the LSTM layer
units_lstm <- 16              # number of units for LSTM model
batch_size_lstm <- 32         # batch size for training LSTM model
epochs_lstm <- 10             # epochs in training LSTM model
optimizer <- 'adam'           # optimizer used in training LSTM model

# Multinomial Naïve-Bayes parameters
smoothing_factor <- 1         # Smoothing parameter for feature counts by class

# Read file
df <- read.csv(here("preprocessed_kindle_review.csv"))

if (use_reviews) {
  input_text <- df$reviewText
} else {
  input_text <- df$summary
}


# Checking for right encoding and character normalization
sum(as.integer(!utf8_valid(input_text)))
sum(utf8_normalize(input_text) != input_text)

# Create boolean vector for deciding if an instance goes to train or test set
ind <- sample(c(TRUE, FALSE), nrow(df), replace=TRUE, prob=c(1-perc_test, perc_test))

################################################################################

# LSTM model
if (method == "LSTM") {
  # Tokenize
  tokenizer <- text_tokenizer(num_words = max_features)
  tokenizer %>% fit_text_tokenizer(input_text)
  
  tokenized_sentences <- texts_to_sequences(tokenizer, input_text)
  tokenized_sentences <- pad_sequences(tokenized_sentences, maxlen = max_length)
  
  
  # Create train + test splits
  x_train <- tokenized_sentences[ind,]
  x_test <- tokenized_sentences[!ind,]
  y_train <- dummy_cols(df[ind,], select_columns = "rating") %>%
    select(c((ncol(.)-4):ncol(.))) %>%
    as.matrix(.) %>%
    tf$convert_to_tensor(.)
  y_test <- dummy_cols(df[!ind,], select_columns = "rating") %>%
    select(c((ncol(.)-4):ncol(.))) %>%
    as.matrix(.) %>%
    tf$convert_to_tensor(.)
  
  
  
  if (use_own_embedding) {
    weights <- NULL
  } else {
    # Model architecture of the word embedder
    embedder <- build_embedder(tokenizer, embedding_size = embedding_size)
    
    # Train embedder model
    embedder %>%
      fit(
        skipgrams_generator(input_text, tokenizer, skip_window, negative_samples), 
        steps_per_epoch = steps_embedding, epochs = epochs_embedding, 
        batch_size = batch_size_embedding
      )
    
    weights <- tf$reshape(tf$convert_to_tensor(get_weights(embedder)[[1]]),
                          shape(1,max_features+1,embedding_size))
  }
  
  # Create LSTM
  lstm <- build_lstm(max_features = max_features, embedding_size = embedding_size,
                     weights = weights, max_length = max_length,
                     units_lstm = units_lstm, dropout_lstm = dropout_lstm)
  
  # Fit LSTM
  lstm %>% fit(
    x_train, y_train,
    batch_size = batch_size_lstm, epochs = epochs_lstm,
    validation_data = list(x_test, y_test))
  
  # Generate predictions
  predictions_lstm <- predict(lstm, x_test)
  
  # Reconvert labels to integers
  pred_lstm <- c()
  for (idx in 1:nrow(predictions_lstm)) pred_lstm[idx] <- 
    which.max(predictions_lstm[idx,])
  
  pred_lstm <- factor(pred_lstm)
  
  # Output metrics
  conf_mat_lstm <- confusionMatrix(pred_lstm, factor(df$rating[!ind]))
  cat("Confusion matrix:\n")
  print(conf_mat_lstm$table)
  cat("\nMetrics by class:\n")
  print(conf_mat_lstm$byClass)
  cat("\nOverall metrics:\n")
  print(conf_mat_lstm$overall)
  cat("\nRMSE: ")
  cat(norm(as.integer(pred_lstm) - df$rating[!ind],"2")/sqrt(length(pred_lstm)))

################################################################################

# Multinomial Naive Bayes
} else if (method == "MNB") {
  dfmat_train <- dfm(tokens(input_text[ind], what = "word", remove_punct = TRUE, 
                          remove_symbols = TRUE))
  
  dfmat_test <- dfm(tokens(input_text[!ind], what = "word", remove_punct = TRUE, 
                           remove_symbols = TRUE)) %>%
    dfm_match(., features = featnames(dfmat_train))
  
  pred_mnb <- textmodel_nb(dfmat_train, df$rating[ind], distribution = "multi", 
                           prior = "termfreq", smooth = smoothing_factor) %>%
    predict(., newdata = dfmat_test)
  
  conf_mat_mnb <- confusionMatrix(pred_mnb, factor(df$rating[!ind]))
  cat("Confusion matrix:\n")
  print(conf_mat_mnb$table)
  cat("\nMetrics by class:\n")
  print(conf_mat_mnb$byClass)
  cat("\nOverall metrics:\n")
  print(conf_mat_mnb$overall)
  cat("\nRMSE: ")
  cat(norm(as.integer(pred_mnb) - df$rating[!ind],"2")/sqrt(length(pred_mnb)))
} else cat("Unrecognised method, please set 'method' to \"MNB\" or \"LST\"")