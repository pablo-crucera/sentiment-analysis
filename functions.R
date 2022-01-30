library(reticulate)
library(purrr)
library(keras)
library(naivebayes)

build_embedder <- function(tokenizer, embedding_size = 128) {
  input_target <- layer_input(shape = 1)
  input_context <- layer_input(shape = 1)
  
  embedding <- layer_embedding(
    input_dim = tokenizer$num_words + 1,
    output_dim = embedding_size,
    input_length = 1, 
    name = "embedding"
  )
  
  target_vector <- input_target %>% 
    embedding() %>% 
    layer_flatten() # to return the dimension of the input
  
  context_vector <- input_context %>%
    embedding() %>%
    layer_flatten()
  
  dot_product <- layer_dot(list(target_vector, context_vector), axes = 1)
  
  output <- layer_dense(dot_product, units = 1, activation = "sigmoid")
  
  model <- keras_model(list(input_target, input_context), output)
  model %>% compile(loss = "binary_crossentropy", optimizer = "adam")
  
  return(model)
}

skipgrams_generator <- function(text, tokenizer, window_size, negative_samples){
  
  gen <- texts_to_sequences_generator(tokenizer, sample(text))
  
  function() {
    skip <- generator_next(gen) %>%
      skipgrams(
        vocabulary_size = tokenizer$num_words, 
        window_size = window_size, 
        negative_samples = 1
      )
    
    x <- transpose(skip$couples) %>% map(. %>% unlist %>% as.matrix(ncol = 1))
    y <- skip$labels %>% as.matrix(ncol = 1)
    
    list(x, y)
  }
  
}

build_lstm <- function(max_features = 3000, embedding_size = 128, weights = NULL,
                       max_length = 1000, units_lstm = 64, dropout_lstm = 0) {
  lstm <- keras_model_sequential() 
  lstm %>%
    layer_embedding(input_dim = max_features + 1,
                    output_dim = embedding_size,
                    weights = weights,
                    input_length = max_length) %>%
    bidirectional(layer_lstm(units = units_lstm, dropout = dropout_lstm)) %>% 
    layer_dropout(rate = 0.5)    %>%
    layer_dense(units = 5, activation = 'softmax') %>% 
    compile(loss = 'categorical_crossentropy',
            optimizer = optimizer,
            metrics = c('accuracy'))
  
  return(lstm)
}
