######################## First Exercice R ###################################
############################ MAALEJ Zied & Prince ###########################"



#Installing the needed packages: 
install.packages("readtext")
install.packages("stopwords")
install.packages("tm")
install.packages("textTinyR")

#Loading the needed pachages:
library (NLP)
library(stopwords)
library(tm)
library(readtext)
library(RColorBrewer)
library (wordcloud)
library(textTinyR)
library (Matrix)
library(proxy)

# Loading the data (In this example we will load the first 
# file acq and we will do all the coding according to that)

reuters_path <- "training"
corpus <- readtext(reuters_path)
sample_size <- round(lengths(corpus) * 0.1)

### Cleaning the data: 
# We will include all the steps in one function so it could be used later on if needed: 
clean_text_data <- function(text_data) {
  #Convert text to lowercase:
  text_data$text <- tolower(text_data$text)
  #Remove punctuation and special characters: (methode 1)
  text_data$text <- gsub("[[:punct:]]", "", text_data$text)
  #Remove numbers:
  text_data$text <- gsub("\\d+", "", text_data$text)
  #Remove stopwords:
  stopwords <- stopwords::stopwords("english")
  text_data$text <- removeWords(text_data$text, stopwords)
  #Remove excess whitespace:
  text_data$text <- gsub("\\s+", " ", text_data$text)
  text_data$text <- trimws(text_data$text)
  #Perform stemming or lemmatization (optional): 
  text_data$text <- tm::stemDocument(text_data$text)
  #Remove punctuation and special characters: (methode 2)
  #text_data$text <- gsub("[^a-zA-Z0-9 ]", "", text_data$text)
  
  return(text_data)
}

subset_corpus <- corpus[1:4024, ]

cleaned_corpus <- clean_text_data(corpus)



######################################Statistics###################################### 

###Show the high dimensionality and sparsity of the data

#To demonstrate the high dimensionality and sparsity of a text corpus, 
#ywe will calculate and display the number of unique terms (vocabulary size) 
#and the sparsity of the document-term matrix.
dtm <- DocumentTermMatrix(cleaned_corpus$text)
class(dtm)

#change dtm to matrix 
m = as.matrix(dtm)

# Get the vocabulary size
length(m)  


#207000

# Calculate the sparsity
length(which(m==0))/length(m)
#0.9730338
#A sparsity of 0.9730338 indicates that the document-term matrix is highly sparse, 
#meaning it has a large proportion of zero values compared to non-zero values.

###Worldcloud: 
#1: 
frequency <- colSums(m)
frequency 
frequency <- sort(frequency, decreasing = TRUE)
words <- names(frequency)
pal2 <- brewer.pal(8,"Dark2")
wordcloud(words[1:100], frequency[1:100],  colors =pal2)

###TF-IDF
tfidf <- weightTfIdf(dtm)


# Convert TF-IDF matrix to a data frame
tfidf_df <- as.data.frame(as.matrix(tfidf))

# View the TF-IDF table
print(tfidf_df)

similarity_matrix <- proxy::simil(tfidf_df[, -ncol(tfidf_df)], method = "cosine")
#proxy::simil() is a function from the proxy package that calculates similarity/dissimilarity 
#between pairs of objects using various methods. In this case, we are using it to 
#calculate cosine similarity.

tfidf_similarity <- as.data.frame(as.matrix(similarity_matrix))
#In this dataframe we have the value of similarity between each two document. 

####################################Classification#################################### 

#train_path <- "h:\\Desktop\\Reuters_Dataset\\train"

#corpus <- readtext(train_path)

test_path <- "h:\\Desktop\\Reuters_Dataset\\test"

test_corpus <- readtext(test_path)


# Create a corpus (the corpus is the training set)
corpus <- Corpus(VectorSource(cleaned_corpus$text))

# Create a document-term matrix
dtm1 <- DocumentTermMatrix(corpus)

# Convert the document-term matrix to a data frame
bag_of_words <- as.data.frame(as.matrix(dtm))

cleaned_corpus$doc <- sub("/.*", "", cleaned_corpus$doc)
# Add document identifiers if needed
bag_of_words$doc <- cleaned_corpus$doc

# View the bag of words representation
print(bag_of_words)

bag_of_words$doc
library(e1071)

set.seed(123)
train_indices <- sample(1:nrow(bag_of_words), nrow(bag_of_words) * 0.7)  # 70% for training
train_data <- bag_of_words[train_indices, ]
test_data <- bag_of_words[-train_indices, ]

# Create and train the Naive Bayes classifier
classifier <- naiveBayes(doc ~ ., data = train_data)

# Make predictions on the test set
predictions <- predict(classifier, newdata = test_data)

# Calculate the accuracy of the classifier
accuracy <- sum(predictions == test_data$label) / nrow(test_data)
print(paste("Accuracy:", accuracy))

train_topics <- train_data$doc




















