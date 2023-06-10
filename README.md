# Text Analysis, Classification, and Clustering

This project focuses on applying text analysis, classification, and clustering methods to a real-world corpus using R and Python. The goal is to clean the data, perform statistical analysis, create a classification model, and apply a clustering method.

## Project Details

- **Due Date**: June 18th, 2023

## Dataset

The dataset used for this project is the Reuters collection of documents. It is already split into a training set and a test set, each containing 91 topics/subfolders.

## First Part (R)

### Objectives

- Clean the data
- Perform statistical analysis
- Create a Bayesian classifier

### Cleaning the Data

To clean the data, we will use R and the `readtext` library. The data will be processed to remove non-alphabetic characters, extra whitespaces, and convert the text to lowercase.

### Statistics

- Show high dimensionality and sparsity of the data
- Select topics from the training and test sets and create word clouds to visualize the most frequent words in each topic
- Represent each document as a vector of TF-IDF values and measure similarity between pairs of documents

### Classification

- Represent each document as a bag of words
- Create a subset of data containing less than 10 topics
- Train and test a Bayesian classifier using the subset data

## Second Part (Python)

### Objectives

- Clean, tokenize, and lemmatize the data
- Perform statistical analysis
- Apply a clustering method

### Cleaning, Tokenizing, and Lemmatizing the Data

In Python, we will use the `nltk` library for cleaning, tokenizing, and lemmatizing the data. This will involve removing unwanted characters, splitting the text into words, and reducing words to their base form.

### Statistics

- Compute the number of occurrences of each word in the corpus and the number of documents containing it

### Clustering

- Define a cluster as a set of documents containing a common frequent termset
- Find frequent termsets in the data
- Identify documents that include each frequent termset to form clusters

## Recommended Libraries

### R

- `readtext`
- `gsub`
- Additional libraries as needed

### Python

- `nltk`
- `re`
- Additional libraries as needed
