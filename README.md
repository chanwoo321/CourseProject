# CourseProject

Please fork this repository and paste the github link of your fork on Microsoft CMT. Detailed instructions are on Coursera under Week 1: Course Project Overview/Week 9 Activities.

# Table of Contents

[Introduction](#intro)

[Brief Overview](#overview)

[Obtaining and Organizing the Data](#data)

[Running the Software](#run)

[Code Information](#code)

 - [Baseline Model](#baseline)

 - [Cross-Collection model](#ccm)

<a name="intro"/>

# Introduction



Hello! Our group consists of three people: Jonathan Kim, Michael Xiang, and Tyler Ruckstaetter. This repository was created for the final project for the class CS 410: Text Information Systems at the University of Illinois at Urbana-Champaign. The purpose of this project was to reproduce the Comparatrive Text Mining model described in the paper "A Cross-Collection Mixture Model for Comparative Text Mining", which can be found [here](http://sifaka.cs.uiuc.edu/czhai/pub/sigkdd04-ctm.pdf). 

<a name="overview"/>

# Brief Overview

The model in the paper was created in order to solve the novel text mining problem of "Comaparative Text Mining". This problem consists of trying to find common themes across some collections of texts and to summarize the similarities and differences between these collections. Thus, a generative probabilistic mixture model was proposed. It performs cross-collection clustering and within-collection clustering. This is done to find themes across collections and utilize the fact that each collection may have information on a similar topic to the other collections as opposed to a completely different one.

The data used in the paper and the model implemented in this repository were laptop reviews and war news. In particular, the laptop reviews analyzed the Apple iBook Mac Notebook, the Dell Inspiron 8200, and the IBM ThinkPad T20 2647. The war news covered the Iraq war and the Afghanistan war. 

To verify the validity of this model, a simple baseline mixture model was also implemented that takes the data of all of the laptops or all of the wars and tries to cluster documents without utiliing the differences in different collections. This cross-collection mixture model works notably better than the baseline.

<a name="data"/>

# Obtaining and Organizing the Data


In order to keep the study as close as possible to the paper, the reviews available in this repository were collected based on the description of the paper. The war news was collected from articles from BBC and CNN for one year starting from November 2001. On the other hand, the laptop reviews were pulled from epinions.com. However, epinions.com is no longer available at the time this project was completed. In order to maintain as accurate of a reproduction of this model as possible, a internet archiving website was used to see what was available on epinions.com in 2001. 

The data is organized inside of the "data" folder of this repository. Each text file in the data folder represent data collected for each individual laptop and war. Inside the data folder, there is another folder called "combined". This folder contains two files: laptops.txt and wars.txt. Laptops.txt contain all of the laptop reviews in one file and wars.txt contain all of the war articles in one file. This was done so that the baseline model could access all of the needed reviews or articles as necessary.

<a name="run"/>

# Running the software


This software uses Python3 and uses numpy. To run the baseline mixture model, ensure that numpy and python are properly installed and run the following code in the terminal:

```
python main.py
```

This will run the baseline model on data/combined/laptops.txt by default. If another data set should be analyzed, provide the file location as a parameter. For example, if analysis wants to be run on war models, run the following:

```
python main.py ./data/combined/wars.txt
```


<a name="code"/>

# Code Information

<a name="baseline"/>

## Baseline model

The code from the baseline model is primarily based on the PLSA algorithm as used in MP3 of CS 410. This model was estimated using the EM (Estimation-Maximization) algorithm. Here is a quick overview of the functions provided:

#### normalize(input_matrix)

Normalizes the rows of a 2d input_matrix so they sum to 1.

### class NaiveModel(object)

Class that actually runs the baseline mixture model. Includes the following methods:

#### build_corpus(self)

Fills in self.documents with a list of list of words by reading from the document path

#### build_vocabulary(self)

Constructs a list of unique works in the whole corpus and updates self.vocabulary

#### build_term_doc_matrix(self)

Constructs a term document matrix where each row represents a document, and each column represents a vocabulary term.


#### initialize_randomly(self, number_of_topics)

Randomly sets the normalized matrices self.document_topic_prob, self.topic_word_prob, and self.background_prob.

#### initialize_uniformly(self, number_of_topics)

Uniformly sets the normalized matrices self.document_topic_prob, self.topic_word_prob, and self.background_prob.

#### initialize(self, number_of_topics, random=False)

Sets up the matrices of the model using initalize_randomly or initialize_uniformly.

#### expectation_step(self)

Runs the expectation_step as part of the EM algorithm. 

#### maximization_step(self, number_of_topics)

Runs the maximization_step as part of the EM algorithm.

#### calculate_likelihood(self, number_of_topics)

Calculates the log-likelihood of the model using the model's updated probability matrices. Used to determine when the EM algorithm is complete/converged.

#### naivemodel(self, number_of_topics, max_iter, epsilon)

Runs the model in its entirety on self.document_path and the provided parameters.

#### main(documents_path)

This is the default function used when running from the terminal. Runs the model with default parameters.

<a name="ccm"/>

## Cross Collection mixture model

The code from the Cross Collection model is primarily based on the PLSA algorithm as used in MP3 of CS 410. This model was estimated using the EM (Estimation-Maximization) algorithm. Here is a quick overview of the functions provided (functions shared with baseline are omitted):

### class CCModel(object)

Class that actually runs the Cross Collection mixture model. Includes the following methods:

#### build_corpus(self)

Fills in self.documents with a list of list of words by reading from the document path

#### build_vocabulary(self)

Constructs a list of unique works in the whole corpus and updates self.vocabulary

#### build_term_doc_matrix(self)

Constructs a term document matrix where each row represents a document, and each column represents a vocabulary term.


#### initialize_randomly(self, number_of_topics)

Randomly sets the normalized matrices self.document_topic_prob, self.topic_word_prob, and self.background_prob.

#### initialize_uniformly(self, number_of_topics)

Uniformly sets the normalized matrices self.document_topic_prob, self.topic_word_prob, and self.background_prob.

#### initialize(self, number_of_topics, random=False)

Sets up the matrices of the model using initalize_randomly or initialize_uniformly.

#### expectation_step(self)

Runs the expectation_step as part of the EM algorithm. 

#### maximization_step(self, number_of_topics)

Runs the maximization_step as part of the EM algorithm.

#### calculate_likelihood(self, number_of_topics)

Calculates the log-likelihood of the model using the model's updated probability matrices. Used to determine when the EM algorithm is complete/converged.

#### naivemodel(self, number_of_topics, max_iter, epsilon)

Runs the model in its entirety on self.document_path and the provided parameters.

#### main(documents_path)

This is the default function used when running from the terminal. Runs the model with default parameters.

