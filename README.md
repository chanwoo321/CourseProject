# CourseProject

Please fork this repository and paste the github link of your fork on Microsoft CMT. Detailed instructions are on Coursera under Week 1: Course Project Overview/Week 9 Activities.

# Table of Contents

[Introduction](#intro)

[Brief Overview](#overview)

[Obtaining and Organizing the Data](#data)

[Running the Software](#run)

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

This will run the baseline model.
