"""
This file attempts to recreate the solution presented in the paper "A Cross-Collection Mixture Model for Comparitive Text Mining"
as a final project for CS 410 at the University of Illinois at Urbana-Champaign.

Created by Jonathan Kim, Michael Xiang, and Tyler Ruckstaetter.
"""

import numpy as np
import math

def normalize(input_matrix):
    """
    Normalizes the rows of a 2d input_matrix so they sum to 1
    """

    if len(input_matrix.shape) == 1:
        return input_matrix / input_matrix.sum()

    row_sums = input_matrix.sum(axis=1)
    # try:
    #     assert (np.count_nonzero(row_sums)==np.shape(row_sums)[0]) # no row should sum to zero
    # except Exception:
    #     raise Exception("Error while normalizing. Row(s) sum to zero")
    if (np.count_nonzero(row_sums)==np.shape(row_sums)[0]):
        return input_matrix
    new_matrix = input_matrix / row_sums[:, np.newaxis]
    return new_matrix

def normalize_c(input_matrix):
    """
    Normalizes the columns of a 2d input matrix so they sum to 1
    """

    temp = input_matrix.copy()
    return normalize(temp.T).T

class NaiveModel(object):
    """
    Model for topic mining with the baseline mixture model.
    """

    def __init__(self, documents_path, b_lambda=.8):
        self.documents = []
        self.vocabulary = []
        self.likelihoods = []
        self.documents_path = documents_path
        self.term_doc_matrix = None # P(z | d)
        self.document_topic_prob = None  # P(w | z)
        self.topic_word_prob = None  # P(z | d, w)
        self.topic_prob = None # P(w | B)
        self.background_word_prob = None

        self.number_of_documents = 0
        self.vocabulary_size = 0

        self.b_lambda = b_lambda
        self.number_of_topics = 0

    def build_corpus(self):
        """
        Read document, fill in self.documents, a list of list of words
        self.documents = [["word", "word", "word2", ...], [], [],...]
        """

        open_file = open(self.documents_path, 'r')
        new_documents = []
        contents = open_file.readlines()
        for line in range(len(contents)):
            doc = []
            for word in contents[line].split():
                doc.append(word)
            new_documents.append(doc)
        self.documents = new_documents
        self.number_of_documents = len(new_documents)

    def build_vocabulary(self):
        """
        Construct a lit of unique words in the whole corpus, and puts it in self.vocabulary
        Also updates self.vocabulary_size
        Ex) ["word1", "word2", "word3",...]
        """
        words_dict = set()
        new_vocabulary = []
        open_file = open(self.documents_path, 'r')
        contents = open_file.readlines()
        for line in range(len(contents)):
            for word in contents[line].split():
                if word in words_dict:
                    continue
                words_dict.add(word)
                new_vocabulary.append(word)

        self.vocabulary = new_vocabulary
        self.vocabulary_size = len(new_vocabulary)

    def build_term_doc_matrix(self):
        """
        Construct the term-document matrix where each row represents a document,
        and each column represents a vocabulary term.
        self.term_doc_matrix[i][j] is the count of term j in document i
        """
        new_term_doc_matrix = np.zeros((self.number_of_documents, self.vocabulary_size))

        open_file = open(self.documents_path)
        contents = open_file.readlines()
        for line in range(len(contents)):
            words_dict = {}
            for word in contents[line].split():
                if word in words_dict:
                    words_dict[word] += 1
                else:
                    words_dict[word] = 1
            for key in words_dict:
                new_term_doc_matrix[line][self.vocabulary.index(key)] = words_dict[key]
        self.term_doc_matrix = new_term_doc_matrix

    def initialize_randomly(self, number_of_topics):
        """
        Randomly initializes the matrices self.document_topic_prob, self.topic_word_prob, and self.background_prob
        with a random probability distribution
        """
        self.number_of_topics = number_of_topics

        self.document_topic_prob = np.random.random_sample((self.number_of_documents, number_of_topics))
        self.topic_word_prob = np.random.random_sample((number_of_topics, self.vocabulary_size))
        self.background_word_prob = np.random.random_sample(self.vocabulary_size)

        self.document_topic_prob = normalize(self.document_topic_prob)
        self.topic_word_prob = normalize(self.topic_word_prob)
        self.background_word_prob = normalize(self.background_word_prob)

    def initialize_uniformly(self, number_of_topics):
        """
        Initializes the matrices: self.document_topic_prob and self.topic_word_prob with a uniform
        probability distribution. This is used for testing purposes.
        DO NOT CHANGE THIS FUNCTION
        """
        self.number_of_topics = number_of_topics
        self.document_topic_prob = np.ones((self.number_of_documents, number_of_topics))
        self.topic_word_prob = np.ones((number_of_topics, len(self.vocabulary)))
        self.background_word_prob = np.ones((self.m))

        self.document_topic_prob = normalize(self.document_topic_prob)
        self.topic_word_prob = normalize(self.topic_word_prob)
        self.background_word_prob = normalize(self.background_word_prob)

    def initialize(self, number_of_topics, random=False):
        """ Call the functions to initialize the matrices document_topic_prob and topic_word_prob
        """
        print("Initializing...")

        if random:
            self.initialize_randomly(number_of_topics)
        else:
            self.initialize_uniformly(number_of_topics)

    def expectation_step(self):
        """ The E-step updates P(z | w, d)
        """
        print("E step:")
        for doc in range(self.number_of_documents):
            for word in range(self.vocabulary_size):
                fm = 0.0
                for topic in range(self.number_of_topics):
                    self.topic_prob[doc, word, topic] = self.topic_word_prob[topic, word] * self.document_topic_prob[doc, topic]
                    fm += self.topic_prob[doc, word, topic]
                if fm == 0:
                    for topic in range(self.number_of_topics):
                        self.topic_prob[doc, word, topic] = 0
                    self.background_prob[doc, word] = 1
                else:
                    self.topic_prob[doc] = normalize_c(self.topic_prob[doc])
                    curr_back_prob = self.background_word_prob[word]
                    back_fm = self.b_lambda * curr_back_prob + (1 - self.b_lambda) * fm
                    self.background_prob[doc, word] = self.b_lambda * curr_back_prob / back_fm

    def maximization_step(self, number_of_topics):
        """ The M-step updates P(w | z)
        """
        print("M step:")

        new_topic_word_prob = np.zeros(self.topic_word_prob.shape)
        for topic in range(number_of_topics):
            for word in range(self.vocabulary_size):
                for doc in range(self.number_of_documents):
                    new_topic_word_prob += self.term_doc_matrix[topic, word] * self.topic_prob[doc, word, topic]
        self.topic_word_prob = normalize(new_topic_word_prob)

        new_document_topic_prob = np.zeros(self.document_topic_prob.shape)
        for doc in range(self.number_of_documents):
            for topic in range(number_of_topics):
                for word in range(self.vocabulary_size):
                    new_document_topic_prob[doc, topic] += self.term_doc_matrix[doc, word] * self.topic_prob[doc, word, topic]

        self.document_topic_prob = normalize(self.document_topic_prob)

    def calculate_likelihood(self, number_of_topics):
        """ Calculate the current log-likelihood of the model using
        the model's updated probability matrices
        Append the calculated log-likelihood to self.likelihoods
        """
        # ############################
        # your code here
        # ############################

        log_likelihood = 0
        for doc in range(self.number_of_documents):
            for word in range(self.vocabulary_size):
                inner_sum = 0
                for topic in range(number_of_topics):
                    inner_sum += self.document_topic_prob[doc][topic] * self.topic_word_prob[topic][word]
                total_prob = self.background_word_prob[word] * self.b_lambda + (1 - b_lambda) * inner_sum
                log_likelihood += self.term_doc_matrix[doc][word] * math.log(total_prob)
        self.likelihoods.append(log_likelihood)

    def naivemodel(self, number_of_topics, max_iter, epsilon):

        """
        Model topics.
        """
        print ("EM iteration begins...")

        # build term-doc matrix
        self.build_term_doc_matrix()

        # Create the counter arrays.

        # P(z | d, w)
        self.topic_prob = np.zeros([self.number_of_documents, number_of_topics, self.vocabulary_size], dtype=np.float)
        self.background_prob = np.zeros([self.number_of_documents, self.vocabulary_size], dtype=np.float)

        # P(z | d) P(w | z)
        self.initialize(number_of_topics, random=True)

        # Run the EM algorithm
        current_likelihood = 0.0

        for iteration in range(max_iter):
            print("Iteration #" + str(iteration + 1) + "...")

            # ############################
            # your code here
            # ############################

            self.expectation_step()
            self.maximization_step(number_of_topics)
            self.calculate_likelihood(number_of_topics)
            current_likelihood = self.likelihoods[-1]
            if iteration > 2:
                print(self.likelihoods[-2])
                print(self.likelihoods[-1])
                if abs(self.likelihoods[-2] - self.likelihoods[-1]) < .0001:
                    break


def main():
    documents_path = './data/mac.txt'
    model = NaiveModel(documents_path)
    model.build_corpus()
    model.build_vocabulary()
    print(model.vocabulary)
    print("Vocabulary size:" + str(len(model.vocabulary)))
    print("Number of documents:" + str(len(model.documents)))
    number_of_topics = 2
    max_iterations = 200
    epsilon = 0.001
    model.naivemodel(number_of_topics, max_iterations, epsilon)


if __name__ == '__main__':
    main()