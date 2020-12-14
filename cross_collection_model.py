"""
This file attempts to recreate the solution presented in the paper "A Cross-Collection Mixture Model for Comparitive Text Mining"
as a final project for CS 410 at the University of Illinois at Urbana-Champaign.

Created by Jonathan Kim, Michael Xiang, and Tyler Ruckstaetter.
"""

import math
import sys
import numpy as np
import pandas as pd


def normalize(input_matrix):
    """
    Normalizes the rows of a 2d input_matrix so they sum to 1
    """
    if len(input_matrix.shape) == 1:
        return input_matrix / input_matrix.sum()

    row_sums = input_matrix.sum(axis=1)
    try:
        assert (np.count_nonzero(row_sums)==np.shape(row_sums)[0]) # no row should sum to zero
    except Exception:
        raise Exception("Error while normalizing. Row(s) sum to zero")
    new_matrix = input_matrix / row_sums[:, np.newaxis]

    return new_matrix

def normalize_c(input_matrix):


        col_sums = np.nan_to_num(input_matrix).sum(axis=0, keepdims=True)

        new_matrix = np.divide(input_matrix, col_sums)
        return np.nan_to_num(new_matrix)

class CCModel(object):
    """
    Model for topic mining with the baseline mixture model.
    """

    def __init__(self, documents_path, collections, b_lambda=.95, c_lambda=.5):
        self.documents = [] # contains a list of list of list of words.
        self.vocabulary = []
        self.likelihoods = []
        self.documents_path = documents_path # path to a collective corpus
        self.collections = collections # path to a list of individual collections that make up a corpus
        self.term_doc_matrix = [] # Given [i][j,k], returns the count of word k in doc j of collection i.
        self.document_topic_prob = [] # Given [i][j,k], gives the probability document j in collection i is of topic k
        self.topic_word_prob = None # Given [i][j], gives the probability that word j is of topic i
        self.topic_prob_j = []
        self.topic_prob_C = []
        self.topic_prob_B = []
        self.background_word_prob = None
        self.background_prob = None

        self.topic_word_prob_per_collection = []


        self.number_of_documents = 0
        self.number_of_documents_per_collection = []
        self.number_of_collections = len(collections)
        self.vocabulary_size = 0

        self.b_lambda = b_lambda
        self.c_lambda = c_lambda
        self.number_of_topics = 0

    def build_corpus(self):
        """
        Read document, fill in self.documents, a list of list of words
        self.documents = [["word", "word", "word2", ...], [], [],...]

        3-D list that contains the documents in an easy to read format. Collection -> document number -> word
        """

        for doc_num in range(len(self.collections)):
            open_file = open(self.collections[doc_num], 'r')
            new_documents = []
            contents = open_file.readlines()
            for line in range(len(contents)):
                doc = []
                for word in contents[line].split():
                    doc.append(word)
                new_documents.append(doc)
            self.documents.append(new_documents)
            self.number_of_documents += len(new_documents)
            self.number_of_documents_per_collection.append(len(new_documents))

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

        UPDATE: self.term_doc_matrix[i][j,k] is the count of term k in document j in collection i
        Notice that it is a list of 2D numpy arrays to make a 3D data structure!
        """

        for collection in range(self.number_of_collections):
            new_term_doc_matrix = np.zeros((self.number_of_documents_per_collection[collection], self.vocabulary_size))
            open_file = open(self.collections[collection])
            contents = open_file.readlines()
            for line in range(len(contents)):
                words_dict = {}
                for word in contents[line].split():
                    if word in words_dict:
                        words_dict[word] += 1
                    else:
                        words_dict[word] = 1
                for key in words_dict:
                    new_term_doc_matrix[line, self.vocabulary.index(key)] = words_dict[key]
            self.term_doc_matrix.append(new_term_doc_matrix)



    def initialize_randomly(self, number_of_topics):
        """
        Randomly initializes the matrices self.document_topic_prob, self.topic_word_prob, and self.background_prob
        with a random probability distribution
        """

        self.number_of_topics = number_of_topics

        for i in range(self.number_of_collections):
            self.document_topic_prob.append(normalize(np.random.random_sample((self.number_of_documents_per_collection[i], number_of_topics))))

        self.topic_word_prob = normalize(np.random.random_sample((number_of_topics, self.vocabulary_size)))

        for i in range(self.number_of_collections):
            self.topic_word_prob_per_collection.append(normalize(np.random.random_sample((number_of_topics, self.vocabulary_size))))
        # Given [i][j,k], gives the the probability that word k is of topic j of collection i

        self.background_word_prob = normalize(np.random.random_sample(self.vocabulary_size))

    def initialize(self, number_of_topics, random=False):
        """ Call the functions to initialize the matrices document_topic_prob and topic_word_prob

        UPDATE: Made it always random
        """
        print("Initializing...")

        self.initialize_randomly(number_of_topics)



    def expectation_step(self):
        """ The E-step updates P(z | w, d)
        """

        # updates j and B

        for collection in range(self.number_of_collections):

            for doc in range(self.number_of_documents_per_collection[collection]):

                # j formula here
                to_mult_j = (1 - self.c_lambda) * self.topic_word_prob_per_collection[collection]
                to_mult_j += (self.c_lambda *  self.topic_word_prob)
                to_mult_j = np.transpose([self.document_topic_prob[collection][doc]]) * to_mult_j

                back = self.b_lambda * self.background_word_prob

                # save denominator for background prob calculation
                denom = back + ((1 - self.b_lambda) * to_mult_j.sum(axis=0, keepdims=True))

                # remember to normalize
                to_mult_j = normalize_c(to_mult_j)
                self.topic_prob_j[collection][doc] = to_mult_j

                self.topic_prob_B[collection][doc] = np.nan_to_num(np.divide(back, denom))

                # put C formula here
                c_lamb = self.c_lambda * self.topic_word_prob
                denom = c_lamb + ((1 - self.c_lambda) * self.topic_word_prob_per_collection[collection])
                self.topic_prob_C[collection][doc] = np.nan_to_num(np.divide(c_lamb, denom))

        #print(self.topic_prob_j)



    def maximization_step(self, number_of_topics):
        """ The M-step updates P(w | z)
        """

        for collection in range(self.number_of_collections):
            for doc in range(self.number_of_documents_per_collection[collection]):
                for topic in range(self.number_of_topics):
                    sum = 0
                    for word in range(self.vocabulary_size):
                        sum += self.term_doc_matrix[collection][doc,word] * self.topic_prob_j[collection][doc,topic,word]
                        # outer_sum += self.term_doc_matrix[collection][doc,word] * self.topic_prob_j[collection][doc,topic,word]
                    self.document_topic_prob[collection][doc,topic] = sum
            self.document_topic_prob[collection] = normalize(self.document_topic_prob[collection])

        # # Update the background_word_prob
        for j in range(0, self.vocabulary_size):
            sum = 0
            for collection in range(self.number_of_collections):
                for doc in range(0, self.number_of_documents_per_collection[collection]):
                    sum += self.term_doc_matrix[collection][doc, j] * self.topic_prob_B[collection][doc, j]
            self.background_word_prob[j] = sum
        self.background_word_prob = normalize(self.background_word_prob)

        # update p^(n+1) (...j), which is topic_word_prob[topic, word]
        # also update the otehr one, which is topic_word_prob_per_collection[collection][topic,word]
        for topic in range(0, number_of_topics):
            for word in range(self.vocabulary_size):
                sum1 = 0
                for collection in range(self.number_of_collections):
                    for doc in range(self.number_of_documents_per_collection[collection]):
                        prod = self.term_doc_matrix[collection][doc,word] * (1 - self.topic_prob_B[collection][doc,word])
                        prod *= self.topic_prob_j[collection][doc,topic,word]
                        sum1 += prod * self.topic_prob_C[collection][doc,topic,word]
                        # sum2 += prod * (1 - self.topic_prob_C[collection][doc,topic,word])
                self.topic_word_prob[topic, word] = sum1

        for collection in range(self.number_of_collections):
            for z in range(0, number_of_topics):
                for j in range(self.vocabulary_size):
                    sum2 = 0
                    for doc in range(self.number_of_documents_per_collection[collection]):
                        prod = self.term_doc_matrix[collection][doc,j] * (1 - self.topic_prob_B[collection][doc,j])
                        prod *= self.topic_prob_j[collection][doc,z,j]
                        sum2 += prod * (1 - self.topic_prob_C[collection][doc,z,j])
                    self.topic_word_prob_per_collection[collection][z, j] = sum2

        self.topic_word_prob = normalize(self.topic_word_prob)
        for collection in range(self.number_of_collections):
            self.topic_word_prob_per_collection[collection] = normalize(self.topic_word_prob_per_collection[collection])


    def calculate_likelihood(self, number_of_topics):
        """ Calculate the current log-likelihood of the model using
        the model's updated probability matrices
        Append the calculated log-likelihood to self.likelihoods
        """

        log_likelihood = 0
        for collection in range(self.number_of_collections):
            for doc in range(self.number_of_documents_per_collection[collection]):
                for word in range(self.vocabulary_size):
                    inner_sum = 0
                    for topic in range(number_of_topics):
                        inner_sum += self.document_topic_prob[collection][doc, topic] * self.topic_word_prob[topic, word]
                    total_prob = self.background_word_prob[word] * self.b_lambda + (1 - self.b_lambda) * inner_sum
                    log_likelihood += self.term_doc_matrix[collection][doc, word] * math.log(total_prob)
        self.likelihoods.append(log_likelihood)


    def ccmodel(self, number_of_topics, max_iter, epsilon):

        """
        Model topics.
        """
        print ("EM iteration begins...")

        # build term-doc matrix
        self.build_term_doc_matrix()

        # P(z | d, w)
        # self.topic_prob = np.zeros([self.number_of_documents, number_of_topics, self.vocabulary_size], dtype=np.float)
        # self.background_prob = np.zeros([self.number_of_documents, self.vocabulary_size], dtype=np.float)

        self.topic_prob_j = []
        self.topic_prob_C = []
        self.topic_prob_B = []
        for i in range(self.number_of_collections):
            self.topic_prob_j.append(np.zeros([self.number_of_documents_per_collection[i], number_of_topics, self.vocabulary_size,], dtype=np.float))
            self.topic_prob_C.append(np.zeros([self.number_of_documents_per_collection[i], number_of_topics, self.vocabulary_size,], dtype=np.float))
            self.topic_prob_B.append(np.zeros([self.number_of_documents_per_collection[i], self.vocabulary_size], dtype=np.float))


        # P(z | d) P(w | z)
        self.initialize(number_of_topics, random=True)

        # Run the EM algorithm
        current_likelihood = 0.0

        for iteration in range(max_iter): # change this to number of iterations
            print("Iteration #" + str(iteration + 1) + "...")

            self.expectation_step()
            self.maximization_step(number_of_topics)
            self.calculate_likelihood(number_of_topics)
            current_likelihood = self.likelihoods[-1]
            if iteration > 2:
                if abs(self.likelihoods[-2] - self.likelihoods[-1]) < epsilon:
                    break

        return self.topic_word_prob, self.topic_word_prob_per_collection

def show_top_10(matrix, model):
    prob_dict = dict()

    for j in range(len(matrix)):
        prob_dict[j] = list()

        for i in range(len(matrix[j])):
            if matrix[j][i] != 0:
                # if the word prob != 0 for a topic, add to topic dict
                prob_dict[j].append((model.vocabulary[i], matrix[j][i]))

    for topic in range(len(matrix)):
        df = pd.DataFrame(prob_dict[topic], columns = ['word','probability'])
        df = df.sort_values(by='probability', ascending=False)
        print(list(df.head(10).to_records(index=False))) # get the top 10 words by their probability in topic 0


def main():
    if (len(sys.argv) > 2):
        iterations = sys.argv[1]
        documents_path = sys.argv[2]
        collections = sys.argv[3:]
    else:
        iterations = 10
        documents_path = './data/combined/laptops.txt'
        collections = ['./data/inspiron.txt', './data/mac.txt', './data/thinkpad.txt']

    print("File path: " + documents_path)
    model = CCModel(documents_path, collections)
    model.build_corpus()
    model.build_vocabulary()
    print("Vocabulary size:" + str(len(model.vocabulary)))
    print("Number of collections:" + str(len(model.documents)))
    number_of_topics = 8
    epsilon = 0.001
    topic_word, coll_topic_word = model.ccmodel(number_of_topics, int(iterations), epsilon)

    show_top_10(topic_word, model)

    for collection in range(len(collections)):
        print(collections[collection])
        show_top_10(coll_topic_word[collection], model)

def show_top_10(matrix, model):
    prob_dict = dict()

    for j in range(len(matrix)):
        prob_dict[j] = list()

        for i in range(len(matrix[j])):
            if matrix[j][i] != 0:
                # if the word prob != 0 for a topic, add to topic dict
                prob_dict[j].append((model.vocabulary[i], matrix[j][i]))

    for topic in range(len(matrix)):
        df = pd.DataFrame(prob_dict[topic], columns = ['word','probability'])
        df = df.sort_values(by='probability', ascending=False)
        print(list(df.head(10).to_records(index=False))) # get the top 10 words by their probability in topic 0


if __name__ == '__main__':
    main()
