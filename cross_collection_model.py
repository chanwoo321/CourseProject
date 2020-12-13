"""
This file attempts to recreate the solution presented in the paper "A Cross-Collection Mixture Model for Comparitive Text Mining"
as a final project for CS 410 at the University of Illinois at Urbana-Champaign.

Created by Jonathan Kim, Michael Xiang, and Tyler Ruckstaetter.
"""

import math

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
    """
    Normalizes the columns of a 2d input matrix so they sum to 1
    """

    temp = input_matrix.copy()
    return normalize(temp.T).T

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

        # open_file = open(self.documents_path, 'r')
        # new_documents = []
        # contents = open_file.readlines()
        # for line in range(len(contents)):
        #     doc = []
        #     for word in contents[line].split():
        #         doc.append(word)
        #     new_documents.append(doc)
        # self.documents = new_documents
        # self.number_of_documents = len(new_documents)

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
        # new_term_doc_matrix = np.zeros((self.number_of_documents, self.vocabulary_size))
        #
        # open_file = open(self.documents_path)
        # contents = open_file.readlines()
        # for line in range(len(contents)):
        #     words_dict = {}
        #     for word in contents[line].split():
        #         if word in words_dict:
        #             words_dict[word] += 1
        #         else:
        #             words_dict[word] = 1
        #     for key in words_dict:
        #         new_term_doc_matrix[line, self.vocabulary.index(key)] = words_dict[key]
        # self.term_doc_matrix = new_term_doc_matrix

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
        # self.number_of_topics = number_of_topics
        #
        # self.document_topic_prob = np.random.random_sample((self.number_of_documents, number_of_topics))
        # self.topic_word_prob = np.random.random_sample((number_of_topics, self.vocabulary_size))
        # self.background_word_prob = np.random.random_sample(self.vocabulary_size)
        #
        # self.document_topic_prob = normalize(self.document_topic_prob)
        # self.topic_word_prob = normalize(self.topic_word_prob)
        # self.background_word_prob = normalize(self.background_word_prob)

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

        # if random:
        #     self.initialize_randomly(number_of_topics)
        # else:
        #     self.initialize_uniformly(number_of_topics)
        self.initialize_randomly(number_of_topics)

    def expectation_step(self):
        """ The E-step updates P(z | w, d)
        """

        # for doc in range(self.number_of_documents):
        #     for word in range(self.vocabulary_size):
        #         topic_prob_sum = 0.0
        #
        #         for topic in range(self.number_of_topics):
        #             self.topic_prob[doc, topic, word] = self.topic_word_prob[topic, word] * self.document_topic_prob[doc, topic]
        #             topic_prob_sum += self.topic_prob[doc, topic, word]
        #
        #         if topic_prob_sum == 0:
        #             for topic in range(self.number_of_topics):
        #                 self.topic_prob[doc, topic, word] = 0
        #             self.background_prob[doc, word] = 1
        #
        #         else:
        #             self.topic_prob[doc,:,word] /= topic_prob_sum
        #             curr_back_prob = self.background_word_prob[word]
        #             back_sum = self.b_lambda * curr_back_prob + ((1 - self.b_lambda) * topic_prob_sum)
        #             self.background_prob[doc, word] = self.b_lambda * curr_back_prob / back_sum

        # updates j and B
        for collection in range(self.number_of_collections):
            for doc in range(self.number_of_documents_per_collection[collection]):
                for word in range(self.vocabulary_size):
                    topic_prob_sum = 0.0

                    for topic in range(self.number_of_topics):
                        # build numberator of p(z,c,w = j)
                        self.topic_prob_j[collection][doc,topic,word] = self.document_topic_prob[collection][doc, topic]
                        to_mult = self.c_lambda * self.topic_word_prob[topic,word]
                        to_mult += (1 - self.c_lambda) * self.topic_word_prob_per_collection[collection][topic, word]
                        self.topic_prob_j[collection][doc,topic,word] *= to_mult

                        topic_prob_sum += self.topic_prob_j[collection][doc,topic,word]

                        # fill out C!
                        self.topic_prob_C[collection][doc,topic,word] = self.c_lambda * self.topic_word_prob[topic, word]
                        denom = self.topic_prob_C[collection][doc,topic,word]
                        denom += (1 - self.c_lambda) * self.topic_word_prob_per_collection[collection][topic, word]
                        self.topic_prob_C[collection][doc,topic,word] /= denom

                    if topic_prob_sum == 0:
                        """
                        IDK if this part needs to be implemented - left a print statement in case it does
                        """
                        print("please implement something for this case")
                        assert(False)
                    else:
                        self.topic_prob_j[collection][doc,topic,word] /= topic_prob_sum

                        # fill out the background)
                        self.topic_prob_B[collection][doc,word] = self.b_lambda * self.background_word_prob[word]
                        denom = self.topic_prob_B[collection][doc,word]
                        denom += (1 - self.b_lambda) * topic_prob_sum
                        self.topic_prob_B[collection][doc,word] /= denom
        #print(self.topic_prob_j)



    def maximization_step(self, number_of_topics):
        """ The M-step updates P(w | z)
        """
        # # print("M step:")
        # for z in range(0, number_of_topics):
        #     for j in range(0, self.vocabulary_size):
        #         sum = 0
        #         for d_index in range(0, len(self.documents)):
        #             sum += self.term_doc_matrix[d_index, j] * self.topic_prob[d_index, z, j] * (1 - self.background_prob[d_index, j])
        #         self.topic_word_prob[z, j] = sum
        # self.topic_word_prob = normalize(self.topic_word_prob)
        #
        # # Update the background_word_prob
        # for j in range(0, self.vocabulary_size):
        #     sum = 0
        #     for d_index in range(0, len(self.documents)):
        #         sum += self.term_doc_matrix[d_index, j] * self.background_prob[d_index, j]
        #     self.background_word_prob[j] = sum
        #
        # # update P(z | d)
        # for d_index in range(0, len(self.documents)):
        #     for z in range(0, number_of_topics):
        #         sum = 0
        #         for j in range(0, self.vocabulary_size):
        #             sum += self.term_doc_matrix[d_index, j] * self.topic_prob[d_index, z, j] * (1 - self.background_prob[d_index, j])
        #         self.document_topic_prob[d_index, z] = sum
        # #print(self.document_topic_prob[0])
        # self.document_topic_prob = normalize(self.document_topic_prob)

        # update pi, which is document_topic_prob[collection][doc, topic]

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
        for z in range(0, number_of_topics):
            for j in range(self.vocabulary_size):
                sum1 = 0
                # sum2 = 0
                for collection in range(self.number_of_collections):
                    for doc in range(self.number_of_documents_per_collection[collection]):
                        prod = self.term_doc_matrix[collection][doc,word] * (1 - self.topic_prob_B[collection][doc,word])
                        prod *= self.topic_prob_j[collection][doc,topic,word]
                        sum1 += prod * self.topic_prob_C[collection][doc,topic,word]
                        # sum2 += prod * (1 - self.topic_prob_C[collection][doc,topic,word])
                self.topic_word_prob[z, j] = sum1
                # bad I think this line below is weird, topic word prob per collection doesnt seem to get updated
                # self.topic_word_prob_per_collection ########################################################
        for collection in range(self.number_of_collections):
            for z in range(0, number_of_topics):
                for j in range(self.vocabulary_size):
                    sum2 = 0
                        for doc in range(self.number_of_documents_per_collection[collection]):
                            prod = self.term_doc_matrix[collection][doc,word] * (1 - self.topic_prob_B[collection][doc,word])
                            prod *= self.topic_prob_j[collection][doc,topic,word]
                            sum2 += prod * (1 - self.topic_prob_C[collection][doc,topic,word])
                    # bad I think this line below is weird, topic word prob per collection doesnt seem to get updated
                    self.topic_word_prob_per_collection[collection][z, j] = sum2

        self.topic_word_prob = normalize(self.topic_word_prob)
        # print(type(self.topic_word_prob))
        # print(type(self.topic_word_prob_per_collection))
        # print(type(self.topic_word_prob_per_collection))
        for collection in range(self.number_of_collections):
            self.topic_word_prob_per_collection[collection] = normalize(self.topic_word_prob_per_collection[collection])


    def calculate_likelihood(self, number_of_topics):
        """ Calculate the current log-likelihood of the model using
        the model's updated probability matrices
        Append the calculated log-likelihood to self.likelihoods
        """

        log_likelihood = 0
        for doc in range(self.number_of_documents):
            for word in range(self.vocabulary_size):
                inner_sum = 0
                for topic in range(number_of_topics):
                    inner_sum += self.document_topic_prob[doc, topic] * self.topic_word_prob[topic, word]
                total_prob = self.background_word_prob[word] * self.b_lambda + (1 - self.b_lambda) * inner_sum
                log_likelihood += self.term_doc_matrix[doc, word] * math.log(total_prob)
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

        for iteration in range(20):
            print("Iteration #" + str(iteration + 1) + "...")

            self.expectation_step()
            self.maximization_step(number_of_topics)
            # self.calculate_likelihood(number_of_topics)
            # current_likelihood = self.likelihoods[-1]
            # if iteration > 2:
            #     if abs(self.likelihoods[-2] - self.likelihoods[-1]) < epsilon:
            #         break

        return self.topic_word_prob, self.topic_word_prob_per_collection

def show_top_10(matrix, model, number_of_topics):
    prob_dict = dict()

    for j in range(number_of_topics):
        prob_dict[j] = list()

        for i in range(len(matrix[j])):
            if matrix[j][i] != 0:
                # if the word prob != 0 for a topic, add to topic dict
                prob_dict[j].append((model.vocabulary[i], matrix[j][i]))

    for topic in range(number_of_topics):
        df = pd.DataFrame(prob_dict[topic], columns = ['word','probability'])
        df = df.sort_values(by='probability', ascending=False)
        print(list(df.head(10).to_records(index=False))) # get the top 10 words by their probability in topic 0


def main():
    documents_path = './data/combined/wars.txt'
    collections = ['./data/afghanistan.txt', './data/iraq.txt']
    print("File path: " + documents_path)
    model = CCModel(documents_path, collections)
    model.build_corpus()
    model.build_vocabulary()
    print("Vocabulary size:" + str(len(model.vocabulary)))
    print("Number of documents:" + str(len(model.documents)))
    number_of_topics = 5
    max_iterations = 200
    epsilon = 0.001
    topic_word, coll_topic_word = model.ccmodel(number_of_topics, max_iterations, epsilon)

    show_top_10(topic_word, model, number_of_topics)
    show_top_10(coll_topic_word, model, len(collections))

if __name__ == '__main__':
    main()
