from __future__ import division
import re
import tensorflow as tf
import collections
import numpy as np
import itertools
import random

data_home = "./"
def load_sst_data(path):
    data = []
    with open(path) as f:
        for i, line in enumerate(f):
            example = {}
            text = re.sub(r'\s*(\(\d)|(\))\s*', '', line)
            example["text"] = text[1:]
            data.append(example)
    return data


''' this function is for get the cooccurrences of all word pair''' 
def extract_cooccurrences(dataset, vocab, context_size, word_to_index_map):
    cooccurrence = np.zeros([len(vocab), len(vocab)])
    non_zero_pair = set()
    start = 0; limit = 10000000
    print("Extracing co-occurrence matrix, with %s words" %limit)
    for text in dataset:
        text_word = text["text"].split() # split text with space
        for word_index in range(limit): #range(len(text_word))
            if word_index % 100000 == 0:
                print(word_index)
            target_word = text_word[word_index]
            if target_word not in word_to_index_map:
                    continue
            min_search_index = max(0, word_index - context_size)
            max_search_index = min(word_index + context_size + 1, len(text_word))
            for search_word_index in range(min_search_index, word_index) + range(word_index+1, max_search_index):
                context_word = text_word[search_word_index]
                if context_word not in word_to_index_map:
                    continue
                row = word_to_index_map[target_word]
                column = word_to_index_map[context_word] 
                cooccurrence[row][column] += 1
                non_zero_pair.add((row, column))
    return cooccurrence, non_zero_pair

def dataset_prepare():
    training_set = load_sst_data(data_home + "tree/train.txt")
    word_counter = collections.Counter()
    for example in training_set:
        word_counter.update(example["text"].split())
    vocabulary = [pair[0] for pair in word_counter.most_common()[:10000]]
    index_to_word_map = dict(enumerate(vocabulary))
    word_to_index_map = dict([(index_to_word_map[index], index) for index in index_to_word_map])
    cooccurrence, non_zero_pair = extract_cooccurrences(training_set, vocabulary, 4, word_to_index_map)
    return cooccurrence, non_zero_pair, len(vocabulary), vocabulary, word_to_index_map, index_to_word_map

def textpure_dataset_prepare():
    data_analogy = data_home + "text8"
    file = open(data_analogy, "rb")
    training_set = [{}]

    for line in file:
        training_set[0]["text"] = line
        word_counter = collections.Counter(line.strip().split())
    vocabulary = [pair[0] for pair in word_counter.most_common()[:10000]]
    index_to_word_map = dict(enumerate(vocabulary))
    word_to_index_map = dict([(index_to_word_map[index], index) for index in index_to_word_map])
    cooccurrence, non_zero_pair = extract_cooccurrences(training_set, vocabulary, 4, word_to_index_map)

    return cooccurrence, non_zero_pair, len(vocabulary), vocabulary, word_to_index_map, index_to_word_map


def test_dataset_prepare(word_to_index_map):
    '''load the analogy test set, 
    return semantic and syntactic questions with word indices
    '''

    file = open("questions-words.txt")
    topic = 0
    skip_question = 0
    semantic = []; syntactic = []; all_word = set()
    for line in file:
        word = line.strip().lower().split()
        word_index = [word_to_index_map.get(w.strip()) for w in word]
        if ":" in line:
            topic += 1
            continue
        if None in word_index or len(word_index) != 4:
            skip_question += 1
        else:
            if topic < 6:
                semantic.append(np.array(word_index))
            else:
                syntactic.append(np.array(word_index))
        all_word |= set(word)
    print("Analogy questions:", len(semantic) + len(syntactic))
    print("Skipped:", skip_question)
    return np.array(semantic), np.array(syntactic), all_word
        



class glove(object):
    def __init__(self, num_word):
        # hyperparameters:
        self.dim           = 100         # embedding dimensions
        self.num_word      = num_word   # vocabulary size
        self.alpha         = 0.75       # weight function alpha
        self.xmax          = 50         # weight function xmax
        self.learning_rate = 0.5        # learning rate for sgd
        self.batch_size    = 1024       # batch size
        self.epoch         = 10000       # total epoch to train
        self.embedding     = None
        self.display_epoch = 10
        self.test_top_n    = 100        # check whether the true word in the most likely n word
        print("Settings:")
        print("Embedding dim: %s" %self.dim)
        print("Vocab size: %s" %self.num_word)
        print("learning rate: %s" %self.learning_rate)
        print("test top n word : %s" %self.test_top_n)
        with tf.device('/cpu:0'):
            # glove model parameters: J = sum_i,j f(X_i,j) * W_i^T dot W_j + b_i + b_j - log X_i,j 
            self.count = tf.placeholder(tf.float32, shape = [self.batch_size])
            self.X_i   = tf.placeholder(tf.int32, shape = [self.batch_size]) # hold the target word index
            self.X_j   = tf.placeholder(tf.int32, shape = [self.batch_size]) # hold the context word index
            
            # W are word vectors, b are biases
            self.word_vec   = tf.Variable(tf.random_normal([self.num_word, self.dim], mean = 0, stddev = 1))
            self.context_word_vec   = tf.Variable(tf.random_normal([self.num_word, self.dim], mean = 0, stddev = 1))
            self.b_i   = tf.Variable(tf.random_normal([self.num_word], mean = 0, stddev = 1))
            self.b_j   = tf.Variable(tf.random_normal([self.num_word], mean = 0, stddev = 1))

            # setup input to the model with batchsize
            self.word_vec_input = tf.nn.embedding_lookup([self.word_vec], self.X_i)
            self.context_word_vec_input = tf.nn.embedding_lookup([self.context_word_vec], self.X_j)
            self.b_i_input = tf.nn.embedding_lookup([self.b_i], self.X_i)
            self.b_j_input = tf.nn.embedding_lookup([self.b_j], self.X_j)
            
            
            
            # define weight function and loss function
            self.weight_function = tf.minimum(1.0, tf.div(self.count, self.xmax) ** 0.75)
            self.weight_product = tf.reduce_sum(tf.mul(self.word_vec_input, self.context_word_vec_input), 1) 
            self.loss_2 = tf.square(self.weight_product+ self.b_i_input + self.b_j_input - tf.log(self.count))
            self.loss = tf.mul(self.weight_function, self.loss_2)
            self.total_loss = tf.reduce_mean(self.loss)
            
            
            # define optimizer and final word embedding
            self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.total_loss)
            self.final_word_embedding = tf.add(self.word_vec, self.context_word_vec) 
            
            
            ''' evaluate on test set'''

            # contain the evaluate word id
            self.analogy_a = tf.placeholder(dtype = tf.int32)
            self.analogy_b = tf.placeholder(dtype = tf.int32)
            self.analogy_c = tf.placeholder(dtype = tf.int32)

            # normalize the word embedding over the 2 dimension
            normal_emb = tf.nn.l2_normalize(self.final_word_embedding, 1)

            a_emb = tf.gather(normal_emb, self.analogy_a)
            b_emb = tf.gather(normal_emb, self.analogy_b)
            c_emb = tf.gather(normal_emb, self.analogy_c)
            target = c_emb + (b_emb - a_emb) 

            # compute the cosine distance
            dist = tf.matmul(target, normal_emb, transpose_b = True)
            _, self.pred_idex = tf.nn.top_k(dist, self.test_top_n)
            

            self.init = tf.initialize_all_variables()
            self.sess = tf.Session()
            self.sess.run(self.init)

    def train(self, cooccurrence, non_zero_pair, semantic, syntactic, index_to_word_map):
        print("Training")
        for epoch_i in xrange(self.epoch):
            if epoch_i % 1500 == 0:
                self.learning_rate /= 2
            random.shuffle(non_zero_pair)
            total_batch = int(len(non_zero_pair) / self.batch_size)
            ave_cost = 0  
            for batch_i in range(total_batch):
                index = range(batch_i * self.batch_size, (batch_i + 1) * self.batch_size)
                batch_word_count = [cooccurrence[non_zero_pair[x]] for x in index]
                X_i = zip(*non_zero_pair[batch_i * self.batch_size : (batch_i + 1) * self.batch_size])[0]
                X_j = zip(*non_zero_pair[batch_i * self.batch_size : (batch_i + 1) * self.batch_size])[1]

                _, c, self.embedding = self.sess.run([self.optimizer, self.total_loss, self.final_word_embedding], feed_dict = {self.count: batch_word_count, self.X_i: X_i, self.X_j: X_j} )
                ave_cost += c / total_batch
            if epoch_i % self.display_epoch == 0:
                print("Epoch: %s, avg cost: %s, score: " %(epoch_i, ave_cost))
        
                self.test(semantic, syntactic, index_to_word_map)

    def predict(self, analogy):
        ''' predict the top 10 answers for analogy questions. '''
        idx, = self.sess.run([self.pred_idex],feed_dict = {
                            self.analogy_a: analogy[:,0],
                            self.analogy_b: analogy[:,1],
                            self.analogy_c: analogy[:,2]
                            })
        return idx

    def test(self, semantic, syntactic, index_to_word_map):
        score_semantic = 0
        score_syntactic = 0
        total_semantic = semantic.shape[0]
        total_syntactic = syntactic.shape[0]

        start = 0
        while start < total_semantic:
            limit = start + 2500
            sub = semantic[start:limit]
            idx = self.predict(sub)
            start = limit
            for question in xrange(sub.shape[0]):
                for j in xrange(self.test_top_n):
                    if idx[question, j] == sub[question, 3]:
                        score_semantic += 1
                        break
                    elif idx[question, j] in sub[question, :3]:
                        continue
                    else:
                        break
        print("Evaluation %4d/%d accuracy = %4.1f%%" % (score_semantic, total_semantic, score_semantic * 100.0 / total_semantic))
        start = 0
        while start < total_syntactic:
            limit = start + 2500
            sub = syntactic[start:limit]
            idx = self.predict(sub)
            start = limit
            for question in xrange(sub.shape[0]):  
                for j in xrange(self.test_top_n):
                    if idx[question, j] == sub[question, 3]:
                        score_syntactic += 1
                        break
                    elif idx[question, j] in sub[question, :3]:
                        continue
                    else:
                        break
        print("Evaluation %4d/%d accuracy = %4.1f%%" % (score_syntactic, total_syntactic, score_syntactic * 100.0 / total_syntactic))




def main():
    cooccurrence, non_zero_pair, num_word, vocabulary, word_to_index_map, index_to_word_map = textpure_dataset_prepare()#dataset_prepare()
    semantic, syntactic, all_word = test_dataset_prepare(word_to_index_map)
    gloveptb = glove(num_word)
    gloveptb.train(cooccurrence, list(non_zero_pair), semantic, syntactic, index_to_word_map)
    gloveptb.test(semantic, syntactic, index_to_word_map)

if __name__ == "__main__":
    main()

