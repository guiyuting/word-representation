import re
import collections
import numpy as np
import random
import tensorflow as tf

sst_home = "../trees" 

# Let's do 2-way positive/negative classification instead of 5-way
easy_label_map = {0:0, 1:0, 2:None, 3:1, 4:1}

def load_sst_data(path):
    data = []
    with open(path) as f:
        for i, line in enumerate(f): 
            example = {}
            # Strip out the parse information and the phrase labels---we don't need those here
            text = re.sub(r'\s*(\(\d)|(\))\s*', '', line)
            example['text'] = text[1:]
            data.append(example)

    random.seed(1)
    random.shuffle(data)
    return data
     

def sentence_to_padded_index_sequence(datasets):
    '''Annotates datasets with feature vectors.'''
    
    START = "<S>"
    END = "</S>"
    END_PADDING = "<PAD>"
    UNKNOWN = "<UNK>"
    SEQ_LEN = 21
    
    # Extract vocabulary
    def tokenize(string):
        return string.lower().split()
    
    word_counter = collections.Counter()
    for example in datasets[0]:
        word_counter.update(tokenize(example['text']))
    
    vocabulary = set([word for word in word_counter if word_counter[word] > 25])
    vocabulary = list(vocabulary)
    vocabulary = [START, END, END_PADDING, UNKNOWN] + vocabulary
    word_indices = dict(zip(vocabulary, range(len(vocabulary))))
    indices_to_words = {v: k for k, v in word_indices.items()}
        
    for i, dataset in enumerate(datasets):
        for example in dataset: # example is a dict{"text": <S> I like apple</S>, "index_sequence": [0,45,2,3,1]}
            example['index_sequence'] = np.zeros((SEQ_LEN), dtype=np.int32)
            
            token_sequence = [START] + tokenize(example['text']) + [END]
            
            for i in range(SEQ_LEN):
                if i < len(token_sequence):
                    if token_sequence[i] in word_indices:
                        index = word_indices[token_sequence[i]]
                    else:
                        index = word_indices[UNKNOWN]
                else:
                    
                    index = word_indices[END_PADDING]
                example['index_sequence'][i] = index
    return indices_to_words, word_indices
    


class LanguageModel:
    def __init__(self, vocab_size, sequence_length, itw, wi):
        # Define the hyperparameters
        self.indices_to_words = itw
        self.word_indices = wi
        self.learning_rate = 0.3  # Should be about right
        self.training_epochs = 250  # How long to train for - chosen to fit within class time
        self.display_epoch_freq = 1  # How often to test and print out statistics
        self.dim = 32  # The dimension of the hidden state of the RNN
        self.embedding_dim = 16  # The dimension of the learned word embeddings
        self.batch_size = 256  # Somewhat arbitrary - can be tuned, but often tune for speed, not accuracy
        self.vocab_size = vocab_size  # Defined by the file reader above
        self.sequence_length = sequence_length  # Defined by the file reader above
        self.keep_rate = 0.75  # Used in dropout (at training time only, not at sampling time)
        
        #### Start main editable code block ####
        
        self.E = tf.Variable(tf.random_normal([self.vocab_size, self.embedding_dim], stddev = 0.1))

        self.W_f = tf.Variable(tf.random_normal([self.embedding_dim + self.dim, self.dim], stddev = 0.1))
        self.W_i = tf.Variable(tf.random_normal([self.embedding_dim + self.dim, self.dim], stddev = 0.1))
        self.W_o = tf.Variable(tf.random_normal([self.embedding_dim + self.dim, self.dim], stddev = 0.1))
        self.W_c = tf.Variable(tf.random_normal([self.embedding_dim + self.dim, self.dim], stddev = 0.1))

        self.b_f = tf.Variable(tf.random_normal([self.dim], stddev = 0.1))
        self.b_i = tf.Variable(tf.random_normal([self.dim], stddev = 0.1))
        self.b_o = tf.Variable(tf.random_normal([self.dim], stddev = 0.1))
        self.b_c = tf.Variable(tf.random_normal([self.dim], stddev = 0.1))

        self.W_y = tf.Variable(tf.random_normal([self.dim, self.vocab_size], stddev = 0.1))
        self.b_y = tf.Variable(tf.random_normal([self.vocab_size], stddev = 0.1))
        # Define the input placeholder(s).
        # I'll supply this one, since it's needed in sampling. Add any others you need.
        self.x = tf.placeholder(tf.int64, [None, self.sequence_length])

        # TODO: Build the rest of the LSTM LM!
        def step(x, h_prev, c_prev):
            emb    = tf.nn.embedding_lookup(self.E, x)
            embh   = tf.concat(1, [emb, h_prev]) # batch_size * embedding_dim + dim
            forget = tf.sigmoid(tf.matmul(embh, self.W_f) + self.b_f)
            info   = tf.sigmoid(tf.matmul(embh, self.W_i) + self.b_i)
            c_tilt = tf.tanh(tf.matmul(embh, self.W_c) + self.b_c)
            c_prev = tf.mul(forget, c_prev) + tf.mul(info, c_tilt)
            out    = tf.sigmoid(tf.matmul(embh, self.W_o) + self.b_o)
            h_prev = tf.mul(out, tf.tanh(c_prev))
            return h_prev, c_prev # batch_size * dim

        self.x_slices = tf.split(1, self.sequence_length, self.x)

        # Your model should populate the following four python lists.
        # self.logits should contain one [batch_size, vocab_size]-shaped TF tensor of logits 
        #   for each of the 20 steps of the model.
        # self.costs should contain one [batch_size]-shaped TF tensor of cross-entropy loss 
        #   values for each of the 20 steps of the model.
        # self.h and c should each start contain one [batch_size, dim]-shaped TF tensor of LSTM
        #   activations for each of the 21 *states* of the model -- one tensor of zeros for the 
        #   starting state followed by one tensor each for the remaining 20 steps.
        # Don't rename any of these variables or change their purpose -- they'll be needed by the
        # pre-built sampler.
        self.h_zero = tf.zeros([self.batch_size, self.dim])
        self.c_zero = tf.zeros([self.batch_size, self.dim])

        self.logits = []
        self.costs = []
        self.h = [self.h_zero]
        self.c = [self.c_zero]
        c_prev = self.c_zero
        h_prev = self.h_zero

        for i in range(self.sequence_length - 1):
            x = tf.reshape(self.x_slices[i], [-1]) # should be batch_size * 1
            h_prev, c_prev = step(x, h_prev, c_prev)
            y = tf.reshape(self.x_slices[i+1], [-1])
            logits = tf.matmul(h_prev, self.W_y) + self.b_y
            costs = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y)
            self.logits.append(logits)
            self.costs.append(costs)
            self.h.append(h_prev)
            self.c.append(c_prev)
        #### End main editable code block ####
        # Sum costs for each word in each example, but average cost across examples.
        self.costs_tensor = tf.concat(1, [tf.expand_dims(cost, 1) for cost in self.costs])
        self.cost_per_example = tf.reduce_sum(self.costs_tensor, 1)
        self.total_cost = tf.reduce_mean(self.cost_per_example)
            
        # This library call performs the main SGD update equation
        opt_obj = tf.train.GradientDescentOptimizer(self.learning_rate)
        gvs = opt_obj.compute_gradients(self.total_cost)
        capped_gvs = [(tf.clip_by_norm(grad, 5.0), var) \
                for grad, var in gvs if grad is not None]
        self.optimizer = opt_obj.apply_gradients(capped_gvs)
        
        
        # Create an operation to fill zero values in for W and b
        self.init = tf.initialize_all_variables()
        
        # Create a placeholder for the session that will be shared between training and evaluation
        self.sess = None
        
    def train(self, training_set):
        def get_minibatch(dataset, start_index, end_index):
            indices = range(start_index, end_index)
            vectors = np.vstack([dataset[i]['index_sequence'] for i in indices])
            return vectors
        
        self.sess = tf.Session()
        
        self.sess.run(self.init)
        print 'Training.'

        # Training cycle
        for epoch in range(self.training_epochs):
            random.shuffle(training_set)
            avg_cost = 0.
            total_batch = int(len(training_set) / self.batch_size)
            
            # Loop over all batches in epoch
            for i in range(total_batch):
                # Assemble a minibatch of the next B examples
                minibatch_vectors = get_minibatch(training_set, self.batch_size * i,\
                                    self.batch_size * (i + 1))

                # Run the optimizer to take a gradient step, and also fetch the value of the 
                # cost function for logging
                _, c = self.sess.run([self.optimizer, self.total_cost], 
                                     feed_dict={self.x: minibatch_vectors})
                                                                    
                # Compute average loss
                avg_cost += c / (total_batch * self.batch_size)
                
            # Display some statistics about the step
            if (epoch+1) % self.display_epoch_freq == 0:
                print "Epoch:", (epoch+1), "Cost:", avg_cost#, "Sample:", self.sample()
    
    def sample(self):
        # This samples a sequence of tokens from the model starting with <S>.
        # We only ever run the first timestep of the model, and use an effective batch size of one
        # but we leave the model unrolled for multiple steps, and use the full batch size to simplify 
        # the training code. This slows things down.

        def brittle_sampler():
            # The main sampling code. Can fail randomly due to rounding errors that yield probibilities
            # that don't sum to one.
            
            self.word_indices = [0] # 0 here is the "<S>" symbol
            for i in range(self.sequence_length - 1):
                dummy_x = np.zeros((self.batch_size, self.sequence_length))
                dummy_x[0][0] = self.word_indices[-1]
                feed_dict = {self.x: dummy_x}
                if i > 0:
                    feed_dict[self.h_zero] = h
                    feed_dict[self.c_zero] = c                
                h, c, logits = self.sess.run([self.h[1], self.c[1], self.logits[0]], 
                                             feed_dict=feed_dict)  
                logits = logits[0, :] # Discard all but first batch entry
                exp_logits = np.exp(logits - np.max(logits))
                distribution = exp_logits / exp_logits.sum()
                sampled_index = np.flatnonzero(np.random.multinomial(1, distribution))[0]
                self.word_indices.append(sampled_index)
            words = [self.indices_to_words[index] for index in self.word_indices]
            return ' '.join(words)
        
        while True:
            try:
                sample = brittle_sampler()
                return sample
            except ValueError as e:  # Retry if we experience a random failure.
                pass



def main():
    training_set = load_sst_data(sst_home + '/train.txt')
    dev_set = load_sst_data(sst_home + '/dev.txt')
    test_set = load_sst_data(sst_home + '/test.txt')

    indices_to_words, word_indices = sentence_to_padded_index_sequence([training_set, dev_set, test_set])
    model = LanguageModel(len(word_indices), 21, indices_to_words, word_indices)
    model.train(training_set)

    model.sample()

if __name__ == "__main__":
    main()
