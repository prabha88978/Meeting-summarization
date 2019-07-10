import getpass
import nltk
from nltk.tokenize import word_tokenize
import re
import collections
import pickle
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from gensim.test.utils import get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec

train_article_path = "train.article.txt" 

#path for training text output (headline)
train_title_path   ="train.title.txt"

#path for validation text (article)
valid_article_path = "valid.article.filter.txt"

#path for validation text output(headline)
valid_title_path   = "valid.title.filter.txt"

input_size_ = 200000

def clean_str(sentence):
    sentence = re.sub("[#]+", "#", sentence)
    return sentence

def get_text_list(data_path, toy):
    with open (data_path, "r", encoding="utf-8") as f:
        if not toy:
            return [clean_str(x.strip()) for x in f.readlines()][:input_size_]
        else:
            return [clean_str(x.strip()) for x in f.readlines()][:1000]


def build_dict(step, toy=False):
    if step == "train":
        train_article_list = get_text_list(train_article_path, toy)
        train_title_list = get_text_list(train_title_path, toy)

        '''words = list()
        for sentence in train_article_list + train_title_list:
            for word in word_tokenize(sentence):
                words.append(word)

        word_counter = collections.Counter(words).most_common()
        #print(word_counter)
        # ('#', 363119), ('the', 307009), (',', 204614), ('to', 202066) ouuput 
        word_dict = dict()
        word_dict["<padding>"] = 0
        word_dict["<unk>"] = 1
        word_dict["<s>"] = 2
        word_dict["</s>"] = 3
        # we are creating the word to int dictionary 
        for word, _ in word_counter:
            word_dict[word] = len(word_dict) 

        with open("word_dict.pickle", "wb") as f:
            pickle.dump(word_dict, f)'''
        with open("word_dict.pickle", "rb") as f:
            word_dict = pickle.load(f)

    elif step == "valid":
        with open("word_dict.pickle", "rb") as f:
            word_dict = pickle.load(f)
     

    reversed_dict = dict(zip(word_dict.values(), word_dict.keys()))

    '''article_max_len = max([len(word_tokenize(d)) for d in train_article_list])
    print(article_max_len)
    summary_max_len = max([len(word_tokenize(d)) for d in train_title_list])
    print(summary_max_len)'''
    article_max_len = 150
    summary_max_len = 50

    return word_dict, reversed_dict, article_max_len, summary_max_len


def build_dataset(step,article_list, title_list, word_dict, article_max_len, summary_max_len, toy=False):
    if step == "train":
        article_list = [clean_str(x.strip()) for x in article_list]
        title_list   = [clean_str(x.strip()) for x in title_list]
    elif step == "valid":
        article_list = [clean_str(x.strip()) for x in article_list]
    else:
        raise NotImplementedError
    x = [word_tokenize(d) for d in article_list]
    x = [[word_dict.get(w, word_dict["<unk>"]) for w in d] for d in x]
    x = [d[:article_max_len] for d in x]
    x = [d + (article_max_len - len(d)) * [word_dict["<padding>"]] for d in x]
    
    if step == "valid":
        return x
    else:        
        y = [word_tokenize(d) for d in title_list]
        y = [[word_dict.get(w, word_dict["<unk>"]) for w in d] for d in y]
        y = [d[:(summary_max_len - 1)] for d in y]
        return x, y

def get_init_embedding(reversed_dict, embedding_size):
    print("Loading Glove vectors...")
    with open("glove/model_glove_300.pkl", 'rb') as handle:
        word_vectors = pickle.load(handle)
        
    word_vec_list = list()
    for _, word in sorted(reversed_dict.items()):
        try:
            word_vec = word_vectors.word_vec(word)
        except KeyError:
            word_vec = np.zeros([embedding_size], dtype=np.float32)

        word_vec_list.append(word_vec)

    # Assign random vector to <s>, </s> token
    word_vec_list[2] = np.random.normal(0, 1, embedding_size)
    word_vec_list[3] = np.random.normal(0, 1, embedding_size)

    return np.array(word_vec_list)

import tensorflow as tf
from tensorflow.contrib import rnn

class Model(object):
    def __init__(self, reversed_dict, article_max_len, summary_max_len, args, forward_only=False):
        self.vocabulary_size = len(reversed_dict)
        self.embedding_size = args.embedding_size
        self.num_hidden = args.num_hidden
        self.num_layers = args.num_layers
        self.learning_rate = args.learning_rate
        self.beam_width = args.beam_width
        
        if not forward_only:
            self.keep_prob = args.keep_prob
        else:
            self.keep_prob = 1.0
        self.cell = tf.nn.rnn_cell.BasicLSTMCell
        with tf.variable_scope("decoder/projection"):
            self.projection_layer = tf.layers.Dense(self.vocabulary_size, use_bias=False)

        self.batch_size = tf.placeholder(tf.int32, (), name="batch_size")
        self.X = tf.placeholder(tf.int32, [None, article_max_len])
        self.X_len = tf.placeholder(tf.int32, [None])
        self.decoder_input = tf.placeholder(tf.int32, [None, summary_max_len])
        self.decoder_len = tf.placeholder(tf.int32, [None])
        self.decoder_target = tf.placeholder(tf.int32, [None, summary_max_len])
        self.global_step = tf.Variable(0, trainable=False)
        with tf.name_scope("embedding"):
            if not forward_only and args.glove:
                init_embeddings = tf.constant(get_init_embedding(reversed_dict, self.embedding_size), dtype=tf.float32)
            else:
                init_embeddings = tf.random_uniform([self.vocabulary_size, self.embedding_size], -1.0, 1.0)
            self.embeddings = tf.get_variable("embeddings", initializer=init_embeddings)
            self.encoder_emb_inp = tf.transpose(tf.nn.embedding_lookup(self.embeddings, self.X), perm=[1, 0, 2])
            self.decoder_emb_inp = tf.transpose(tf.nn.embedding_lookup(self.embeddings, self.decoder_input), perm=[1, 0, 2])

        with tf.name_scope("encoder"):
            fw_cells = [self.cell(self.num_hidden) for _ in range(self.num_layers)]
            bw_cells = [self.cell(self.num_hidden) for _ in range(self.num_layers)]
            fw_cells = [rnn.DropoutWrapper(cell) for cell in fw_cells]
            bw_cells = [rnn.DropoutWrapper(cell) for cell in bw_cells]

            encoder_outputs, encoder_state_fw, encoder_state_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                fw_cells, bw_cells, self.encoder_emb_inp,
                sequence_length=self.X_len, time_major=True, dtype=tf.float32)
            self.encoder_output = tf.concat(encoder_outputs, 2)
            encoder_state_c = tf.concat((encoder_state_fw[0].c, encoder_state_bw[0].c), 1)
            encoder_state_h = tf.concat((encoder_state_fw[0].h, encoder_state_bw[0].h), 1)
            self.encoder_state = rnn.LSTMStateTuple(c=encoder_state_c, h=encoder_state_h)

        with tf.name_scope("decoder"), tf.variable_scope("decoder") as decoder_scope:
            decoder_cell = self.cell(self.num_hidden * 2)

            if not forward_only:
                attention_states = tf.transpose(self.encoder_output, [1, 0, 2])
                attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                    self.num_hidden * 2, attention_states, memory_sequence_length=self.X_len, normalize=True)
                decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism,
                                                                   attention_layer_size=self.num_hidden * 2)
                initial_state = decoder_cell.zero_state(dtype=tf.float32, batch_size=self.batch_size)
                initial_state = initial_state.clone(cell_state=self.encoder_state)
                helper = tf.contrib.seq2seq.TrainingHelper(self.decoder_emb_inp, self.decoder_len, time_major=True)
                decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, initial_state)
                outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, output_time_major=True, scope=decoder_scope)
                self.decoder_output = outputs.rnn_output
                self.logits = tf.transpose(
                    self.projection_layer(self.decoder_output), perm=[1, 0, 2])
                self.logits_reshape = tf.concat(
                    [self.logits, tf.zeros([self.batch_size, summary_max_len - tf.shape(self.logits)[1], self.vocabulary_size])], axis=1)
            else:
                tiled_encoder_output = tf.contrib.seq2seq.tile_batch(
                    tf.transpose(self.encoder_output, perm=[1, 0, 2]), multiplier=self.beam_width)
                tiled_encoder_final_state = tf.contrib.seq2seq.tile_batch(self.encoder_state, multiplier=self.beam_width)
                tiled_seq_len = tf.contrib.seq2seq.tile_batch(self.X_len, multiplier=self.beam_width)
                attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                    self.num_hidden * 2, tiled_encoder_output, memory_sequence_length=tiled_seq_len, normalize=True)
                decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism,
                                                                   attention_layer_size=self.num_hidden * 2)
                initial_state = decoder_cell.zero_state(dtype=tf.float32, batch_size=self.batch_size * self.beam_width)
                initial_state = initial_state.clone(cell_state=tiled_encoder_final_state)
                decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                    cell=decoder_cell,
                    embedding=self.embeddings,
                    start_tokens=tf.fill([self.batch_size], tf.constant(2)),
                    end_token=tf.constant(3),
                    initial_state=initial_state,
                    beam_width=self.beam_width,
                    output_layer=self.projection_layer
                )
                outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
                    decoder, output_time_major=True, maximum_iterations=summary_max_len, scope=decoder_scope)
                self.prediction = tf.transpose(outputs.predicted_ids, perm=[1, 2, 0])

        with tf.name_scope("loss"):
            if not forward_only:
                crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.logits_reshape, labels=self.decoder_target)
                weights = tf.sequence_mask(self.decoder_len, summary_max_len, dtype=tf.float32)
                self.loss = tf.reduce_sum(crossent * weights / tf.to_float(self.batch_size))

                params = tf.trainable_variables()
                gradients = tf.gradients(self.loss, params)
                clipped_gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
                optimizer = tf.train.AdamOptimizer(self.learning_rate)
                self.update = optimizer.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)

def get_text_list_test(data_path, toy,input_size_):
    with open (data_path, "r", encoding="utf-8") as f:
        if not toy:
            return [clean_str(x.strip()) for x in f.readlines()][:input_size_]
        else:
            return [clean_str(x.strip()) for x in f.readlines()][:1000]


def testing_part(article_text):


    import tensorflow as tf
    import pickle

    #path for validation text (article)
    valid_article_path = "valid.article.filter.txt"

    #path for validation text output(headline)
    valid_title_path   = "valid.title.filter.txt"

    tf.reset_default_graph()

    class args:
        pass
      
    args.num_hidden=150
    args.num_layers=2
    args.beam_width=10
    args.glove="store_true"
    args.embedding_size = 300

    args.learning_rate=1e-3
    args.batch_size= 1
    args.num_epochs= 1
    args.keep_prob = 0.8

    args.toy = False

    args.with_model="store_true"
    args.input_size_ = 1


    print("Loading dictionary...")
    print("Loading dictionary...")
    word_dict, reversed_dict, article_max_len, summary_max_len = build_dict("valid", args.toy)
    print("Loading validation dataset...")
    #valid_x = build_dataset("valid", word_dict, article_max_len, summary_max_len, args.toy)
    #valid_x_len = [len([y for y in x if y != 0]) for x in valid_x]
    print("Loading article and reference...")
    article = get_text_list_test(valid_article_path, args.toy, args.input_size_)
    reference = get_text_list_test(valid_title_path, args.toy, args.input_size_)
    summary_array = []
    summari = ''
    with tf.Session() as sess:
        print("Loading saved model...")
        model = Model(reversed_dict, article_max_len, summary_max_len, args, forward_only=True)
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state("saved_model/")
        saver.restore(sess, ckpt.model_checkpoint_path)

        #batches = batch_iter(valid_x, [0] * len(valid_x), args.batch_size, 1)

        print("Writing summaries to 'result.txt'...")
        for inde in range(args.num_epochs):
            with open("valid.article.filter.txt", "r", encoding="utf-8") as f1:
                print("Hello")
                text_ = list()
                text_.append(article_text)
                inputs_ = list()
                outputs_ = list()
                for index, inputs in enumerate(text_):
                    if((index+1) % args.batch_size == 0 ):
                        #print(inputs_)
                        inputs_.append(inputs)
                        print(inputs_)
                        #outputs_.append(outputs)
                        inputss = build_dataset("valid", inputs_, outputs_, word_dict, article_max_len, summary_max_len)
                        print("here it is working better")
                        inputss = np.array(inputss)
                        batch_x_len = list(map(lambda x: len([y for y in x if y != 0]), inputss))
                        valid_feed_dict = {
                            model.batch_size: len(inputss),
                            model.X: inputss,
                            model.X_len: batch_x_len,
                        }
                        prediction = sess.run(model.prediction, feed_dict=valid_feed_dict)
                        prediction_output = [[reversed_dict[y] for y in x] for x in prediction[:, 0, :]]
                        with open("result.txt", "w+") as f:
                            for line in prediction_output:
                                summary = list()
                                
                                for word in line:
                                    if word == "</s>":
                                        break
                                    if word not in summary:
                                        summary.append(word)
                                        summari = summari + word + ' '
                                summary_array.append(" ".join(summari))
                                #print(" ".join(summary), file=f)
                                f.write(summari + "\n")
                        inputs_, outputs_ = [],[]
                    else:
                        inputs_.append(inputs)
                        #outputs_.append(outputs)
                    if index == 0:
                        break
        print('Summaries have been generated')
    print(summari)
    return summari

