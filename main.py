import os
import sys
import math

import tensorflow as tf

from tqdm import tqdm

from evaluator import MultiGPUEvaluator
from model import get_multi_gpu_models
from graph_handler import GraphHandler
from trainer import MultiGPUTrainer


from mytensorflow import get_num_params

from read_data import read_data, update_config

import numpy as np
from collections import Counter, defaultdict
from itertools import count
import random
from pprint import pprint


import slackweb
# slack web hook for notifying training progress log. 
slack = slackweb.Slack(url=os.environ['TRAINING_SLACK_NOTI_URL'])
header= "encoder_decoder_simple" # description for this training. 
 


flags = tf.app.flags


# Names and directories
# flags.DEFINE_string("data_dir", "../data/snli_simple", "Data dir [../data/snli_simple]")
flags.DEFINE_string("data_dir", "tiny", "Data dir [../data/snli_simple]") # sanity check with tiny dataset.
flags.DEFINE_string("model_name", "basic", "Model name [basic]")
flags.DEFINE_string("run_id", "0", "Run ID [0]")
flags.DEFINE_string("out_base_dir", "out", "out base dir [out]")
flags.DEFINE_string("forward_name", "single", "Forward name [single]")
flags.DEFINE_string("answer_path", "", "Answer path []")
flags.DEFINE_string("eval_path", "", "Eval path []")
flags.DEFINE_string("load_path", "", "Load path []")
flags.DEFINE_string("shared_path", "", "Shared path []")

# Device placement
flags.DEFINE_string("device", "/cpu:0", "default device for summing gradients. [/cpu:0]")
flags.DEFINE_string("device_type", "gpu", "device for computing gradients (parallelization). cpu | gpu [gpu]")
flags.DEFINE_integer("num_gpus", 1, "num of gpus or cpus for computing gradients [1]")


# Essential training and test options
flags.DEFINE_string("mode", "test", "trains | test | forward [test]")
flags.DEFINE_boolean("load", True, "load saved data? [True]")
flags.DEFINE_bool("single", False, "supervise only the answer sentence? [False]")
flags.DEFINE_boolean("debug", False, "Debugging mode? [False]")
flags.DEFINE_bool('load_ema', True, "load exponential average of variables when testing?  [True]")
flags.DEFINE_bool("eval", True, "eval? [True]")
flags.DEFINE_float("th", 0.5, "Threshold [0.5]")


# Training / test parameters
flags.DEFINE_integer("batch_size", 1, "Batch size [1000]")
flags.DEFINE_integer("val_num_batches", 0, "validation num batches [0]. "+ \
    "Use non-zero value to run evaluation on subset of the validation set.")
flags.DEFINE_integer("test_num_batches", 0, "test num batches [0]")
flags.DEFINE_integer("num_epochs", 12, "Total number of epochs for training [12]")
flags.DEFINE_integer("num_steps", 100000, "Number of steps [20000]")
flags.DEFINE_integer("load_step", 0, "load step [0]")
flags.DEFINE_float("init_lr", 0.001, "Initial learning rate [0.001]")
flags.DEFINE_float("input_keep_prob", 0.8, "Input keep prob for the dropout of LSTM weights [0.8]")
flags.DEFINE_float("keep_prob", 0.8, "Keep prob for the dropout of Char-CNN weights [0.8]")
flags.DEFINE_float("wd", 0.0, "L2 weight decay for regularization [0.0]")
flags.DEFINE_integer("hidden_size", 150, "Hidden size [100]")
flags.DEFINE_integer("char_out_size", 100, "char-level word embedding size [100]")
flags.DEFINE_integer("char_emb_size", 8, "Char emb size [8]")
flags.DEFINE_string("out_channel_dims", "100", "Out channel dims of Char-CNN, separated by commas [100]")
flags.DEFINE_string("filter_heights", "5", "Filter heights of Char-CNN, separated by commas [5]")
flags.DEFINE_bool("finetune", False, "Finetune word embeddings? [False]")
flags.DEFINE_bool("highway", False, "Use highway? [True]")
flags.DEFINE_integer("highway_num_layers", 2, "highway num layers [2]")
flags.DEFINE_bool("share_cnn_weights", True, "Share Char-CNN weights [True]")
flags.DEFINE_bool("share_lstm_weights", True, "Share pre-processing (phrase-level) LSTM weights [True]")
flags.DEFINE_float("var_decay", 0.999, "Exponential moving average decay for variables [0.999]")

# Optimizations
flags.DEFINE_bool("cluster", False, "Cluster data for faster training [False]")
flags.DEFINE_bool("len_opt", False, "Length optimization? [False]")
flags.DEFINE_bool("cpu_opt", False, "CPU optimization? GPU computation can be slower [False]")

# Logging and saving options
flags.DEFINE_boolean("progress", True, "Show progress? [True]")
flags.DEFINE_integer("log_period", 100, "Log period [100]")
flags.DEFINE_integer("eval_period", 100, "Eval period [1000]")
flags.DEFINE_integer("save_period", 1000, "Save Period [1000]")
flags.DEFINE_integer("max_to_keep", 20, "Max recent saves to keep [20]")
flags.DEFINE_bool("dump_eval", True, "dump eval? [True]")
flags.DEFINE_bool("dump_answer", True, "dump answer? [True]")
flags.DEFINE_bool("vis", False, "output visualization numbers? [False]")
flags.DEFINE_bool("dump_pickle", True, "Dump pickle instead of json? [True]")
flags.DEFINE_float("decay", 0.9, "Exponential moving average decay for logging values [0.9]")

# Thresholds for speed and less memory usage
flags.DEFINE_integer("word_count_th", 100, "word count th [100]")
flags.DEFINE_integer("char_count_th", 505, "char count th [500]")
flags.DEFINE_integer("sent_size_th", 400, "sent size th [64]")
flags.DEFINE_integer("word_size_th", 16, "word size th [16]") # what's difference word count and word size?

# Advanced training options
flags.DEFINE_bool("lower_word", True, "lower word [True]")
flags.DEFINE_bool("squash", False, "squash the sentences into one? [False]")
flags.DEFINE_bool("swap_memory", True, "swap memory? [True]")
flags.DEFINE_string("data_filter", "max", "max | valid | semi [max]")
flags.DEFINE_bool("use_glove_for_unk", True, "use glove for unk [False]")
flags.DEFINE_bool("known_if_glove", True, "consider as known if present in glove [False]")
flags.DEFINE_string("logit_func", "tri_linear", "logit func [tri_linear]")
flags.DEFINE_string("answer_func", "linear", "answer logit func [linear]")
flags.DEFINE_string("sh_logit_func", "tri_linear", "sh logit func [tri_linear]")

# Ablation options
flags.DEFINE_bool("use_char_emb", False, "use char emb? [True]")
flags.DEFINE_bool("use_word_emb", True, "use word embedding? [True]")
flags.DEFINE_bool("q2c_att", True, "question-to-context attention? [True]")
flags.DEFINE_bool("c2q_att", True, "context-to-question attention? [True]")
flags.DEFINE_bool("dynamic_att", False, "Dynamic attention [False]")


def _config_debug(config):
    if config.debug:
        config.num_steps = 2
        config.eval_period = 1
        config.log_period = 1
        config.save_period = 1
        config.val_num_batches = 2
        config.test_num_batches = 2

def set_dirs(config):
    import shutil
    # create directories
    assert config.load or config.mode == 'train', "config.load must be True if not training"
    if not config.load and os.path.exists(config.out_dir):
        shutil.rmtree(config.out_dir)

    config.save_dir = os.path.join(config.out_dir, "save")
    config.log_dir = os.path.join(config.out_dir, "log")
    config.eval_dir = os.path.join(config.out_dir, "eval")
    config.answer_dir = os.path.join(config.out_dir, "answer")
    if not os.path.exists(config.out_dir):
        os.makedirs(config.out_dir)
    if not os.path.exists(config.save_dir):
        os.mkdir(config.save_dir)
    if not os.path.exists(config.log_dir):
        os.mkdir(config.log_dir)
    if not os.path.exists(config.answer_dir):
        os.mkdir(config.answer_dir)
    if not os.path.exists(config.eval_dir):
        os.mkdir(config.eval_dir)



def main(_):
    config = flags.FLAGS
    config.out_dir = os.path.join(config.out_base_dir, config.model_name, str(config.run_id).zfill(2))

    set_dirs(config)
    with tf.device(config.device):
        if config.mode == 'train':
            _train(config)
        elif config.mode == 'test':
            _test(config)
        else:
            raise ValueError("invalid value for 'mode': {}".format(config.mode))


def _train(config):
    np.set_printoptions(threshold=np.inf)
    train_data = read_data(config, 'train', config.load)
    dev_data = read_data(config, 'dev', True)
    update_config(config, [train_data, dev_data])

    _config_debug(config)

    word2vec_dict = train_data.shared['lower_word2vec'] if config.lower_word else train_data.shared['word2vec']
    word2idx_dict = train_data.shared['word2idx']
    idx2vec_dict = {word2idx_dict[word]: vec for word, vec in word2vec_dict.items() if word in word2idx_dict}
    emb_mat = np.array([idx2vec_dict[idx] if idx in idx2vec_dict
                        else np.random.multivariate_normal(np.zeros(config.word_emb_size), np.eye(config.word_emb_size))
                        for idx in range(config.word_vocab_size)])
    config.emb_mat = emb_mat


    def make_idx2word():
        """
        return index of the word from the preprocessed dictionary. 
        """
        idx2word = {}
        d = train_data.shared['word2idx']
        for word, idx in d.items():
            print(word)
            idx2word[idx] = word
        if config.use_glove_for_unk:
            d2 = train_data.shared['new_word2idx']
            for word, idx in d2.items():
                print(word)
                idx2word[idx+len(d)] = word
        return idx2word


    idx2word = make_idx2word()
    # Save total number of words used in this dictionary: words in GloVe + etc tokens(including UNK, POS, ... etc)
    print("size of config.id2word len:", len(idx2word))
    print("size of config.total_word_vocab_size:", config.total_word_vocab_size)


    # construct model graph and variables (using default graph)
    pprint(config.__flags, indent=2)
    models = get_multi_gpu_models(config)
    model = models[0]
    print("num params: {}".format(get_num_params()))
    trainer = MultiGPUTrainer(config, models)
    evaluator = MultiGPUEvaluator(config, models, tensor_dict=model.tensor_dict if config.vis else None)
    graph_handler = GraphHandler(config, model)  # controls all tensors and variables in the graph, including loading /saving

    # Variables
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    graph_handler.initialize(sess)

    num_steps = config.num_steps or int(math.ceil(train_data.num_examples / (config.batch_size * config.num_gpus))) * config.num_epochs
    min_val = {}
    min_val['loss'] = 100.0
    min_val['acc'] = 0
    min_val['step'] = 0
    min_val['patience'] = 0


    for batches in tqdm(train_data.get_multi_batches(config.batch_size, config.num_gpus,
                                                     num_steps=num_steps, shuffle=True, cluster=config.cluster), total=num_steps):
        global_step = sess.run(model.global_step) + 1  # +1 because all calculations are done after step
        get_summary = global_step % config.log_period == 0
        loss, summary, train_op = trainer.step(sess, batches, get_summary=get_summary)        
        if get_summary:
            graph_handler.add_summary(summary, global_step)

        # occasional saving
        if global_step % config.save_period == 0:
            graph_handler.save(sess, global_step=global_step)

        if not config.eval:
            continue
        # Occasional evaluation
        if global_step % config.eval_period == 0:
            num_steps = math.ceil(dev_data.num_examples / (config.batch_size * config.num_gpus))
            
            # num_steps: total steps to finish this training session. 
            # val_num_batches: 100
            if 0 < config.val_num_batches < num_steps:
                # if config.val_num_batches is less the the actual steps required to run whole dev set. Run evaluation up to the step.  
                num_steps = config.val_num_batches

            
            # This train loss is calulated from sampling the same number of data size of dev_data. 

            e_train = evaluator.get_evaluation_from_batches(
                sess, tqdm(train_data.get_multi_batches(config.batch_size, config.num_gpus, num_steps=num_steps), total=num_steps)
            )
            graph_handler.add_summaries(e_train.summaries, global_step)

            # This e_dev may differ from the dev_set used in test time because some data is filtered out here. 
            e_dev = evaluator.get_evaluation_from_batches(
                sess, tqdm(dev_data.get_multi_batches(config.batch_size, config.num_gpus, num_steps=num_steps), total=num_steps))
            graph_handler.add_summaries(e_dev.summaries, global_step)
            print("%s e_train: loss=%.4f" % (header, e_train.loss))
            print("%s e_dev: loss=%.4f" % (header, e_dev.loss))
            print()
            if min_val['loss'] > e_dev.loss:
                min_val['loss'] = e_dev.loss
                min_val['step'] = global_step
                min_val['patience'] = 0
            else:
                min_val['patience'] = min_val['patience'] + 1
                if min_val['patience'] >= 1000:
                    slack.notify(text="%s patience reached %d. early stopping." % (header, min_val['patience']))
                    break

            slack.notify(text="%s e_dev: loss=%.4f" % (header, e_dev.loss))

            if config.dump_eval:
                graph_handler.dump_eval(e_dev)
            if config.dump_answer:
                graph_handler.dump_answer(e_dev)
        
        
    
    slack.notify(text="%s <@U024BE7LH|insikk> Train is finished. e_dev: loss=%.4f at step=%d\nPlease assign another task to get more research result" % (header, min_val['loss'], min_val['step']))    
     
    if global_step % config.save_period != 0:
        graph_handler.save(sess, global_step=global_step)



def _test(config):
    test_data = read_data(config, 'test', True)
    update_config(config, [test_data])

    _config_debug(config)

    if config.use_glove_for_unk:
        word2vec_dict = test_data.shared['lower_word2vec'] if config.lower_word else test_data.shared['word2vec']
        new_word2idx_dict = test_data.shared['new_word2idx']
        idx2vec_dict = {idx: word2vec_dict[word] for word, idx in new_word2idx_dict.items()}
        new_emb_mat = np.array([idx2vec_dict[idx] for idx in range(len(idx2vec_dict))], dtype='float32')
        config.new_emb_mat = new_emb_mat

    pprint(config.__flags, indent=2)
    models = get_multi_gpu_models(config)
    model = models[0]
    evaluator = MultiGPUEvaluator(config, models, tensor_dict=models[0].tensor_dict if config.vis else None)
    graph_handler = GraphHandler(config, model)

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    graph_handler.initialize(sess)
    num_steps = math.ceil(test_data.num_examples / (config.batch_size * config.num_gpus))
    if 0 < config.test_num_batches < num_steps:
        num_steps = config.test_num_batches
    e = None
    for multi_batch in tqdm(test_data.get_multi_batches(config.batch_size, config.num_gpus, num_steps=num_steps, cluster=config.cluster), total=num_steps):
        ei = evaluator.get_evaluation(sess, multi_batch)
        e = ei if e is None else e + ei
        if config.vis:
            eval_subdir = os.path.join(config.eval_dir, "{}-{}".format(ei.data_type, str(ei.global_step).zfill(6)))
            if not os.path.exists(eval_subdir):
                os.mkdir(eval_subdir)
            path = os.path.join(eval_subdir, str(ei.idxs[0]).zfill(8))
            graph_handler.dump_eval(ei, path=path)

    print("test acc: %f, loss: %f" % (e.acc, e.loss))
    if config.dump_answer:
        print("dumping answer ...")
        graph_handler.dump_answer(e)
    if config.dump_eval:
        print("dumping eval ...")
        graph_handler.dump_eval(e)
        

    
if __name__ == "__main__":
    tf.app.run()
    
