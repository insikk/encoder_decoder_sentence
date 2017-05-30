import numpy as np
import tensorflow as tf

from read_data import DataSet
from mytensorflow import padded_reshape


def argmax(x):
    return np.unravel_index(x.argmax(), x.shape)

class Evaluation(object):
    def __init__(self, data_type, global_step, idxs, yp, tensor_dict=None):
        self.data_type = data_type
        self.global_step = global_step
        self.idxs = idxs
        self.yp = yp
        self.num_examples = len(yp)
        self.tensor_dict = None
        self.dict = {'data_type': data_type,
                     'global_step': global_step,
                     'yp': yp,
                     'idxs': idxs,
                     'num_examples': self.num_examples}
        if tensor_dict is not None:
            self.tensor_dict = {key: val.tolist() for key, val in tensor_dict.items()}
            for key, val in self.tensor_dict.items():
                self.dict[key] = val
        self.summaries = None

    def __repr__(self):
        return "{} step {}".format(self.data_type, self.global_step)

    def __add__(self, other):
        if other == 0:
            return self
        assert self.data_type == other.data_type
        assert self.global_step == other.global_step
        new_yp = self.yp + other.yp
        new_idxs = self.idxs + other.idxs
        new_tensor_dict = None
        if self.tensor_dict is not None:
            new_tensor_dict = {key: val + other.tensor_dict[key] for key, val in self.tensor_dict.items()}
        return Evaluation(self.data_type, self.global_step, new_idxs, new_yp, tensor_dict=new_tensor_dict)

    def __radd__(self, other):
        return self.__add__(other)


class LabeledEvaluation(Evaluation):
    def __init__(self, data_type, global_step, idxs, yp, y, tensor_dict=None):
        super(LabeledEvaluation, self).__init__(data_type, global_step, idxs, yp, tensor_dict=tensor_dict)
        self.y = y
        self.dict['y'] = y

    def __add__(self, other):
        if other == 0:
            return self
        assert self.data_type == other.data_type
        assert self.global_step == other.global_step
        new_yp = self.yp + other.yp
        new_y = self.y + other.y
        new_idxs = self.idxs + other.idxs
        if self.tensor_dict is not None:
            new_tensor_dict = {key: np.concatenate((val, other.tensor_dict[key]), axis=0) for key, val in self.tensor_dict.items()}
        return LabeledEvaluation(self.data_type, self.global_step, new_idxs, new_yp, new_y, tensor_dict=new_tensor_dict)


class AccuracyEvaluation(LabeledEvaluation):
    def __init__(self, data_type, global_step, idxs, yp, y, correct, loss, tensor_dict=None):
        super(AccuracyEvaluation, self).__init__(data_type, global_step, idxs, yp, y, tensor_dict=tensor_dict)
        self.loss = loss
        self.correct = correct
        self.acc = sum(correct) / len(correct)
        self.dict['loss'] = loss
        self.dict['correct'] = correct
        self.dict['acc'] = self.acc
        loss_summary = tf.Summary(value=[tf.Summary.Value(tag='{}/loss'.format(data_type), simple_value=self.loss)])
        acc_summary = tf.Summary(value=[tf.Summary.Value(tag='{}/acc'.format(data_type), simple_value=self.acc)])
        self.summaries = [loss_summary, acc_summary]

    def __repr__(self):
        return "{} step {}: accuracy={}, loss={}".format(self.data_type, self.global_step, self.acc, self.loss)

    def __add__(self, other):
        if other == 0:
            return self
        assert self.data_type == other.data_type
        assert self.global_step == other.global_step
        new_idxs = self.idxs + other.idxs
        new_yp = self.yp + other.yp
        new_y = self.y + other.y
        new_correct = self.correct + other.correct
        new_loss = (self.loss * self.num_examples + other.loss * other.num_examples) / len(new_correct)
        if self.tensor_dict is not None:
            new_tensor_dict = {key: np.concatenate((val, other.tensor_dict[key]), axis=0) for key, val in self.tensor_dict.items()}
        return AccuracyEvaluation(self.data_type, self.global_step, new_idxs, new_yp, new_y, new_correct, new_loss, tensor_dict=new_tensor_dict)


class Evaluator(object):
    def __init__(self, config, model, tensor_dict=None):
        self.config = config
        self.model = model
        self.global_step = model.global_step
        self.logits = model.logits
        self.tensor_dict = {} if tensor_dict is None else tensor_dict

    def get_evaluation(self, sess, batch):
        idxs, data_set = batch
        feed_dict = self.model.get_feed_dict(data_set, False, supervised=False)
        global_step, logits, vals = sess.run([self.global_step, self.logits, list(self.tensor_dict.values())], feed_dict=feed_dict)
        logits = logits[:data_set.num_examples]
        tensor_dict = dict(zip(self.tensor_dict.keys(), vals))
        e = Evaluation(data_set.data_type, int(global_step), idxs, logits.tolist(), tensor_dict=tensor_dict)
        return e

    def get_evaluation_from_batches(self, sess, batches):
        e = sum(self.get_evaluation(sess, batch) for batch in batches)
        return e


class LabeledEvaluator(Evaluator):
    def __init__(self, config, model, tensor_dict=None):
        super(LabeledEvaluator, self).__init__(config, model, tensor_dict=tensor_dict)
        self.z = model.z

    def get_evaluation(self, sess, batch):
        idxs, data_set = batch
        feed_dict = self.model.get_feed_dict(data_set, False, supervised=False)
        global_step, logits, vals = sess.run([self.global_step, self.logits, list(self.tensor_dict.values())], feed_dict=feed_dict)
        logits = logits[:data_set.num_examples]
        z = feed_dict[self.z]
        tensor_dict = dict(zip(self.tensor_dict.keys(), vals))
        e = LabeledEvaluation(data_set.data_type, int(global_step), idxs, logits.tolist(), z.tolist(), tensor_dict=tensor_dict)
        return e


class AccuracyEvaluator(LabeledEvaluator):
    def __init__(self, config, model, tensor_dict=None):
        super(AccuracyEvaluator, self).__init__(config, model, tensor_dict=tensor_dict)
        self.loss = model.loss

    def get_evaluation(self, sess, batch):        
        idxs, data_set = self._split_batch(batch)
        assert isinstance(data_set, DataSet)
        feed_dict = self.model.get_feed_dict(data_set, False)
        global_step, logits, loss, vals = sess.run([self.global_step, self.logits, self.loss, list(self.tensor_dict.values())], feed_dict=feed_dict)
        z = data_set.data['z_list']
        logits = logits[:data_set.num_examples]
        correct = [self.__class__.compare(yi, ypi) for yi, ypi in zip(z, logits)]
        tensor_dict = dict(zip(self.tensor_dict.keys(), vals))
        e = AccuracyEvaluation(data_set.data_type, int(global_step), idxs, logits.tolist(), z, correct, float(loss), tensor_dict=tensor_dict)
        return e
    
    def _split_batch(self, batch):
        return batch

    def _get_feed_dict(self, batch):
        return self.model.get_feed_dict(batch[1], False)

    @staticmethod
    def compare(yi, ypi):
        if int(np.argmax(yi)) == int(np.argmax(ypi)):
            return True
        return False


class MultiGPUEvaluator(AccuracyEvaluator):
    def __init__(self, config, models, tensor_dict=None):
        super(MultiGPUEvaluator, self).__init__(config, models[0], tensor_dict=tensor_dict)
        self.models = models
        with tf.name_scope("eval_concat"):
            self.logits = tf.concat(axis=0, values=[model.logits for model in models])
            self.loss = tf.add_n([model.loss for model in models])/len(models)

    def _split_batch(self, batches):
        idxs_list, data_sets = zip(*batches)
        idxs = sum(idxs_list, ())
        data_set = sum(data_sets, data_sets[0].get_empty())
        return idxs, data_set

    def _get_feed_dict(self, batches):
        feed_dict = {}
        for model, (_, data_set) in zip(self.models, batches):
            feed_dict.update(model.get_feed_dict(data_set, False))
        return feed_dict
