import os
import sys
import json
import itertools
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
import jieba.posseg as pseg
import jieba
import nrekit
import ipdb


# Server setting
##########################################################################################################################
app = Flask(__name__)


# Pre-processing
##########################################################################################################################
D_REL = dict()
relation_path = "../datasets/DuNRE/relation_nre.json"
with open(relation_path) as f:
    d = json.load(f)
    for k in d:
        D_REL[d[k]] = k

def get_pair(lst_text, w_now, w_tag=True):
    # input: lst_text = [(word, word_pos), ...]
    #     w_now is a string
    #     w_tag is a tag to distinct head or tail (True for head, False for tail)
    # output: lst_pair = [(head, tail), ...]
    assert isinstance(w_now, str)
    lst_pair = list()
    if len(w_now) > 0:
        for w, wp in lst_text:
            if w == w_now:  # entity should not match itself
                continue
            if (('n' in wp) or (wp in ['i', 'j', 's', 'l'])):  # select eligible w according to its part-of-speech
                if w_tag:
                    lst_pair.append((w_now, w))
                else:
                    lst_pair.append((w, w_now))
    elif len(w_now) == 0:
        lst_entity = [w for w, wp in lst_text if ('n' in wp) or (wp in ['i', 'j', 's', 'l'])]
        lst_entity = list(set(lst_entity))
        for head, tail in itertools.combinations(lst_entity, 2):
            lst_pair.append((head, tail))
            lst_pair.append((tail, head))    
    else:
        pass
    lst_pair = list(set(lst_pair))
    return lst_pair


def string2json(text, head, tail, save_path=None):
    # input: text, head, tail, save_path are both string
    # output: sample is a json， save sample to a json file and return True
    #      sample = [d_1, d_2, ...], where d_i = {sentence: str, head:{word:str, id: str}, relation:empty str}
    assert isinstance(save_path, str)
    assert isinstance(head, str)
    assert isinstance(tail, str)
    assert isinstance(text, str)
    # generate sentence
    lst_text = pseg.lcut(text)
    sentence = [i for i, j in lst_text]
    sentence = " ".join(sentence)
    # generate entity-pairs
    if len(head) > 0 and len(tail) > 0:
        lst_pair = [(head, tail)]
    elif len(head) > 0 and len(tail) == 0:
        lst_pair = get_pair(lst_text, head, True)
    elif len(head) == 0 and len(tail) > 0:
        lst_pair = get_pair(lst_text, tail, False)
    else:
        lst_pair = get_pair(lst_text, "")
    # generate sample with json structure
    sample = list()
    for head, tail in lst_pair:
        d = {
            "sentence": sentence,
            "head": {"word": str(head), "id": str(head)},
            "tail": {"word": str(tail), "id": str(tail)},
            "relation": ""}
        sample.append(d)
    # save sample
    with open(save_path, "w") as f:
        json.dump(sample, f)
    return True


# Load train and predicted data
##########################################################################################################################
def load_data(sample_path, dataset_name=None, dataset_dir="../datasets/"):
    assert dataset_name is not None
    dataset_dir = dataset_dir + dataset_name
    train_loader = nrekit.data_loader.json_file_data_loader(os.path.join(dataset_dir, 'train_nre.json'), 
                                                        os.path.join(dataset_dir, 'word_dictionary_nre.json'),
                                                        os.path.join(dataset_dir, 'relation_nre.json'),  
                                                        mode=nrekit.data_loader.json_file_data_loader.MODE_RELFACT_BAG,
                                                        shuffle=True)
    test_loader = nrekit.data_loader.json_file_data_loader(sample_path, 
                                         os.path.join(dataset_dir, 'word_dictionary_nre.json'),
                                         os.path.join(dataset_dir, 'relation_nre.json'), 
                                         mode=nrekit.data_loader.json_file_data_loader.MODE_ENTPAIR_BAG,
                                         shuffle=False)
    return train_loader, test_loader


# Define model
##########################################################################################################################
class model(nrekit.framework.re_model):
    encoder = "pcnn"
    selector = "att"

    def __init__(self, train_data_loader, batch_size, max_length=120):
        nrekit.framework.re_model.__init__(self, train_data_loader, batch_size, max_length=max_length)
        self.mask = tf.placeholder(dtype=tf.int32, shape=[None, max_length], name="mask")
        
        # Embedding
        x = nrekit.network.embedding.word_position_embedding(self.word, self.word_vec_mat, self.pos1, self.pos2)

        # Encoder
        if model.encoder == "pcnn":
            x_train = nrekit.network.encoder.pcnn(x, self.mask, keep_prob=0.5)
            x_test = nrekit.network.encoder.pcnn(x, self.mask, keep_prob=1.0)
        elif model.encoder == "cnn":
            x_train = nrekit.network.encoder.cnn(x, keep_prob=0.5)
            x_test = nrekit.network.encoder.cnn(x, keep_prob=1.0)
        elif model.encoder == "rnn":
            x_train = nrekit.network.encoder.rnn(x, self.length, keep_prob=0.5)
            x_test = nrekit.network.encoder.rnn(x, self.length, keep_prob=1.0)
        elif model.encoder == "birnn":
            x_train = nrekit.network.encoder.birnn(x, self.length, keep_prob=0.5)
            x_test = nrekit.network.encoder.birnn(x, self.length, keep_prob=1.0)
        else:
            raise NotImplementedError

        # Selector
        if model.selector == "att":
            self._train_logit, train_repre = nrekit.network.selector.bag_attention(x_train, self.scope, self.ins_label, self.rel_tot, True, keep_prob=0.5)
            self._test_logit, test_repre = nrekit.network.selector.bag_attention(x_test, self.scope, self.ins_label, self.rel_tot, False, keep_prob=1.0)
        elif model.selector == "ave":
            self._train_logit, train_repre = nrekit.network.selector.bag_average(x_train, self.scope, self.rel_tot, keep_prob=0.5)
            self._test_logit, test_repre = nrekit.network.selector.bag_average(x_test, self.scope, self.rel_tot, keep_prob=1.0)
            self._test_logit = tf.nn.softmax(self._test_logit)
        elif model.selector == "max":
            self._train_logit, train_repre = nrekit.network.selector.bag_maximum(x_train, self.scope, self.ins_label, self.rel_tot, True, keep_prob=0.5)
            self._test_logit, test_repre = nrekit.network.selector.bag_maximum(x_test, self.scope, self.ins_label, self.rel_tot, False, keep_prob=1.0)
            self._test_logit = tf.nn.softmax(self._test_logit)
        else:
            raise NotImplementedError
        
        # Classifier
        self._loss = nrekit.network.classifier.softmax_cross_entropy(self._train_logit, self.label, self.rel_tot, weights_table=self.get_weights())
 
    def loss(self):
        return self._loss

    def train_logit(self):
        return self._train_logit

    def test_logit(self):
        return self._test_logit

    def get_weights(self):
        with tf.variable_scope("weights_table", reuse=tf.AUTO_REUSE):
            print("Calculating weights_table...")
            _weights_table = np.zeros((self.rel_tot), dtype=np.float32)
            for i in range(len(self.train_data_loader.data_rel)):
                _weights_table[self.train_data_loader.data_rel[i]] += 1.0 
            _weights_table = 1 / (_weights_table ** 0.05)
            weights_table = tf.get_variable(name='weights_table', dtype=tf.float32, trainable=False, initializer=_weights_table)
            print("Finish calculating")
        return weights_table

   
# Post-processing
##########################################################################################################################
def post_ranking(lst_res, threshold):
    # input: lst_res = [d_1, d_2, ...], where d_i = {"score": float, "entpair": string(byte), "relation": int}
    #     threshold, a float in (0, 1] to decide whether the current entpair is accepted
    # output: lst_pre = [d_1, d_2, ...], where d_i = {"head": str, "tail": str, "relation", "score": float}
    # rank results according to threshold
    d_pre = dict()
    for res in lst_res:
        entpair_now = res["entpair"]
        relation_now = res["relation"]
        score_now = res["score"]
        if entpair_now in d_pre:
            if score_now > d_pre[entpair_now]["score"]:
                d_pre[entpair_now]["relation"] = relation_now
                d_pre[entpair_now]["score"] = score_now
        else:
            d_pre[entpair_now] = {"score": score_now, "relation": relation_now}
    # prettify format      
    lst_pre = list()
    for k in d_pre:
        if d_pre[k]["score"] > threshold:
            d_pretty = {
                "head": k.decode("utf-8").split("#")[0],
                "tail": k.decode("utf-8").split("#")[1],
                "relation": D_REL[d_pre[k]["relation"]],
                "score": d_pre[k]["score"]}
            lst_pre.append(d_pretty)
    return lst_pre


# Main
##########################################################################################################################
def main(sample_text, head, tail, model=model):
    # input: sample_text: string
    #     head: string, entity_1
    #     tail: string, entity_2
    #     model: relation extraction model
    # output: lst_res = [(head, tail, relation, score), ...]
    # intialization
    if not isinstance(sample_text, str):
        return []
    if not isinstance(head, str):
        head = ""
    if not isinstance(tail, str):
        tail = ""
    sample_prefix = "sample_now"
    dataset_name = "DuNRE"
    tag = False
    sample_path = sample_prefix+".json"
    # pre-processing
    tag = string2json(sample_text, head, tail, sample_path)
    # extract relation
    if tag:
        train_loader, test_loader = load_data(sample_path, dataset_name)
        framework = nrekit.framework.re_framework(train_loader, test_loader)
        ckpt_name = "./checkpoint/" + dataset_name + "_" + model.encoder + "_" + model.selector
        _, pred_result = framework.test(model, ckpt=ckpt_name, return_result=True)
        pred_result = post_ranking(pred_result, threshold=0)
    else:
        pred_result = []
    return pred_result

    
# Test
##########################################################################################################################
@app.route("/", methods=["POST"])
def hello():
    context = request.json.get("context")
    head = request.json.get("head")
    tail = request.json.get("tail")
    result = main(context, head, tail)
    return jsonify({"result": result})


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=7777, threaded=True)
