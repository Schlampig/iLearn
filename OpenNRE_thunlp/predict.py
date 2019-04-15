import os
import sys
import json
import itertools
import numpy as np
import tensorflow as tf
import jieba.posseg as pseg
import jieba
import nrekit
import ipdb


# Pre-process sample
##########################################################################################################################
def string2json(text, save_path=None):
    # input: text is a string
    # output: sample is a json， save sample to a json file and return True
    assert isinstance(save_path, str)
    # generate sentence and eligible entities
    lst_text = pseg.lcut(text)
    sentence = list()
    lst_entity = list()
    for i, j in lst_text:
        if ('n' in j) or (j in ['i', 'j', 's', 'l']):  # 名词，成语、习语、空间词、临时语，也包含未知词"un"
            lst_entity.append(i)
        sentence.append(i)
    sentence = " ".join(sentence)
    lst_entity = list(set(lst_entity))
    # generate sample with json structure
    sample = list()
    for head, tail in itertools.combinations(lst_entity, 2):  # 候选词两两组合
        d = {
            "sentence": sentence,
            "head": {"word": str(head), "id": str(head)},
            "tail": {"word": str(tail), "id": str(tail)},
            "relation": ""}
        sample.append(d)
        # 对称
        d = {
            "sentence": sentence,
            "head": {"word": str(tail), "id": str(tail)},
            "tail": {"word": str(head), "id": str(head)},
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


# Define framework and load model
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

   
# Ranking
##########################################################################################################################
def ranking(lst_res, threshold):
    # input: lst_res = [d_1, d_2, ...], where d_i = {"score": float, "entpair": string(byte), "relation": int}
    #     threshold, a float in (0, 1] to decide whether the current entpair is accepted
    # output: lst_pre, same structure as lst_res containing filtered items
    d_pre = dict()
    for res in lst_res:
        score_now = res["score"]
        entpair_now = res["entpair"]
        relation_now = res["relation"]
        if entpair_now in d_pre:
            if score_now > d_pre[entpair_now]["score"]:
                d_pre[entpair_now]["relation"] = relation_now
                d_pre[entpair_now]["score"] = score_now
        else:
            d_pre[entpair_now] = {"score": score_now, "relation": relation_now}
    lst_pre = [{"entpair": k.decode("utf-8"), "relation": d_pre[k]["relation"], "score": d_pre[k]["score"]} for k in d_pre if d_pre[k]["score"] > threshold]
    return lst_pre
    

# Main
##########################################################################################################################
def main(sample_text, model):
    sample_prefix = "sample_now"
    dataset_name = "DuNRE"
    if string2json(sample_text, sample_prefix+".json"):
        train_loader, test_loader = load_data(sample_prefix+".json", dataset_name)
        framework = nrekit.framework.re_framework(train_loader, test_loader)
        auc, pred_result = framework.test(model, ckpt="./checkpoint/" + dataset_name + "_" + model.encoder + "_" + model.selector, return_result=True)
        pred_result = ranking(pred_result, threshold=0)
    else:
        auc, pred_result = 0, []
    return auc, pred_result


# Trifles
##########################################################################################################################
def print_result(pred_result, auc, relation_path="../datasets/DuNRE/relation_nre.json", s=None):
    # load relation dictionary
    d_rel = dict()
    with open(relation_path) as f:
        d = json.load(f)
    for k in d:
        d_rel[d[k]] = k
    # print results
    print()
    print("predictions:")
    if s is not None:
        print("sentence: ", s)
        print()
        print("prediction:")
    for pred in pred_result:
        e_head = pred["entpair"].split("#")[0]
        e_tail = pred["entpair"].split("#")[1]
        r = d_rel[pred["relation"]]
        score = pred["score"]
        print("  {}->{}->{}, with score: {}".format(e_head, r, e_tail, score))  
    print()
    print("AUC score: ", auc)
    return None


def text2sentence(text):
    lst_text = jieba.lcut(text)
    sentence = " ".join(lst_text)
    return sentence


def test_DuNRE(model, load_path="../datasets/DuNRE/dev_nre.json", tag=False):
    # 从公共数据集中随机选一个样本测试
    import random
    sample_prefix = "sample_now"
    dataset_name = "DuNRE"
    # load and pre_process data
    lst_sentence = list()
    with open(load_path) as f:
        lines = json.load(f)
        sample = random.choice(lines)  # 随机选一个样本
        sentence = sample["sentence"]
    if tag:
        sample["relation"] = ""
    with open(sample_prefix+".json", "w") as f:
        json.dump([sample], f)
    # predict
    train_loader, test_loader = load_data(sample_prefix+".json", dataset_name)
    framework = nrekit.framework.re_framework(train_loader, test_loader)
    auc, pred_result = framework.test(model, ckpt="./checkpoint/" + dataset_name + "_" + model.encoder + "_" + model.selector, return_result=True)
    
    lst_p = [(r["entpair"].decode("utf-8"), r["relation"], r["score"]) for r in pred_result]
    print(lst_p)
    print()
    
    pred_result = ranking(pred_result, threshold=0)
    print_result(pred_result, auc, s=sentence)
    return None
    

# Test
##########################################################################################################################
if __name__ == "__main__":
    s = """
       约翰·罗纳德·瑞尔·托尔金，英国作家、诗人、语言学家及大学教授，以创作经典严肃奇幻作品《霍比特人》、《魔戒》与《精灵宝钻》而闻名于世。
       """
    auc, pred_result = main(s, model)
    print_result(pred_result, auc, s=s)
#     test_DuNRE(model)
    print("Finished.")
