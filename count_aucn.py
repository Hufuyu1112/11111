# coding: utf-8
# author: lw
# create date: 2020/05/26

from __future__ import division

import json
import random
import warnings

import numpy as np

warnings.filterwarnings('ignore')

class Evaluation:
    def __init__(self):
        self.drug_entity_name_emb_dict = {}
        self.protein_entity_name_emb_dict = {}
        self.drug_id_name_dict = {}
        self.protein_id_name_dict = {}
        np.random.seed(1)

    def load_emb(self, emb_name):
        """
        load embeddings
        :param emb_name:
        :return:
        """
        with open(emb_name, 'r') as emb_file:
            emb_dict = json.load(emb_file)
        return emb_dict

    def evaluation(self,emb_dict):
        entity_emb = emb_dict['ent_embeddings.weight']
        with open('../data/dblp/node2id.txt','r') as e2i_file:
            lines = e2i_file.readlines()

        for i in range(1,len(lines)):
            tokens = lines[i].strip().split('\t')
            if lines[i][0:2] == 'DB':
                self.drug_id_name_dict[tokens[1]] = tokens[0]
            if lines[i][0] in ["O", "P", "Q", "A"]:
                self.protein_id_name_dict[tokens[1]] = tokens[0]
        for p_id,p_name in self.drug_id_name_dict.items():
            p_emb = map(lambda x: float(x),entity_emb[int(p_id)])
            self.drug_entity_name_emb_dict[p_name] = p_emb
        for p_id,p_name in self.protein_id_name_dict.items():
            p_emb = map(lambda x: float(x),entity_emb[int(p_id)])
            self.protein_entity_name_emb_dict[p_name] = p_emb

    def count1(self):
        f1 = open('../data/dblp/1Mytest_drug_protein.txt', 'r')
        edges = [list(map(int, i.strip().split('\t')[:2])) for i in f1]
        nodes = []
        len_edges = len(edges)
        for i in range(len_edges):
            nodes.append(edges[i][1])
        a = b = 0
        for i, j in edges:
            if self.drug_id_name_dict[str(i)] and self.protein_id_name_dict[str(j)]:
                dot1 = np.dot(self.drug_entity_name_emb_dict[self.drug_id_name_dict[str(i)]],
                              self.protein_entity_name_emb_dict[self.protein_id_name_dict[str(j)]])
                random_node = random.sample(nodes, 1)[0]
                while self.checkRandomNote(i, random_node, edges):
                    random_node = random.sample(nodes, 1)[0]
                dot2 = np.dot(self.drug_entity_name_emb_dict[self.drug_id_name_dict[str(i)]],
                              self.protein_entity_name_emb_dict[self.protein_id_name_dict[str(random_node)]])
                if dot1 > dot2:
                    a += 1
                elif dot1 == dot2:
                    a += 0.5
                b += 1
        print("1:Auc value(D-P):{}".format(float(a) / b))

    def count2(self):
        f1 = open('../data/dblp/2Mytest_drug_protein.txt', 'r')
        edges = [list(map(int, i.strip().split('\t'))) for i in f1]
        P = 0
        N = 0
        pos_relation = []
        neg_relation = []
        for n1, n2, flag in edges:
            if flag == 0:
                neg_relation.append((n1, n2))
                N += 1
            else:
                pos_relation.append((n1, n2))
                P += 1
        sum_rsult = 0
        for pos in pos_relation:
            dot1 = np.dot(self.drug_entity_name_emb_dict[self.drug_id_name_dict[str(pos[0])]],
                          self.protein_entity_name_emb_dict[self.protein_id_name_dict[str(pos[1])]])
            for neg in neg_relation:
                dot2 = np.dot(self.drug_entity_name_emb_dict[self.drug_id_name_dict[str(neg[0])]],
                              self.protein_entity_name_emb_dict[self.protein_id_name_dict[str(neg[1])]])
                if dot1 > dot2:
                    sum_rsult += 1
                elif dot1 == dot2:
                    sum_rsult += 0.5
        print("2:Auc value(D-P):{}".format(float(sum_rsult) / (N * P)))

    def checkRandomNote(self, node_a, random_node, edges):
        if [node_a,random_node] in edges:
            return True
        return False


if __name__ == '__main__':
    # 把训练生成的.json文件存放到目录 "output/embJson/***.json"
    # 把下面的文件名改成对应的
    exp = Evaluation()
    emb1 = exp.load_emb('../res/dblp/embedding.vec.ap_pt_apt+pc_apc200dim-noar.json')
    exp.evaluation(emb1)
    exp.count1()
    exp.count2()
