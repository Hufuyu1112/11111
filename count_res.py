# coding: utf-8
# author: lw
# create date: 2020/07/2

from __future__ import division

import json
import random
import warnings
from sklearn.metrics import f1_score, normalized_mutual_info_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

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
            self.drug_entity_name_emb_dict[p_name] = list(p_emb)
        for p_id,p_name in self.protein_id_name_dict.items():
            p_emb = map(lambda x: float(x),entity_emb[int(p_id)])
            self.protein_entity_name_emb_dict[p_name] = list(p_emb)


    def classification(self, x, y):
        """
        train_test_split函数：
            用于将矩阵随机划分为训练子集和测试子集，并返回划分好的训练集测试集样本和训练集测试集标签
            x:被划分的样本特征集
            y:被划分的样本标签
            test_size：如果是浮点数，在0-1之间，表示样本占比；如果是整数的话就是样本的数量
            random_state：是随机数的种子（0或不填，每次不一样的随机数）
        LogisticRegression函数：
            LogisticRegression回归模型在Sklearn.linear_model子类下，调用sklearn逻辑回归算法步骤比较简单，即：
            (1) 导入模型。调用逻辑回归LogisticRegression()函数。
            (2) fit()训练。调用fit(x,y)的方法来训练模型，其中x为数据的属性，y为所属类型。
            (3) predict()预测。利用训练得到的模型对数据集进行预测，返回预测结果
        f1_score函数：
            F1分数（F1-score）是分类问题的一个衡量指标。
            它是精确率和召回率的调和平均数，最大为1，最小为0。
            y_valid：目标的真实类别
            y_valid_pred：分类器预测得到的类别
            average：macro:先计算出每个类别的F1值，然后去平均
            average：micro:先计算总体的TP，FN和FP的数量，再计算F1
        """
        x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, random_state=9)

        lr = LogisticRegression()
        lr.fit(x_train, y_train)
        y_valid_pred = lr.predict(x_valid)
        print("******************")
        AUC = roc_auc_score(y_valid, y_valid_pred, average='micro')
        AUPR = average_precision_score(y_valid, y_valid_pred, average='macro')
        print('AUC: {}'.format(AUC))
        print('AUPR: {}'.format(AUPR))
        return AUC, AUPR


    def count1(self):
        """
        效仿第一篇论文计算AUC 的方法
        :return:
        """
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
        """
        根据官方文档写的AUC
        :return:
        """
        f1 = open('../data/dblp/2Mytest_drug_protein.txt', 'r')
        # edges = [list(map(int, i.strip().split('\t'))) for i in f1]
        edges = []
        lines = f1.readlines()
        for line in lines:
            t = line.strip().split('\t')
            edges.append((t[0], t[1], t[2]))
        f1.close()
        P = 0
        N = 0
        pos_relation = []
        neg_relation = []
        for n1, n2, flag in edges:
            if flag == "0":
                neg_relation.append((n1, n2))
                N += 1
            else:
                pos_relation.append((n1, n2))
                P += 1
        sum_result = 0
        # print(list(self.drug_entity_name_emb_dict["DB0005
        for pos in pos_relation:
            dot1 = np.dot(self.drug_entity_name_emb_dict[self.drug_id_name_dict[str(pos[0])]],
                          self.protein_entity_name_emb_dict[self.protein_id_name_dict[str(pos[1])]])
            for neg in neg_relation:
                dot2 = np.dot(self.drug_entity_name_emb_dict[self.drug_id_name_dict[str(neg[0])]],
                              self.protein_entity_name_emb_dict[self.protein_id_name_dict[str(neg[1])]])
                if dot1 > dot2:
                    sum_result += 1
                elif dot1 == dot2:
                    sum_result += 0.5
        print("2:Auc value(D-P):{}".format(float(sum_result) / (N * P)))

    def count3(self):
        """
        根据自己理解写的计算AUPR和AUC
        :return:
        """
        f1 = open('../data/dblp/2Mytest_drug_protein.txt', 'r')
        # edges = [list(map(int, i.strip().split('\t'))) for i in f1]
        edges = []
        lines = f1.readlines()
        for line in lines:
            t = line.strip().split('\t')
            edges.append((t[0], t[1], t[2]))
        f1.close()
        P = 0
        N = 0
        pos_relation = []
        neg_relation = []
        for n1, n2, flag in edges:
            if flag == "0":
                neg_relation.append((n1, n2))
                N += 1
            else:
                pos_relation.append((n1, n2))
                P += 1
        sum_result = 0
        y_ture = []
        y_score = []
        # 正样例得计算
        for pos in pos_relation:
            dot1 = np.dot(self.drug_entity_name_emb_dict[self.drug_id_name_dict[str(pos[0])]],
                          self.protein_entity_name_emb_dict[self.protein_id_name_dict[str(pos[1])]])
            tmp_sum = 0
            for neg in neg_relation:
                dot2 = np.dot(self.drug_entity_name_emb_dict[self.drug_id_name_dict[str(neg[0])]],
                              self.protein_entity_name_emb_dict[self.protein_id_name_dict[str(neg[1])]])
                if dot1 > dot2:
                    tmp_sum += 1
                elif dot1 == dot2:
                    tmp_sum += 0.5
            y_ture.append(1)
            y_score.append(tmp_sum / len(neg_relation))

        # 负样例的计算
        for neg in neg_relation:
            dot2 = np.dot(self.drug_entity_name_emb_dict[self.drug_id_name_dict[str(neg[0])]],
                          self.protein_entity_name_emb_dict[self.protein_id_name_dict[str(neg[1])]])
            tmp_sum = 0
            for pos in pos_relation:
                dot1 = np.dot(self.drug_entity_name_emb_dict[self.drug_id_name_dict[str(pos[0])]],
                              self.protein_entity_name_emb_dict[self.protein_id_name_dict[str(pos[1])]])
                if dot1 > dot2:
                    tmp_sum += 1
                elif dot1 == dot2:
                    tmp_sum += 0.5
            y_ture.append(0)
            y_score.append(tmp_sum / len(pos_relation))
        # with open("zz_test_score.txt", "w") as f:
        #     for i in range(len(y_ture)):
        #         f.write(str(y_ture[i]) + "\t" + str(y_score[i]) + "\n")
        AUC =  roc_auc_score(y_ture, y_score)
        AUPR = average_precision_score(y_ture, y_score)
        print("AUC (roc_auc_score):{}".format(AUC))
        print("AUPR (average_precision_score):{}".format(AUPR))

    def checkRandomNote(self, node_a, random_node, edges):
        if [node_a,random_node] in edges:
            return True
        return False


    def get_x_y(self):
        """
        根据某个代码，点积作为预测值，计算AUPR和AUC
        :return:
        """
        f1 = open('../data/dblp/2Mytest_drug_protein.txt', 'r')
        # edges = [list(map(int, i.strip().split('\t'))) for i in f1]
        edges = []
        lines = f1.readlines()
        for line in lines:
            t = line.strip().split('\t')
            edges.append((t[0], t[1], t[2]))
        f1.close()
        P = 0
        N = 0
        pos_relation = []
        neg_relation = []
        for n1, n2, flag in edges:
            if flag == "0":
                neg_relation.append((n1, n2))
                N += 1
            else:
                pos_relation.append((n1, n2))
                P += 1
        sum_result = 0
        y_ture = []
        y_score = []
        # 正样例得计算
        for pos in pos_relation:
            embd1 = np.array(self.drug_entity_name_emb_dict[self.drug_id_name_dict[str(pos[0])]])
            embd2 = np.array(self.protein_entity_name_emb_dict[self.protein_id_name_dict[str(pos[1])]])
            dot1 = np.dot(embd1, embd2)
            y_ture.append(1)
            y_score.append(self.sigmoid(dot1))

        # 负样例的计算
        for neg in neg_relation:
            embd1 = np.array(self.drug_entity_name_emb_dict[self.drug_id_name_dict[str(neg[0])]])
            embd2 = np.array(self.protein_entity_name_emb_dict[self.protein_id_name_dict[str(neg[1])]])
            dot2 = np.dot(embd1, embd2)
            y_ture.append(0)
            y_score.append(self.sigmoid(dot2))

        AUC = roc_auc_score(y_ture, y_score)
        AUPR = average_precision_score(y_ture, y_score)
        print("AUC:{}".format(AUC))
        print("AUPR:{}".format(AUPR))
        return y_score, y_ture

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

if __name__ == '__main__':
    # 把训练生成的.json文件存放到目录 "output/embJson/***.json"
    # 把下面的文件名改成对应的
    exp = Evaluation()
    emb1 = exp.load_emb('../res/dblp/embedding.vec.ap_pt_apt+pc_apc.json')
    exp.evaluation(emb1)
    exp.count1()
    exp.count2()
    #exp.count3()
    x, y = exp.get_x_y()
    # exp.classification(x, y)



