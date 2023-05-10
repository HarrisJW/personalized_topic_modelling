from sklearn.datasets import fetch_20newsgroups
import pandas as pd
import numpy as np
from model import *


class Oraclev1:
    def __init__(self, docs, labels, model):
        self.docs = docs
        self.document_ids = np.array(range(0, len(docs)))
        self.labels = labels
        self.global_model = None
        self.model = model
        self.used_ids = set()

    # find cluster purity
    def get_purity(self, labels):
        ground_truth = self.labels
        d = {i: {} for i in set(labels)}
        for idx, (label, glabel) in enumerate(zip(labels, ground_truth)):
            _d = d[label]
            if glabel in _d:
                _d[glabel] += 1
            else:
                _d[glabel] = 1

        p = {}
        impure_docs = {}
        pure_docs = {}
        for i in d:
            p[i] = max(d[i].values())/sum(d[i].values())
            impure_docs[i] = sum(d[i].values()) - max(d[i].values())
            pure_docs[i] = sum(d[i].values())
        return d, p, impure_docs, pure_docs

    # find document by cluster and associated ground truth
    def get_feedback(self, cluster_labels):
        d, p, impure_docs, pure_docs = self.get_purity(cluster_labels)
        """
        impure_docs = {0:75,1:30,2:20,3:10}
        pure docs = {0:100,1:100,2:100,3:100}
        p={0:0.25, 1: 0.70, 2:0.80, 3:0.90}
        d={
            0:{0:25, 1:25, 2:25, 3:25},
            1:{0:70, 1:10, 2:10, 3:10},
            2:{0:10, 1:80, 2:5, 3:5},
            3:{0:0, 1:5, 2:5, 3:90},
        }
        """
        import random
        most_impure_cluster = max(impure_docs,key=lambda k:impure_docs[k])# 0
        # choose the ground truth target that has the most number of documents in the most_impure_cluster
        gt_target = max(d[most_impure_cluster],key=lambda x:d[most_impure_cluster][x]) # 0
        # find documents in most_impure_cluster belonging to gt_target and move it away from rest
        feedback = []
        for docid, cluster, gt in zip(self.document_ids, cluster_labels, self.labels):
            if cluster==most_impure_cluster and gt==gt_target:
                feedback.append((docid, -1))
        return feedback


    def get_feedback_neg(self, cluster_labels):
        d, p, impure_docs, pure_docs = self.get_purity(cluster_labels)
        """
        impure_docs = {0:75,1:30,2:20,3:10}
        pure docs = {0:100,1:100,2:100,3:100}
        p={0:0.25, 1: 0.70, 2:0.80, 3:0.90}
        d={
            0:{0:25, 1:25, 2:25, 3:25},
            1:{0:70, 1:10, 2:10, 3:10},
            2:{0:10, 1:80, 2:5, 3:5},
            3:{0:0, 1:5, 2:5, 3:90},
        }
        """
        import random
        most_impure_cluster = max(impure_docs,key=lambda k:impure_docs[k])# 0
        feedback = []
        for docid, cluster, gt in zip(self.document_ids, cluster_labels, self.labels):
            # if cluster==most_impure_cluster and gt!=most_impure_cluster: 
                # assumes that the clustering is done such that gt == cluster-number
            if docid not in self.used_ids and cluster==most_impure_cluster and gt!=most_impure_cluster:
                feedback.append((docid, gt))
        return feedback


