from sklearn.datasets import fetch_20newsgroups
import pandas as pd
import numpy as np
# import umap
import hdbscan
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
# from preprocess import *
from datetime import datetime
from typing import Iterable, Dict
from torch import Tensor, full
import torch 
from torch import nn
import torch.nn.functional as F

class Loss(nn.Module):
    def __init__(self,model):
        super().__init__()
        self.model = model 
        self.sm = nn.Linear(384*3,2)

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        embeddings = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        vec = torch.cat([embeddings[0],embeddings[1],embeddings[0]-embeddings[1]],dim=1)
        probs = torch.softmax(self.sm(vec), dim=1) + 1e-5
        loss = labels.view(-1,1)*torch.log(probs[:,0])+(1-labels.view(-1,1))*torch.log(probs[:,1])
        return loss.mean()

def get_first(sentence_features):
    return {k:sentence_features[k][0:1] for k in sentence_features}

def get_mask(sentence_features):
    import copy 
    sf = copy.copy(sentence_features)
    sf['input_ids'] = 0+sf['input_ids']
    sf['input_ids'][torch.rand_like(sf['input_ids'].float())<0.15] = 103
    return sf

class SemLoss(nn.Module):
    cvs = None
    def __init__(self, model):
        super().__init__()
        self.model = model
        # self.linear = nn.Linear(384,20)
    
    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        embeddings = self.model(sentence_features[0])['sentence_embedding']
        aug = get_mask(sentence_features[0])
        # embeddings2 = self.model(aug)['sentence_embedding']
        # logprobs = torch.log_softmax(self.linear(embeddings),dim=1)

        class_vectors = [self.model(get_first(sentence_features[i]))['sentence_embedding'] for i in range(1,len(sentence_features))]
        SemLoss.cvs = torch.stack(class_vectors).squeeze(1).detach().cpu().numpy()
        cosine_distance = lambda x,y: 1-F.cosine_similarity(x, y)
        all_dists = [cosine_distance(embeddings,cv) for cv in class_vectors]
        all_dists = torch.stack(all_dists).t()
        s = labels==-1
        labels[s] = all_dists[s].min(dim=1)[1]
        oh = F.one_hot(labels,len(class_vectors)).float().to(all_dists)
        margin = 0.6
        loss = torch.square((all_dists*oh).sum(dim=1)) + torch.square(F.relu(margin-all_dists*(1-oh))).sum(dim=1)
        
        return 0.5*loss[~s].mean() + 0.5*0.1*loss[s].mean() #+ torch.softmax(-all_dists,dim=1).mean(dim=0).std()

class NegSemLoss(nn.Module):
    cvs = None
    def __init__(self, model):
        super().__init__()
        self.model = model
        # self.linear = nn.Linear(384,20)
    
    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        embeddings = self.model(sentence_features[0])['sentence_embedding']

        class_vectors = [self.model(get_first(sentence_features[i]))['sentence_embedding'] for i in range(1,len(sentence_features))]
        SemLoss.cvs = torch.stack(class_vectors).squeeze(1).detach().cpu().numpy()
        cosine_distance = lambda x,y: 1-F.cosine_similarity(x, y)
        all_dists = [cosine_distance(embeddings,cv) for cv in class_vectors]
        all_dists = torch.stack(all_dists).t()
        oh_positive = F.one_hot(labels[labels>=0],len(class_vectors)).float().to(all_dists)
        oh_negative = F.one_hot(torch.abs(labels[labels<0]+1),len(class_vectors)).float().to(all_dists)
        ad_positive = all_dists[labels>=0]
        ad_negative = all_dists[labels<0]
        margin = 0.5
        # for labels>0:
        loss_positive = (torch.square((ad_positive*oh_positive).sum(dim=1)) + torch.square(F.relu(margin-ad_positive*(1-oh_positive))).sum(dim=1))
        loss_negative = torch.square(F.relu(margin-ad_negative*(oh_negative))).sum(dim=1)
        
        return 0.5*(loss_positive.sum()+loss_negative.sum())/labels.shape[0] #+ torch.softmax(-all_dists,dim=1).mean(dim=0).std()

# class SemLoss(nn.Module):
#     def __init__(self, model):
#         super().__init__()
#         self.model = model
    
#     def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
#         embeddings = self.model(sentence_features[0])['sentence_embedding']
#         aug = get_mask(sentence_features[0])
#         # embeddings2 = self.model(aug)['sentence_embedding']
#         class_vectors = [self.model(get_first(sentence_features[i]))['sentence_embedding'] for i in range(1,len(sentence_features))]
#         cosine_distance = lambda x,y: 1-F.cosine_similarity(x, y)
#         all_dists = [cosine_distance(embeddings,cv) for cv in class_vectors]
#         all_dists = torch.stack(all_dists).t()
#         s = labels==-1
#         labels[s] = all_dists[s].min(dim=1)[1]
#         oh = F.one_hot(labels,len(class_vectors)).float().to(all_dists)
#         margin = 0.5
#         loss = torch.square((all_dists*oh).sum(dim=1)) + torch.square(F.relu(margin-all_dists*(1-oh))).sum(dim=1)

#         return 0.5*loss[~s].mean() + 0.5*0.1*loss[s].mean()
#         print(all_dists[0].shape)
        # class_vectors = torch.cat(class_vectors,dim=0).t()
        # cv_penalty = torch.square(class_vectors-class_vectors.mean(dim=1,keepdim=True)).mean()
        # logits = embeddings@class_vectors
        # log_logits = torch.log_softmax(logits, dim=1)
        # ce_loss = 0
        # count = 0
        # for i in range(len(sentence_features)-1):
        #     ce_loss = ce_loss - log_logits[labels==i].sum()
        #     count += (labels==i).sum()
        # loss = ce_loss/count
        # return loss 

class MLModel:
    def __init__(self, documents, config):
        self.documents = documents
        self.np_docs = np.array(documents, dtype="object")
        self.document_ids = np.array(range(0, len(documents)))
        self.id2doc = dict(zip(self.document_ids, documents))
        self.pretrained_model = None
        self.document_embeddings = None
        self.umap_embeddings_cluster = None
        self.umap_data_viz = None
        self.clusters = None
        self.topic_vectors = None
        self.new_path=None
        self.bank = []
        self.config=config
        self.workdir=config['workdir']
        self.feedbacks = []
        self.v1_dataloader = None 
        self.v2_dataloader = None
        self.v2v1_negdataloader = None
        self.v2v1_posdataloader = None

    def set_pretrained_model(self, model_path):
        self.pretrained_model = SentenceTransformer(model_path)
        #self.document_embeddings = np.load(document_embeddings)
        self.document_embeddings = self.pretrained_model.encode(self.documents)
        if model_path == "all-MiniLM-L6-v2":
            self.model_name = "default"
        else:
            self.model_name = model_path


    def reduce_dims(self):
        """
        res = umap.UMAP(n_neighbors=25,
                                                 n_components=5,
                                                 metric='cosine',
                                                 random_state=42).fit_transform(np.random.randn(10000,200))
        
        """
        self.umap_embeddings_cluster = umap.UMAP(n_neighbors=25,
                                                 n_components=5,
                                                 metric='cosine',
                                                 random_state=42).fit_transform(self.document_embeddings)

        # Visualization by UMAP reduction further to 2d
        self.umap_data_viz = umap.UMAP(n_neighbors=25, n_components=2, min_dist=0.5, metric='cosine',
                                       random_state=42).fit_transform(
            self.document_embeddings)

        self.x_range = [float(min(self.umap_data_viz[:, 0])-1.5), float(max(self.umap_data_viz[:, 0])+1.5)]
        self.y_range = [float(min(self.umap_data_viz[:, 1])-1.5), float(max(self.umap_data_viz[:, 1])+1.5)]

    def make_clusters(self, min_cluster_size):
        config = self.config
        if (config['clus_method']=='hdb'):
            #HDBSCAN Clustering
            import hdbscan
            self.clusters = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                            metric='euclidean',
                                            cluster_selection_method='eom').fit(self.umap_embeddings_cluster)
            self.clusters = self.clusters.labels_

        if (config['clus_method']=='km'):
            self.clusters = KMeans(n_clusters=min_cluster_size, random_state=0).fit(self.umap_embeddings_cluster)
            self.clusters = self.clusters.labels_

        
        if (config['clus_method']=='gm'):
            mean_init = None
            if SemLoss.cvs is not None:
                mean_init = (SemLoss.cvs)
            gm = GaussianMixture(n_components=min_cluster_size, random_state=0, covariance_type='diag',means_init=mean_init).fit(self.umap_embeddings_cluster)
            self.clusters = gm.predict(self.umap_embeddings_cluster)
       
        cluster_labels = self.clusters
        unique_labels = set(cluster_labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)
        self.topic_vectors = (np.vstack([self.document_embeddings[np.where(cluster_labels == label)[0]]
                      .mean(axis=0) for label in unique_labels]))

        # self.topic_words, self.topic_word_scores = self.find_topic_words_and_scores(topic_vectors=self.topic_vectors)
        # self.doc_top, self.doc_dist = self.calculate_documents_topic(self.topic_vectors, self.document_embeddings)
        # self.doc_top = np.array(self.doc_top, dtype="int")
        self.doc_top = self.clusters
        return cluster_labels

    def get_topic_words(self, topic_number):
        return self.topic_words[topic_number][:10], self.topic_word_scores[topic_number][:10]

    def get_document_topic(self, doc_id):
        return self.doc_top[doc_id]

    def render_plot_data(self):
        #hue_cat = [self.topic_words[top][0] for top in self.doc_top]
        hue_cat = [int(top) for top in self.doc_top]
        return self.umap_data_viz[:, 0], self.umap_data_viz[:, 1], hue_cat, ["Topic: "+str(self.doc_top[a])+" Doc_id: "+str(a)+" - "+str(b[:100]) for a,b in zip(self.document_ids, self.documents)]

    def render_selected_topic(self, topics):
        hue_cat = [self.topic_words[top][0] for top in self.doc_top]
        cluster_labels = pd.Series(self.clusters)
        s = (cluster_labels == topics[0])
        return self.umap_data_viz[s][:, 0], self.umap_data_viz[s][:, 1], self.topic_words[topics[0]][0], [str(a)+" - "+str(b[:100]) for a,b in zip(self.document_ids[s], self.np_docs[s])]

    def search_documents_by_topic(self, topic_num, num_docs=10, return_documents=True):

        topic_document_indexes = np.where(self.doc_top == topic_num)[0]
        topic_document_indexes_ordered = np.flip(np.argsort(self.doc_dist[topic_document_indexes]))
        doc_indexes = topic_document_indexes[topic_document_indexes_ordered][0:num_docs]
        doc_scores = self.doc_dist[doc_indexes]
        doc_ids = self.document_ids[doc_indexes]

        if self.documents is not None and return_documents:
            documents = self.np_docs[doc_indexes]
            doc_topics = self.doc_top[doc_indexes]
            return documents, doc_topics, doc_ids, doc_scores
        else:
            return doc_scores, doc_ids

    def find_topic_words_and_scores(self, topic_vectors):
        topic_words = []
        topic_word_scores = []

        res = np.inner(topic_vectors, self.word_vectors)
        top_words = np.flip(np.argsort(res, axis=1), axis=1)
        top_scores = np.flip(np.sort(res, axis=1), axis=1)

        for words, scores in zip(top_words, top_scores):
            topic_words.append([self.vocab[i] for i in words[0:50]])
            topic_word_scores.append(scores[0:50])

        topic_words = np.array(topic_words)
        topic_word_scores = np.array(topic_word_scores)

        return topic_words, topic_word_scores

    def calculate_documents_topic(self, topic_vectors, document_vectors, dist=True, num_topics=None):
        batch_size = 10000
        doc_top = []
        if dist:
            doc_dist = []

        if document_vectors.shape[0] > batch_size:
            current = 0
            batches = int(document_vectors.shape[0] / batch_size)
            extra = document_vectors.shape[0] % batch_size

            for ind in range(0, batches):
                res = np.inner(document_vectors[current:current + batch_size], topic_vectors)

                if num_topics is None:
                    doc_top.extend(np.argmax(res, axis=1))
                    if dist:
                        doc_dist.extend(np.max(res, axis=1))
                else:
                    doc_top.extend(np.flip(np.argsort(res), axis=1)[:, :num_topics])
                    if dist:
                        doc_dist.extend(np.flip(np.sort(res), axis=1)[:, :num_topics])

                current += batch_size

            if extra > 0:
                res = np.inner(document_vectors[current:current + extra], topic_vectors)

                if num_topics is None:
                    doc_top.extend(np.argmax(res, axis=1))
                    if dist:
                        doc_dist.extend(np.max(res, axis=1))
                else:
                    doc_top.extend(np.flip(np.argsort(res), axis=1)[:, :num_topics])
                    if dist:
                        doc_dist.extend(np.flip(np.sort(res), axis=1)[:, :num_topics])
            if dist:
                doc_dist = np.array(doc_dist)
        else:
            res = np.inner(document_vectors, topic_vectors)

            if num_topics is None:
                doc_top = np.argmax(res, axis=1)
                if dist:
                    doc_dist = np.max(res, axis=1)
            else:
                doc_top.extend(np.flip(np.argsort(res), axis=1)[:, :num_topics])
                if dist:
                    doc_dist.extend(np.flip(np.sort(res), axis=1)[:, :num_topics])

        if num_topics is not None:
            doc_top = np.array(doc_top)
            if dist:
                doc_dist = np.array(doc_dist)

        if dist:
            return doc_top, doc_dist
        else:
            return doc_top

    def _get_combined_vec(self, vecs, vecs_neg):
        combined_vector = np.zeros(self.document_embeddings.shape[1], dtype=np.float64)
        for vec in vecs:
            combined_vector += vec
        for vec in vecs_neg:
            combined_vector -= vec
        combined_vector /= (len(vecs) + len(vecs_neg))
        combined_vector = self._l2_normalize(combined_vector)
        return combined_vector

    def get_document_detail(self, doc_id):
        doc_text = str(self.documents[doc_id])
        doc_topic = int(self.doc_top[doc_id])
        return doc_text, doc_topic

    def search_documents_by_keywords(self, keywords, num_docs, keywords_neg=None, return_documents=True,
                                     use_index=False, ef=None):
        if keywords_neg is None:
            keywords_neg = []
        # self._validate_num_docs(num_docs)
        # keywords, keywords_neg = self._validate_keywords(keywords, keywords_neg)
        word_vecs = self.word_vectors[[self.word_indexes[word] for word in keywords]]
        neg_word_vecs = self.word_vectors[[self.word_indexes[word] for word in keywords_neg]]

        combined_vector = self._get_combined_vec(word_vecs, neg_word_vecs)
        doc_indexes, doc_scores = self._search_vectors_by_vector(self.document_embeddings,
                                                                 combined_vector, num_docs)
        doc_ids = self.document_ids[doc_indexes]

        if self.documents is not None and return_documents:
            documents = self.np_docs[doc_indexes]
            doc_topics = self.doc_top[doc_indexes]
            return documents, doc_topics, doc_ids, doc_scores
        else:
            return doc_scores, doc_ids

    @staticmethod
    def find_key_by_value(adict, value):
        return list(adict.keys())[list(adict.values()).index(value)]

    @staticmethod
    def find_sankey_from_dict(key, source, target):
        top_name = source[str(key)]
        return int(target.index(top_name))

    def get_overall_viz_meta(self, current, prev):
        current_topics = set(current["topics"].values())
        prev_topics = set(prev["topics"].values())

        common = current_topics.intersection(prev_topics)
        prev_only = prev_topics.difference(current_topics)
        current_only = current_topics.difference(prev_topics)
        return {"common": list(common), "prev_only": list(prev_only), "current_only": list(current_only)}
        

    def topic_level_sankey(self, current, prev, topic):
        current_topics = set(current["topics"].values())
        prev_topics = set(prev["topics"].values())

        common = current_topics.intersection(prev_topics)
        prev_only = prev_topics.difference(current_topics)
        current_only = current_topics.difference(prev_topics)
        sankey_topics = list(current_only) + list(common) + list(prev_only)

        s_t_v__list = []

        current_topic_idx = self.find_key_by_value(current["topics"], topic)
        current_docs = [key for key, value in current["doc_top"].items() if value == int(current_topic_idx)]
        filtered_doc_top = {doc: prev["doc_top"][str(doc)] for doc in current_docs}
        counted = dict(Counter(filtered_doc_top.values()))
        for key, value in counted.items():
            source = self.find_sankey_from_dict(key, prev["topics"], sankey_topics)
            target = int(sankey_topics.index(topic))
            if not source == target:
                s_t_v__list.append([source, target, value])

        if len(s_t_v__list) < 1:
            return [], [], [], []
        return [item[0] for item in s_t_v__list], [item[1] for item in s_t_v__list], [item[2] for item in
                                                                                      s_t_v__list], sankey_topics

    @staticmethod
    def _l2_normalize(vectors):

        if vectors.ndim == 2:
            return normalize(vectors)
        else:
            return normalize(vectors.reshape(1, -1))[0]

    @staticmethod
    def _search_vectors_by_vector(vectors, vector, num_res):
        ranks = np.inner(vectors, vector)
        indexes = np.flip(np.argsort(ranks)[-num_res:])
        scores = np.array([ranks[res] for res in indexes])

        return indexes, scores

    def apply_feedback(self, feedback):
        """
        The feedback list of tuples is used to fine-tune sentence-bert.
        :param feedback: List[Tuple(Int,Int)]
               List of tuples containing document id and target cluster
               The target cluster can optionally be -1.
        :return: None
        """
        config = self.config
        print("Length of feedback:" + len(feedback)) # 25
        full_train_data = []
        source_cluster = int(self.doc_top[feedback[0][0]]) # 0
        cluster_labels = pd.Series(self.clusters)
        ss = (cluster_labels == source_cluster) # 100 True values
        
        for i in range(0,len(feedback)):
            ss[feedback[i][0]]=False
        source_cluster_docs = self.np_docs[ss] # 75 documents
        
        for id_,target in feedback:# 25 times
            for doc in source_cluster_docs:# 75 times
                full_train_data.append(InputExample(texts=[self.documents[id_],doc], label=0))
        print("Length of full_train_data:" +len(full_train_data)) # 25*75

        for i in range(0,len(feedback)):#
            for j in range(i+1, len(feedback)): #
                id1, id2 = feedback[i][0], feedback[j][0]
                full_train_data.append(InputExample(texts=[self.documents[id1], self.documents[id2]], label=1))
                # 25*24/2
        print("Length of full_train_data:" + len(full_train_data))
        self.bank.extend(full_train_data)
        
        import random
        self.v1_dataloader = DataLoader(random.sample(self.bank,min(len(self.bank),200*500*config['epoch'])), shuffle=True, batch_size=200)
        # if (config['loss']=='sbert_type'):
        #     train_loss = Loss(retrain_model)
        # if (config['loss']=='cos'):
    
    def apply_feedbackv2v1_neg(self, feedback):
        """
        The feedback list of tuples is used to fine-tune sentence-bert.
        :param feedback: List[Tuple(Int,Int)]
               List of tuples containing document id and target cluster
               The target cluster can optionally be -1.
        :return: None
        """
        config = self.config
        print("Length of feedback:" + len(feedback)) # 25
        full_train_data = []

        source_cluster = int(self.doc_top[feedback[0][0]]) # 0
        for id_,target in feedback:# 25 times
            full_train_data.append(InputExample(texts=[self.documents[id_],*self.all_cws], label=target))
        print("Length of full_train_data:" + len(full_train_data))
        self.bank.extend(full_train_data)
        bank = self.bank + self.bm25_data
        import random
        self.v2v1_negdataloader = DataLoader(random.sample(bank,min(len(bank),200*500*config['epoch'])), shuffle=True, batch_size=200)
        # if (config['loss']=='sbert_type'):
        #     train_loss = Loss(retrain_model)
        # if (config['loss']=='cos'):

    def finetune(self, model_path):
        retrain_model = SentenceTransformer(model_path)
        train_objectives = []
        if self.v1_dataloader!=None:
            print('Fine-tuning on v1')
            train_objectives.append((self.v1_dataloader,losses.OnlineContrastiveLoss(retrain_model)))
            epochs = 1
        elif self.v2v1_negdataloader!=None:
            print('Fine-tuning on v2-v1 neg')
            train_objectives.append((self.v2v1_negdataloader,NegSemLoss(retrain_model)))
            epochs = self.config['epoch']
        elif self.v2_dataloader!=None:
            print('Fine-tuning on v2')
            train_objectives.append((self.v2_dataloader,SemLoss(retrain_model)))
            epochs = 1

        # Tune the model
        retrain_model.fit(train_objectives=train_objectives, epochs=epochs,warmup_steps=0)

        curr_time = datetime.now().strftime("%Y%m%d-%H:%M:%S")
        new_path = f"{self.workdir}/retrained-model"+curr_time
        retrain_model.save(new_path)
        return new_path

    # def apply_f2w(self, model_path, feedback, train=True):
    #     import random
    #     config = self.config
    #     if feedback is not None:
    #         self.feedbacks.append(feedback)
    #     full_train_data = []
    #     for cw, feedback in self.feedbacks:
    #         clusters = set()
    #         neg_idxs = set()
    #         fb_set = set(feedback)
    #         for doc_id in feedback:
    #             clusters.add(self.doc_top[doc_id])
    #         for doc_id in range(len(self.documents)):
    #             if doc_id not in fb_set and self.doc_top[doc_id] in clusters:
    #                 if not(any([w in self.documents[doc_id].lower() for w in cw])):
    #                     neg_idxs.add(doc_id)
    #         # print(len(neg_idxs))
    #         for i in range(0,len(feedback)):
    #             d1 = self.documents[feedback[i]]
    #             # for j in range(i+1, len(feedback)): #
    #             #     d2 = self.documents[feedback[j]]
    #             #     # neg_idx = random.choice(list(set(range(len(self.documents)))-set(feedback)))
    #             #     full_train_data.append(InputExample(texts=[d1, d2], label=1.0))
    #             # for neg_idx in neg_idxs:
    #             #     full_train_data.append(InputExample(texts=[d1, self.documents[neg_idx]], label=0.0))
    #             full_train_data.append(InputExample(texts=[d1, ' '.join(cw)], label=1.0))
    #         for neg_idx in neg_idxs:
    #             full_train_data.append(InputExample(texts=[self.documents[neg_idx], ' '.join(cw)], label=0.0))
    #     for idx1 in range(len(self.feedbacks)):
    #         for idx2 in range(idx1+1,len(self.feedbacks)):
    #             cw1, _ = self.feedbacks[idx1]
    #             cw2, _ = self.feedbacks[idx2]
    #             full_train_data.append(InputExample(texts=[' '.join(cw1), ' '.join(cw2)], label=0.0))
    #     print(len(full_train_data))
    #     self.v2_bank = full_train_data + self.bank
    #     if not(train):
    #         return
    #     retrain_model = SentenceTransformer(model_path)
    #     import random
    #     train_dataloader = DataLoader(random.sample(self.v2_bank,min(len(self.v2_bank),200*500*config['epoch'])), shuffle=True, batch_size=200)
    #     if (config['loss']=='sbert_type'):
    #         train_loss = Loss(retrain_model)
    #     if (config['loss']=='cos'):
    #         train_loss = SemLoss(retrain_model)
    #         # train_loss = losses.OnlineContrastiveLoss(retrain_model)
    #         # train_loss = losses.CosineSimilarityLoss(retrain_model)

    #     # Tune the model
    #     retrain_model.fit(train_objectives=[(train_dataloader, train_loss)], scheduler='constantlr', epochs=1, warmup_steps=0, optimizer_params={'lr': 2e-5})

    #     curr_time = datetime.now().strftime("%Y%m%d-%H:%M:%S")
    #     new_path = f"{self.workdir}/retrained-model"+curr_time
    #     retrain_model.save(new_path)
    #     return new_path
    
    def apply_f2w(self, feedback, train=True):
        import random
        config = self.config
        if feedback is not None:
            self.feedbacks.append(feedback)
        full_train_data = []
        self.all_cws = all_cws = [' '.join(f[0]) for f in self.feedbacks]
        seen_idxs = set()
        for c, (cw, feedback) in enumerate(self.feedbacks):
            clusters = set()
            for doc_id in feedback:
                clusters.add(self.doc_top[doc_id])
            
            for i in range(0,len(feedback)):
                d1 = self.documents[feedback[i]]
                seen_idxs.add(i)
                full_train_data.append(InputExample(texts=[d1, *all_cws], label=c))
        unlabeled_data_v2 = [] 
        for idx in set(range(len(self.documents)))-seen_idxs:
            unlabeled_data_v2.append(InputExample(texts=[self.documents[idx], *all_cws], label=-1))
        
        print("Length of full_train_data:" + len(full_train_data))
        self.bm25_data = full_train_data 
        self.unlab_v2 = unlabeled_data_v2
        self.v2_bank = full_train_data + unlabeled_data_v2
        import random
        self.v2_dataloader = DataLoader(random.sample(self.v2_bank,min(len(self.v2_bank),200*500*config['epoch'])), shuffle=True, batch_size=200)
        # retrain_model = SentenceTransformer(model_path)
        # if (config['loss']=='sbert_type'):
        #     train_loss = Loss(retrain_model)
        # if (config['loss']=='cos'):
        #     train_loss = SemLoss(retrain_model)
        
        # # Tune the model
        # retrain_model.fit(train_objectives=[(train_dataloader, train_loss)], scheduler='constantlr', epochs=1, warmup_steps=0, optimizer_params={'lr': 2e-5})

        # curr_time = datetime.now().strftime("%Y%m%d-%H:%M:%S")
        # new_path = f"{self.workdir}/retrained-model"+curr_time
        # retrain_model.save(new_path)
        # return new_path

    def run_all(self, model_path, min_cluster_size):
        self.min_cluster_size = min_cluster_size
        self.set_pretrained_model(model_path)
        if self.config['umap']:
            self.reduce_dims()
        else:
            self.umap_embeddings_cluster = self.document_embeddings
        self.make_clusters(min_cluster_size)





