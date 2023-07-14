from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from model import *
from rank_bm25 import BM25Okapi
from prettytable import PrettyTable


class Oraclev2:
    def __init__(self, docs, labels, model, target_names = None):
        self.docs = docs
        self.labels = labels
        self.class_words = self.extract_class_words(docs, labels)
        # if target_names is not None:
        print("Printing oraclev2 class_words")
        for c in sorted(self.class_words):
            print(f"{c} {self.class_words[c]}")
        self.global_model = None
        self.model = model
        self.feedback_given = set()
    
    def identify_most_scattered(self, clusters):
        x = PrettyTable()
        labels = self.labels
        labels2clusters={}
        for i in range(0,len(clusters)):
            if (labels[i] not in labels2clusters):
                labels2clusters[labels[i]]=[]
            labels2clusters[labels[i]].append(clusters[i])
        # print(labels2clusters)
        labels2probs = {}
        c = 1
        for key in labels2clusters:
            i = sorted(labels2clusters[key])
            labels2probs[key] = []
            for j in range(0,len(i)):
                if j<len(i)-1 and i[j]==i[j+1]:
                    c+=1 
                else:
                    labels2probs[key].append(c/len(i))
                    c=1
        
        def entropy(x):
            x = np.array(x)
            return -(x*np.log(x+1e-5)).sum()
        labels2probs = {i:entropy(labels2probs[i]) for i in labels2probs}
        most_scattered_label = sorted(labels2probs, key= lambda x: -labels2probs[x])
        d = labels2clusters
        header = sorted(d)
        header = f'|{"GT/Clusters":^20s}|'+'|'.join(map(lambda x:f'{x:^6d}',header))+'|'
        print('='*len(header))
        print(header)
        print('='*len(header))
        for i in sorted(d):
            c = Counter(d[i])
            l = []
            for class_ in sorted(d):
                l.append(c[class_])
            print(f'|{i:^20d}|'+'|'.join(map(lambda x:f'{x:^6d}',l))+'|')
        print("="*len(header))
        return most_scattered_label


    def extract_class_words(self, docs, labels):
        '''

        Input: a list of documents and an ndarray of class labels for these documents.
        Output: a dictionary of the top n words most representative of documents in each class.

        (Train a logistic regression model on the multiclass classification problem of classifying
        documents based on their tf-idf representations and identify the most
        important words for identifying each class)

        '''

        # Convert to TF-IDF format
        tfidfV = TfidfVectorizer(stop_words='english') 
        X_train_tfidfV_dt = tfidfV.fit_transform(docs) # fit_transform learns the vocab and one-hot encodes        
       
        #Logistic regression train
        text_clf = LogisticRegression()#penalty='l2',solver='liblinear')
        text_clf.fit(X_train_tfidfV_dt, labels)
        top_n = 5
        if len(set(labels))==2:
            v = np.concatenate([text_clf.coef_,-text_clf.coef_])
        else:
            v = text_clf.coef_
        indices = np.argsort(-v,axis=1)[:,0:top_n:1]
        i2w = dict()
        for w in tfidfV.vocabulary_:
            i2w[tfidfV.vocabulary_[w]] = w
        class_ = 0
        c2w = dict()
        for c in indices:
            c2w[class_] = []
            for i in c:
                c2w[class_].append(i2w[i])  
            class_ += 1
        print("Printing class words")
        print(c2w)
        words = set(tfidfV.vocabulary_)    
        
        self.tokenized_corpus = []
        self.tfidf_word_document_representations={}
        tokens2class = {}
        for i in range(X_train_tfidfV_dt.shape[0]):
            full_sent = docs[i]
            tokens = tuple([w for w in full_sent.lower().split() if w in words])
            self.tfidf_word_document_representations[tokens]= i
            self.tokenized_corpus.append(tokens)
            tokens2class[tokens] = labels[i]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        return c2w 


    def get_feedback(self, clusters):
        c = self.identify_most_scattered(clusters)
        for i in c:
            if i not in self.feedback_given:
                self.feedback_given.add(i)
                c  = i 
                break 
        if type(c)==list:
            return []
        print(f"Feedback given for class {c}")
        return self.get_bm25_docs_by_cluster_class_words(c)
    
    def get_bm25_docs_by_cluster_class_words(self, c):
        # ids = []
        # for id in range(len(self.docs)):
        #     if self.labels[id]==c:
        #         ids.append(id)
        # return self.class_words[c], ids

        #https://pypi.org/project/rank-bm25/

        tokenized_query = self.class_words[c]
        N = min(50,(self.bm25.get_scores(tokenized_query)!=0).sum())
        docs = self.bm25.get_top_n(tokenized_query,self.tokenized_corpus,n=N)
        feedback = []
        for doc in range(len(docs)):
            feedback.append(self.tfidf_word_document_representations[docs[doc]])
        return tokenized_query, feedback


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


# ############check#########
# from collections import Counter
# bm25 = BM25Okapi(tokenized_corpus)
# for c in range(20):
#     print("="*30)
#     print(mydata_train.target_names[c])
#     tokenized_query = c2w[c]
#     print(tokenized_query)
#     docs = bm25.get_top_n(tokenized_query,tokenized_corpus, n=50)
#     counter = Counter([tokens2class[doc] for doc in docs])
#     print(counter[c]/sum(counter.values()))
