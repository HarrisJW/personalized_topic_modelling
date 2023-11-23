from oracle import *
from oraclev2 import *
from oraclev3 import *
from model import *
from sklearn.datasets import fetch_20newsgroups
import pickle
import sys
import json
import os
import pprint
import pprint
import pandas as pd


def get_dataset(dataset):

    if dataset == '20newsgroups':
        DATASET = fetch_20newsgroups(data_home=".",
                                    subset='train',
                                    remove=('headers', 'footers', 'quotes')).data
        
        GROUND_TRUTH = fetch_20newsgroups(data_home=".",
                                        subset='train',
                                        remove=('headers', 'footers',
                                                'quotes')).target
        return DATASET, GROUND_TRUTH

#Jonathan added this. Jul 1 2023
    elif dataset == 'test':

        testData = pd.read_csv('../20news_500samples.csv')

        # Running into issues with newline characters.
        # https://stackoverflow.com/questions/44227748/removing-newlines-from-messy-strings-in-pandas-dataframe-cells
        testData = testData.replace(r'\n',' ', regex=True).fillna('')

        DATASET = testData.text.tolist()

        GROUND_TRUTH = testData.target.to_numpy()

        return DATASET, GROUND_TRUTH

    elif dataset == 'wvsh':
        DATASET = fetch_20newsgroups(data_home=".",
                                    subset='train',
                                    categories=["comp.os.ms-windows.misc", "comp.sys.ibm.pc.hardware"],
                                    remove=('headers', 'footers', 'quotes')).data
        
        GROUND_TRUTH = fetch_20newsgroups(data_home=".",
                                        subset='train',
                                        categories=["comp.os.ms-windows.misc", "comp.sys.ibm.pc.hardware"],
                                        remove=('headers', 'footers',
                                                'quotes')).target
        return DATASET, GROUND_TRUTH
    elif dataset == 'nova':
        data = fetch_20newsgroups(data_home=".",
                                    subset='train',
                                    remove=('headers', 'footers', 'quotes'))
        DATASET = data.data
        GROUND_TRUTH = data.target
        for i in range(len(GROUND_TRUTH)):
            target = 1*("talk." in data.target_names[GROUND_TRUTH[i]] or ".religion." in data.target_names[GROUND_TRUTH[i]] or "atheism" in data.target_names[GROUND_TRUTH[i]])
            GROUND_TRUTH[i] = target
        return DATASET, GROUND_TRUTH
    elif dataset == 'autovsaviation':
        import re
        def remove_headers(text):
            return text
            return re.sub("((.*?):.*\n)","",text)

        def read_all(path):
            files = os.listdir(path)
            texts = []
            for f in files:
                try:
                    texts.append(remove_headers(open(os.path.join(path,f),errors='ignore').read()))
                except Exception as e:
                    print(e)
                    print(f)
            return texts

        autos = read_all("sraa/realauto")
        autos += read_all("sraa/simauto")
        aviations = read_all("sraa/simaviation")
        aviations += read_all("sraa/realaviation")

        texts = autos+aviations
        targets = [0]*len(autos)+[1]*len(aviations)
        return texts, targets
    elif dataset == 'simvsreal':
        import re
        def remove_headers(text):
            return text
            return re.sub("((.*?):.*\n)","",text)

        def read_all(path):
            files = os.listdir(path)
            texts = []
            for f in files:
                try:
                    texts.append(remove_headers(open(os.path.join(path,f),errors='ignore').read()))
                except Exception as e:
                    print(e)
                    print(f)
            return texts

        sim = read_all("sraa/simauto")
        sim += read_all("sraa/simaviation")
        real = read_all("sraa/realauto")
        real += read_all("sraa/realaviation")

        texts = sim+real
        targets = [0]*len(sim)+[1]*len(real)
        return texts, targets

    
    
def eval(config, debug=False):
    os.makedirs(config['workdir'],exist_ok=True)
    num_feedback_iters = config['num_iters']
    
    purities = []
    result = {
        'purities': [],
        'v2':None,
    }
    DATASET, GROUND_TRUTH = get_dataset(config.get('dataset','20newsgroups'))

    #Convert data to lowercase.
    for i,d in enumerate(DATASET):
        DATASET[i] = d.lower()

    global_model = None
    
    m = MLModel(DATASET, config)
    
    o1 = Oraclev1(DATASET, GROUND_TRUTH, global_model)
    o2 = Oraclev2(DATASET, GROUND_TRUTH, global_model)
    o3 = Oraclev3(DATASET, GROUND_TRUTH, global_model)

    model_path = config["model_path"]

    if 'v3' in config['oracle']:
        top2vec = m.run_all(model_path, config['min_cluster_size'])
        cluster_labels = m.make_clusters(config['min_cluster_size'])
        for cluster_id in range(config["min_cluster_size"]):
            m.apply_f2w(o3.get_bm25_docs_by_prob_word_given_topic_words(cluster_id), train=False)
        L = []
        for i in range(10):
            print(len(cluster_labels))
            print(len(GROUND_TRUTH))
            purity_stats = o2.get_purity(cluster_labels)
            a = np.array(
                sorted(map(
                    lambda k:
                    (purity_stats[1][k], purity_stats[2][k], purity_stats[3][k]),
                    purity_stats[1]),
                    key=lambda x: x[0]))
            pprint.pprint(a)
            print(a.mean(axis=0))
            L.append((a.mean(axis=0)[0],model_path))
            model_path = m.finetune(model_path)
            top2vec = m.run_all(model_path, config['min_cluster_size'])
            cluster_labels = m.make_clusters(config['min_cluster_size'])
        max_purity, model_path = max(L)
        result['v2'] = max_purity
        print(f"Choosing model_path {model_path} with purity={max_purity}")

    #apply word-level feedback through oracle 2
    if 'v2' in config['oracle']:
        top2vec = m.run_all(model_path, config['min_cluster_size'])
        cluster_labels = m.make_clusters(config['min_cluster_size'])
        for cluster_id in range(config["min_cluster_size"]):
            m.apply_f2w(o2.get_bm25_docs_by_cluster_class_words(cluster_id), train=False)
        L = []
        for i in range(10):
            print(len(cluster_labels))
            print(len(GROUND_TRUTH))
            purity_stats = o2.get_purity(cluster_labels)
            a = np.array(
                sorted(map(
                    lambda k:
                    (purity_stats[1][k], purity_stats[2][k], purity_stats[3][k]),
                    purity_stats[1]),
                    key=lambda x: x[0]))
            pprint.pprint(a)
            print(a.mean(axis=0))
            L.append((a.mean(axis=0)[0],model_path))
            model_path = m.finetune(model_path)
            top2vec = m.run_all(model_path, config['min_cluster_size'])
            cluster_labels = m.make_clusters(config['min_cluster_size'])
        max_purity, model_path = max(L)
        result['v2'] = max_purity
        print(f"Choosing model_path {model_path} with purity={max_purity}")

    #apply document-level feedback through oracle1
    if 'v1' in config['oracle']:
        i = 0
        while i < num_feedback_iters:
            # call top2vec
            # give top2vec cluster_labels to oracle
            top2vec = m.run_all(model_path, config['min_cluster_size'])
            cluster_labels = m.make_clusters(config['min_cluster_size'])
            purities.append(o2.get_purity(cluster_labels))
            a = np.array(
                sorted(map(
                    lambda k:
                    (purities[-1][1][k], purities[-1][2][k], purities[-1][3][k]),
                    purities[-1][1]),
                    key=lambda x: x[0]))
            pprint.pprint(a.tolist())
            print(a.mean(axis=0))
            if debug:
                return cluster_labels, o1, o2
            result['purities'].append(a)
            with open(f'{config["workdir"]}/result.pkl', 'wb') as f:
                pickle.dump(result, f)        
            
            if 'v2' in config['oracle']:
                feedback = o1.get_feedback_neg(cluster_labels)
                m.apply_feedbackv2v1_neg(feedback)
            else:
                feedback = o1.get_feedback(cluster_labels)
                m.apply_feedback(feedback)
            model_path = m.finetune(model_path)
            i+=1

    top2vec = m.run_all(model_path, config['min_cluster_size'])
    cluster_labels = m.make_clusters(config['min_cluster_size'])
    purities.append(o2.get_purity(cluster_labels))
    a = np.array(
        sorted(map(
            lambda k:
            (purities[-1][1][k], purities[-1][2][k], purities[-1][3][k]),
            purities[-1][1]),
               key=lambda x: x[0]))
    pprint.pprint(a.tolist())
    print(a.mean(axis=0))

if __name__=="__main__":
    config = json.load(open(sys.argv[1], 'r'))
    eval(config=config)
