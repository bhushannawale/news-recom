

from collections import defaultdict
import math
import operator


class Index(object):

    def __init__(self, docs):

        self.docs = docs
        self.tf = [doc.tf for doc in self.docs]

        self.tf_index = self.create_tf_index(self.tf)

        self.doc_freqs = self.count_doc_frequencies(self.tf_index)
        self.tfidf_list = self.create_tfidf_index(self.tf, self.doc_freqs)
        self.normal_tfidf_list = self.normalize_vector_list(self.tfidf_list)
        self.update_tfidf(self.docs, self.normal_tfidf_list)

        self.topic_doc_freqs = self.count_topic_doc_frequencies(self.docs)
        self.information_entropy = self.cal_information_entropy(self.doc_freqs, self.topic_doc_freqs)
        self.tfidfie_list = self.create_tfidfie_index(self.tfidf_list, self.information_entropy)
        self.normal_tfidfie_list = self.normalize_vector_list(self.tfidfie_list)
        self.update_tfidfie(self.docs, self.normal_tfidfie_list)


    def count_doc_frequencies(self, tf_index):
       

        doc_freqs = defaultdict(lambda: 0)

        for term, freq_list in tf_index.items():
            doc_freqs[term] = len(freq_list)

        return doc_freqs

    def count_topic_doc_frequencies(self, docs):

        topic_doc_freqs = defaultdict(lambda: defaultdict(lambda: 0))

        for doc in docs:
            topic = doc.topic

            for term in doc.tf.keys():
                topic_doc_freqs[topic][term] += 1

        return topic_doc_freqs

    def create_tf_index(self, tf):
        

        tf_index = defaultdict(lambda: list())

        for i in range(len(tf)):
            doc = tf[i]

            for term, freq in doc.items():
                tf_index[term].append([i, freq])

        return tf_index

    def create_tfidf_index(self, tf, doc_freqs):

        doc_count = len(tf)
        tfidf_list = list()

        for doc_tf in tf:
            doc_tfidf = defaultdict(lambda: 0.0)

            for term, freq in doc_tf.items():
                score = freq * math.log(doc_count * 1.0/doc_freqs[term])
                doc_tfidf[term] = score

            tfidf_list.append(doc_tfidf)

        return tfidf_list

    def create_tfidfie_index(self, tfidf_list, information_entropy):

        tfidfie_list = list()
        for item in tfidf_list:
            tfidfie_list.append(item.copy())

        for doc in tfidfie_list:
            for term in doc.keys():
                doc[term] = doc[term]/information_entropy[term]

        return tfidfie_list

    def cal_information_entropy(self, doc_freqs, topic_doc_freqs):
        information_entropy = defaultdict(lambda: 0.0)

        for term in doc_freqs.keys():
            score = 0.0

            for topic in topic_doc_freqs.keys():

                if topic_doc_freqs[topic][term] != 0:
                    temp = topic_doc_freqs[topic][term]*1.0/doc_freqs[term]
                    score -= (temp * math.log(temp))

            if score == 0:
                score = 1.0

            information_entropy[term] = score

        return information_entropy

    def normalize_vector_list(self, vector_list):

        normal_vector_list = vector_list

        for vector in vector_list:
            vector_normal = self.cosine_normalization(vector.values())

            for key in vector.keys():
                vector[key] = vector[key]/vector_normal

        return normal_vector_list

    def cosine_normalization(self, vector):
        
        return math.sqrt(sum(i**2 for i in vector))

    def update_tfidf(self, docs, tfidf_list):
        
        for i in range(len(docs)):
            docs[i].tfidf = dict(sorted(tfidf_list[i].items(), key=operator.itemgetter(1), reverse=True)[:30])

    def update_tfidfie(self, docs, tfidfie_list):
        
        for i in range(len(docs)):
            docs[i].tfidfie = dict(sorted(tfidfie_list[i].items(), key=operator.itemgetter(1), reverse=True)[:30])
