import numpy as np
import warnings
import nltk
import pycrfsuite
import itertools
from numpy.random import randint as rndint
from GADictionaryGroup import ga_dictionary_group
from itertools import chain
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import mutual_info_score, f1_score, classification_report
from sklearn.utils.extmath import cartesian
from tqdm import tqdm_notebook as tqdm
warnings.filterwarnings('ignore')


def findsubsets(S,m):
    return list(set(itertools.combinations(S, m)))[0:4]


class DatasetGenerator:
    def __init__(self, dictionaries={}, stop_words={}, filter_values=[]):
        self.dictionaries = dictionaries
        self.stop_words = stop_words
        self.filter_values = filter_values
        self.dic_word_frequency = {}

    def split_names_in_dictionaries(self):
        for dictionary_name in self.dictionaries:
            for i in range(len(self.dictionaries[dictionary_name])):
                for j in range(len(self.dictionaries[dictionary_name][i][1])):
                    try:
                        self.dictionaries[dictionary_name][i][1][j] = self.dictionaries[dictionary_name][i][1][
                            j].split()
                    except:
                        self.dictionaries[dictionary_name][i][1][j] = []

    def number_of_uniq_words_in_dictionary(self, splitted_dictionary):
        return len(set([token for name in splitted_dictionary for syn in name[1] for token in syn]))

    def add_stop_word(self):
        return self.stop_words['stop_word'][rndint(0, len(self.stop_words['stop_word']))]

    def precalculation(self, word, splitted_dictionary):
        return set([(ind1, ind2) for ind1, name in enumerate(splitted_dictionary)
                    for ind2, syn in enumerate(name[1]) if word in syn])

    def get_all_words(self, splitted_dictionary):
        return list(set([token for name in splitted_dictionary for syn in name[1] for token in syn]))

    def construct_dictionary_frequency(self, big_dictionary):
        list_of_all_words = []
        for name_dic in big_dictionary:
            list_of_all_words += self.get_all_words(big_dictionary[name_dic])
        for word in list_of_all_words:
            self.dic_word_frequency[word] = {}
            for name_dic in big_dictionary:
                self.dic_word_frequency[word][name_dic] = self.precalculation(word, big_dictionary[name_dic])

    def generate_rules(self, position, query, structure):
        if position == -1:
            self.generate_rules(position + 1, query, structure)
            self.generate_rules(position + 1, query + 'show ', structure + ['sw'])
            self.generate_rules(position + 1, query + 'show me ', structure + ['sw', 'sw'])
            # self.generate_rules(position + 1, query + 'show me please ', structure + ['sw', 'sw', 'sw'])
        if position == 0:
            self.generate_rules(position + 1, query + '{m1}', structure + ['m1'])
            # self.generate_rules(position + 1, query + '{m1} and {m2}', structure + ['m1', 'sw', 'm2'])
            # self.generate_rules(position + 1, query + '{m1} {m2}', structure + ['m1', 'm2'])
            # self.generate_rules(position + 1, query + '{m1} vs {m2}', structure + ['m1', 'sw', 'm2'])
            self.generate_rules(position + 3, query + '{pf1} {m1}', structure + ['pf1', 'm1'])
        elif position == 1:
            self.generate_rules(position + 1, query, structure)
            self.generate_rules(position + 1, query + ' {d1}', structure + ['d1'])
            self.generate_rules(position + 1, query + ' by {d1}', structure + ['sw', 'd1'])
            # self.generate_rules(position + 1, query + ' for {d1}', structure + ['sw', 'd1'])
            # self.generate_rules(position + 1, query + ' by {d1} and {d2}', structure + ['sw', 'd1', 'sw', 'd2'])
            # self.generate_rules(position + 1, query + ' by {d1} and by {d2}',
            #                    structure + ['sw', 'd1', 'sw', 'sw', 'd2'])
        elif position == 2:
            self.generate_rules(position + 1, query, structure)
            self.generate_rules(position + 1, query + ' {pf1}', structure + ['pf1'])
            self.generate_rules(position + 1, query + ' for {pf1}', structure + ['sw', 'pf1'])
            # self.generate_rules(position + 1, query + ' where {pf1}', structure + ['sw', 'pf1'])
        elif position == 3:
            self.generate_rules(position + 1, query, structure)
            self.generate_rules(position + 1, query + ' {f1}', structure + ['f1'])
            self.generate_rules(position + 1, query + ' for {f1}', structure + ['sw', 'f1'])
            # self.generate_rules(position + 1, query + ' for {f1} and {f2}', structure + ['sw', 'f1', 'sw', 'f2'])
            # self.generate_rules(position + 1, query + ' for {f1} and for {f2}', structure + ['sw', 'f1', 'sw', 'sw', 'f2'])
            self.generate_rules(position + 2, query + ' {fv1} {f1}', structure + ['fv1', 'f1'])
        elif position == 4 and 'f1' in structure[-1]:
            # self.generate_rules(position + 1, query + ' contains {fv1}', structure + ['sw', 'fv1'])
            # self.generate_rules(position + 1, query + ' less {fv1}', structure + ['sw', 'fv1'])
            # self.generate_rules(position + 1, query + ' greater {fv1}', structure + ['sw', 'fv1'])
            # self.generate_rules(position + 1, query + ' greater than {fv1}', structure + ['sw', 'sw', 'fv1'])
            self.generate_rules(position + 1, query + ' equal {fv1}', structure + ['sw', 'fv1'])
        else:
            self.rules.append([query, structure])

    def generate_queries_denis(self, count=5000):

        self.metrics = [token for metric in self.dictionaries['metric_dictionary'] for token in metric[1]]
        self.dimensions = [token for dimension in self.dictionaries['dimension_dictionary'] for token in dimension[1]]
        self.filters = [token for filt in self.dictionaries['filter_dictionary'] for token in filt[1]]
        self.predefined_filters = [token for predefined_filter in self.dictionaries['predefined_filter_dictionary'] for
                                   token in predefined_filter[1]]

        self.split_names_in_dictionaries()

        self.construct_dictionary_frequency({'metric': self.dictionaries['metric_dictionary'],
                                             'dimension': self.dictionaries['dimension_dictionary'],
                                             'filter': self.dictionaries['filter_dictionary'],
                                             'predefined_filter': self.dictionaries['predefined_filter_dictionary']})

        self.total_numbers = np.array(
            [self.number_of_uniq_words_in_dictionary(self.dictionaries[d_name]) for d_name in self.dictionaries])

        self.rules = []
        self.generate_rules(-1, "", [])

        self.queries, self.queries_with_tags, self.labels = [], [], []
        number_names = 30
        generated_queries = 0

        for rule in tqdm(self.rules):
            numbers1 = [('m1' in rule[1]) * number_names + ('m1' not in rule[1]),
                        ('d1' in rule[1]) * number_names + ('d1' not in rule[1]),
                        ('f1' in rule[1]) * number_names + ('f1' not in rule[1]),
                        ('pf1' in rule[1]) * number_names + ('pf1' not in rule[1]),
                        ('fv1' in rule[1]) * number_names + ('fv1' not in rule[1])]

            random_metrics = np.array(self.metrics)[rndint(0, len(self.metrics), numbers1[0])]
            random_dimens = np.array(self.dimensions)[rndint(0, len(self.dimensions), numbers1[1])]
            random_filters = np.array(self.filters)[rndint(0, len(self.filters), numbers1[2])]
            random_pred_filts = np.array(self.predefined_filters)[rndint(0, len(self.predefined_filters),
                                                                         numbers1[3])]

            filter_values = np.array(self.filter_values)[rndint(0, len(self.filter_values),
                                                                numbers1[4])]

            cartmetric = cartesian([random_metrics, random_metrics])
            cartdimension = cartesian([random_dimens, random_dimens])
            cartfilter = cartesian([random_filters, random_filters])
            cartpredfilter = cartesian([random_pred_filts, random_pred_filts])

            np.random.shuffle(cartmetric)
            np.random.shuffle(cartdimension)
            np.random.shuffle(cartfilter)
            np.random.shuffle(cartpredfilter)
            np.random.shuffle(filter_values)

            c = 3

            for met1, met2 in cartmetric[0:c]:
                for dim1, dim2 in cartdimension[0:c]:
                    for fil1, fil2 in cartfilter[0:c + 2]:
                        for pfil1, pfil2 in cartpredfilter[0:c + 2]:
                            for filv1 in filter_values[0:c]:
                                for mmet1 in findsubsets(met1.split(), rndint(1, len(met1.split()) + 1)):
                                    for mmet2 in findsubsets(met2.split(), rndint(1, len(met2.split()) + 1)):
                                        for ddim1 in findsubsets(dim1.split(), rndint(1, len(dim1.split()) + 1)):
                                            for ddim2 in findsubsets(dim2.split(), rndint(1, len(dim2.split()) + 1)):
                                                for ffil1 in findsubsets(fil1.split(),
                                                                         rndint(1, len(fil1.split()) + 1)):
                                                    for ffil2 in findsubsets(fil2.split(),
                                                                             rndint(1, len(fil2.split()) + 1)):
                                                        for ppfil1 in findsubsets(pfil1.split(),
                                                                                  rndint(1, len(pfil1.split()) + 1)):
                                                            for ppfil2 in findsubsets(pfil2.split(), rndint(1, len(
                                                                    pfil2.split()) + 1)):
                                                                label = []
                                                                for tag in rule[1]:
                                                                    if tag == 'm1':
                                                                        for i in range(len(mmet1)):
                                                                            if i == 0:
                                                                                label.append('B-metric')
                                                                            else:
                                                                                label.append('I-metric')

                                                                    if tag == 'm2':
                                                                        for i in range(len(mmet2)):
                                                                            if i == 0:
                                                                                label.append('B-metric')
                                                                            else:
                                                                                label.append('I-metric')

                                                                    if tag == 'd1':
                                                                        for i in range(len(ddim1)):
                                                                            if i == 0:
                                                                                label.append('B-dimension')
                                                                            else:
                                                                                label.append('I-dimension')

                                                                    if tag == 'd2':
                                                                        for i in range(len(ddim2)):
                                                                            if i == 0:
                                                                                label.append('B-dimension')
                                                                            else:
                                                                                label.append('I-dimension')

                                                                    if tag == 'f1':
                                                                        for i in range(len(ffil1)):
                                                                            if i == 0:
                                                                                label.append('B-filter')
                                                                            else:
                                                                                label.append('I-filter')

                                                                    if tag == 'f2':
                                                                        for i in range(len(ffil2)):
                                                                            if i == 0:
                                                                                label.append('B-filter')
                                                                            else:
                                                                                label.append('I-filter')

                                                                    if tag == 'pf1':
                                                                        for i in range(len(ppfil1)):
                                                                            if i == 0:
                                                                                label.append('B-predefined_filter')
                                                                            else:
                                                                                label.append('I-predefined_filter')

                                                                    if tag == 'pf2':
                                                                        for i in range(len(ppfil2)):
                                                                            if i == 0:
                                                                                label.append('B-predefined_filter')
                                                                            else:
                                                                                label.append('I-predefined_filter')
                                                                    if tag == 'sw':
                                                                        label.append('stop_word')

                                                                    if tag == 'fv1':
                                                                        for i in range(len(filv1.split())):
                                                                            if i == 0:
                                                                                label.append('B-filter_value')
                                                                            else:
                                                                                label.append('I-filter_value')

                                                                self.labels.append(rule[0], )
                                                                self.queries_with_tags.append(
                                                                    list(zip(rule[0].format(d1=' '.join(list(ddim1)),
                                                                                            m1=' '.join(list(mmet1)),
                                                                                            f1=' '.join(list(ffil1)),
                                                                                            pf1=' '.join(list(ppfil1)),
                                                                                            d2=' '.join(list(ddim2)),
                                                                                            m2=' '.join(list(mmet2)),
                                                                                            f2=' '.join(list(ffil2)),
                                                                                            pf2=' '.join(list(ppfil2)),
                                                                                            fv1=filv1).split(),
                                                                             label)))

        np.random.shuffle(self.queries_with_tags)
        self.queries_with_tags = self.queries_with_tags[0:count]

    def construct_name(self, word):
        return [word.split()[i] for i in
                sorted(list(set(rndint(0, len(word.split()), rndint(0, len(word.split()), 1)))))]

    def sent2features(self, sent):
        return [self.word2features(sent, i) for i in range(len(sent))]

    def sent2labels(self, sent):
        return [label for token, label in sent]

    def sent2tokens(self, sent):
        return [token for token, label in sent]

    def add_features_to_words(self, words):
        res = np.zeros(4)
        names = ['metric', 'dimension', 'filter', 'predefined_filter']
        for ind, name in enumerate(names):
            try:
                c = self.dic_word_frequency[words[0]][name]
            except:
                c = set()
            for word in words[1:]:
                try:
                    c = c.intersection(self.dic_word_frequency[word][name])
                except:
                    c = set()
            # print (c)
            res[ind] = len(c)
        return res

    def add_features_ratio(self, words):
        res = np.zeros(4)
        names = ['metric', 'dimension', 'filter', 'predefined_filter']
        for ind, name in enumerate(names):
            try:
                c = self.dic_word_frequency[words[0]][name]
            except:
                c = set()
            for word in words[1:]:
                try:
                    c = c.intersection(self.dic_word_frequency[word][name])
                except:
                    c = set()
            if len(c) == 0:
                res[ind] = 0
            else:
                res[ind] = max(
                    [len(words) / len(self.dictionaries[name + "_dictionary"][pos[0]][1][pos[1]]) for pos in c])
        return res

    def word2features(self, sent, i):
        word = str(sent[i][0])
        # dict_features_non_stop = self.add_dictionary_features_to_word_non_stop(word)
        dict_features0 = self.add_features_to_words([word])
        features = list(itertools.chain.from_iterable([
            ['bias',
             'word.isdigit=%s' % word.isdigit(),
             'from_start=' + str(i),
             'from_end=' + str(len(sent) - i),
             'prop_start=' + str(i / len(sent)),
             'sent_len=' + str(len(sent)),
             'pos_tag=' + nltk.pos_tag([word])[0][1], ],

            self.add_features_words_tag([word], '0:'),
            self.add_features_ratio_tag([word], '0:'),
            self.add_important_words_features_tag(word, '0:'),

        ]))

        if i > 0:
            word1 = str(sent[i - 1][0])
            dict_features = self.add_features_to_words([word1])
            dict_features1 = self.add_features_to_words([word, word1])
            features.extend(list(itertools.chain.from_iterable([
                ['bias',
                 '-1:word.isdigit=%s' % word1.isdigit(),
                 '-1:pos_tag=' + nltk.pos_tag([word1])[0][1],
                 '-1.mutual=' + str(mutual_info_score(dict_features0, dict_features)), ],

                self.add_features_words_tag([word], '-1:'),
                self.add_features_words_tag([word, word1], '-1.1:'),
                self.add_features_ratio_tag([word, word1], '-1.2:'),
                self.add_important_words_features_tag(word1, '-1:'),

            ])))
        else:
            features.append('BOS')

        if i > 1 and i < len(sent):
            word1 = str(sent[i - 2][0])
            dict_features1 = self.add_features_to_words([word, word1])
            features.extend(list(itertools.chain.from_iterable([
                ['bias',
                 '-2:word.isdigit=%s' % word1.isdigit(),
                 '-2:pos_tag=' + nltk.pos_tag([word1])[0][1],
                 '-2.mutual=' + str(mutual_info_score(dict_features0, dict_features)), ],

                self.add_features_words_tag([word, word1], '-2:'),
                self.add_features_ratio_tag([word, word1], '-2:'),
                self.add_important_words_features_tag(word1, '-2:'),

            ])))
        else:
            features.append('BOS')

        if i < len(sent) - 1:
            word1 = str(sent[i + 1][0])
            dict_features = self.add_features_to_words([word1])
            dict_features1 = self.add_features_to_words([word, word1])
            features.extend(list(itertools.chain.from_iterable([

                ['bias',
                 '+1:word.isdigit=%s' % word1.isdigit(),
                 '+1:pos_tag=' + nltk.pos_tag([word1])[0][1],
                 '+1.mutual=' + str(mutual_info_score(dict_features0, dict_features)), ],

                self.add_features_words_tag([word], '+1:'),
                self.add_features_words_tag([word, word1], '+1.1:'),
                self.add_features_ratio_tag([word, word1], '+1.2:'),
                self.add_important_words_features_tag(word1, '+1:'),

            ])))
        else:
            features.append('BOS')

        if i < len(sent) - 2:
            word1 = str(sent[i + 2][0])
            dict_features = self.add_features_to_words([word1])
            dict_features1 = self.add_features_to_words([word, word1])
            features.extend(list(itertools.chain.from_iterable([

                ['bias',
                 '+2:word.isdigit=%s' % word1.isdigit(),
                 '+2:pos_tag=' + nltk.pos_tag([word1])[0][1],
                 '+2.mutual=' + str(mutual_info_score(dict_features0, dict_features)), ],

                self.add_features_words_tag([word, word1], '+2:'),
                self.add_features_ratio_tag([word, word1], '+2:'),
                self.add_important_words_features_tag(word1, '+2:'),

            ])))
        else:
            features.append('BOS')

        if i > 0 and i < len(sent) - 1:
            word = str(sent[i][0])
            word1 = str(sent[i - 1][0])
            word2 = str(sent[i + 1][0])
            dict_features = self.add_features_to_words([word, word1, word2])
            ratio_features = self.add_features_ratio([word, word1, word2])
            features.extend(list(itertools.chain.from_iterable([
                ['bias',
                 '+1-1.0=' + str(dict_features[0]),
                 '+1-1.1=' + str(dict_features[1]),
                 '+1-1.2=' + str(dict_features[2]),
                 '+1-1.3=' + str(dict_features[3]),
                 '+1-1.r0=' + str(ratio_features[0]),
                 '+1-1.r1=' + str(ratio_features[1]),
                 '+1-1.r2=' + str(ratio_features[2]),
                 '+1-1.r3=' + str(ratio_features[3])],
            ])))

        return features

    def sent2features(self, sent):
        return [self.word2features(sent, i) for i in range(len(sent))]

    def sent2labels(self, sent):
        return [label for token, label in sent]

    def sent2tokens(self, sent):
        return [token for token, label in sent]

    def train_test(self, split=0.5):
        self.X_train = [self.sent2features(s) for s in
                        tqdm(self.queries_with_tags[0:int(len(self.queries_with_tags) * split)])]
        self.y_train = [self.sent2labels(s) for s in self.queries_with_tags[0:int(len(self.queries_with_tags) * split)]]
        self.X_test = [self.sent2features(s) for s in
                       tqdm(self.queries_with_tags[int(len(self.queries_with_tags) * split) + 1:])]
        self.y_test = [self.sent2labels(s) for s in
                       self.queries_with_tags[int(len(self.queries_with_tags) * split) + 1:]]

    def add_features_to_query(self, query):

        # self.split_names_in_dictionaries()

        q = [[token] for token in query.split()]
        self.total_numbers = np.array(
            [self.number_of_uniq_words_in_dictionary(self.dictionaries[d_name]) for d_name in self.dictionaries])
        return [self.sent2features(q)]

    def add_custom_query(self, words, tokens):
        DG.queries_with_tags.append(list(zip(words, tokens)))

    def add_important_words_features_tag(self, word, tag='0:'):
        list_of_words = ['by', 'for', 'where', 'with', 'contains', 'greater', 'less', 'equal']
        return [tag + w + '=' + str(word == w) for w in list_of_words]

    def add_features_words_tag(self, words, tag='0:'):
        return [tag + str(ind) + '=' + str(f) for ind, f in enumerate(self.add_features_to_words(words))]

    def add_features_ratio_tag(self, words, tag='0:'):
        return [tag + str(ind) + '=' + str(f) for ind, f in enumerate(self.add_features_ratio(words))]



class NamedEntityRecognition:
    def __init__(self, dictionaries={}, stop_words=[]):
        self.metric_dictionary = dictionaries['metric_dictionary']
        self.dimension_dictionary = dictionaries['dimension_dictionary']
        self.filter_dictionary = dictionaries['filter_dictionary']
        self.predefined_filter_dictionary = dictionaries['predefined_filter_dictionary']
        self.stop_words = stop_words

    def fit(self, X_train, y_train):
        self.trainer = pycrfsuite.Trainer(verbose=False)
        self.trainer.select('lbfgs')

        for xseq, yseq in zip(X_train, y_train):
            self.trainer.append(xseq, yseq)

        self.trainer.set_params({
            'c1': 1.5,  # coefficient for L1 penalty
            'c2': 0.5,  # coefficient for L2 penalty
            'max_iterations': 700,  # stop earlier
            # include transitions that are possible, but not observed
            'feature.possible_transitions': True,
            # 'epsilon': 0.5,
        })

        self.trainer.train('conll2002-esp-best.crfsuite')

    def predict(self, X_test):
        self.tagger = pycrfsuite.Tagger()
        self.tagger.open('conll2002-esp-best.crfsuite')
        self.y_pred = [self.tagger.tag(xseq) for xseq in X_test]
        return self.y_pred

    def bio_classification_report(self, y_true, y_pred):
        """
        Classification report for a list of BIO-encoded sequences.
        It computes token-level metrics and discards "O" labels.

        Note that it requires scikit-learn 0.15+ (or a version from github master)
        to calculate averages properly!
        """
        lb = LabelBinarizer()
        y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
        y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))

        tagset = set(lb.classes_) - {'O'}
        tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
        class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}

        return classification_report(
            y_true_combined,
            y_pred_combined,
            labels=[class_indices[cls] for cls in tagset],
            target_names=tagset,
        )


if __name__ == "__main__":
    stop_words = {'start': ['show me please', 'show me', 'show', 'how many', 'where'],
                  'metric': {'before': [], 'after': []},
                  'dimension': {'before': ['by'], 'after': ['of']},
                  'filter': {'before': [], 'after': []},
                  'predefined_filter': {'before': ['for', 'where', 'with'], 'after': []},
                  'filter_value': {'before': [], 'after': []},
                  'stop_word': ['hey', 'and', 'if', 'well', 'yes', 'no', 'good', 'great', 'fuck'],
                  }

    filter_values = ['russia', 'statsbot.co', '5', '100', '1000', '10000', '100000', 'germany', 'moscow', 'google',
                     'yandex', 'yandex.ru',
                     'europe', 'usa', 'japan']

    ga_groups = [[ga_dictionary_group]]
    metric_dictionary = ga_groups[0][0].original_metric_dictionary
    dimension_dictionary = ga_groups[0][0].original_dimension_dictionary
    filter_dictionary = ga_groups[0][0].original_filter_dictionary
    predefined_filter_dictionary = ga_groups[0][0].original_predefined_filter_dictionary

    dictionaries = {'metric_dictionary': metric_dictionary,
                    'dimension_dictionary': dimension_dictionary,
                    'filter_dictionary': filter_dictionary,
                    'predefined_filter_dictionary': predefined_filter_dictionary}

    DG = DatasetGenerator(dictionaries=dictionaries,
                          stop_words=stop_words)

    DG.split_names_in_dictionaries()
    DG.construct_dictionary_frequency({'metric': DG.dictionaries['metric_dictionary'],
                                       'dimension': DG.dictionaries['dimension_dictionary'],
                                       'filter': DG.dictionaries['filter_dictionary'],
                                       'predefined_filter': DG.dictionaries['predefined_filter_dictionary']})

    NER = NamedEntityRecognition(dictionaries=dictionaries, stop_words=stop_words)

    while (True):
        query = input('Enter query: ')
        query = query.strip()
        res = DG.add_features_to_query(query)
        print(NER.predict(res))
