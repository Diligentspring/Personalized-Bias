import numpy as np
import json
import random
import os
from . import metrics


class Raw_data:
    def __init__(self, data_path=None, file_prefix=None, click_model_dir=None, rank_cut=None, train_dataset=None):
        """
        Initialize a dataset

        Args:
            data_path: (string) the root directory of the experimental dataset.
            file_prefix: (string) the prefix of the data to process, e.g. 'train', 'valid', or 'test'.
            rank_cut: (int) the maximum number of top documents considered in each list.

        Returns:
            None
        """
        self.data_path = data_path
        self.file_prefix = file_prefix
        self.click_model_dir = click_model_dir
        self.feature_size = -1
        self.rank_list_size = -1
        self.removed_feature_ids = []
        #self.features = []
        self.features = {}
        self.dids = []
        self.initial_list = []
        self.qids = []
        self.labels = []
        self.initial_scores = []
        self.initial_list_lengths = []

        if data_path is None:
            return

        # if os.path.isfile(data_path + file_prefix + '/' +
        #                   file_prefix + '.feature'):  # files in ULTRA data format
        #     if click_model_dir != None:
        #         self.load_data_in_ULTRE_format(data_path, file_prefix, click_model_dir, rank_cut)
        #     else:
        #         self.load_data_in_ULTRA_format(data_path, file_prefix, rank_cut)
        # # files in libsvm data format
        # elif os.path.isfile(data_path + file_prefix + '/' + file_prefix + '.txt'):
        #     self.load_data_in_libsvm_format(data_path, file_prefix, rank_cut)

        # self.load_data_in_IPS_format(data_path, file_prefix, rank_cut)
        self.load_data_in_user_bias_format(data_path, file_prefix, rank_cut, train_dataset)

        print("Finished reading %d queries with lists." % len(self.qids))

        assert self.feature_size > 0, "No valid feature has been found."
        assert len(self.qids) > 0, "No valid query has been found."
        assert len(self.dids) > 0, "No valid doc has been found."

        return

    def load_basic_data_information(
            self, data_path=None, file_prefix=None, rank_cut=None):
        """
        Load basic dataset information from data_path including:
            feature_size: the number of features for each query-document pair.
            removed_feature_ids: the idxs of the features to ignore.

        Args:
            data_path: (string) the root directory of the experimental dataset.

        Returns:
            None
        """
        settings = json.load(open(data_path + 'settings.json'))
        self.feature_size = settings['feature_size']
        if 'removed_feature_ids' in settings:
            self.removed_feature_ids = sorted(
                settings['removed_feature_ids'], reverse=True)
            for i in range(len(self.removed_feature_ids)):
                if self.removed_feature_ids[len(
                        self.removed_feature_ids) - 1 - i] > self.feature_size:
                    del self.removed_feature_ids[len(
                        self.removed_feature_ids) - 1 - i]
            print('Remove feature ids: ' + str(self.removed_feature_ids))
        metrics.RankingMetricKey.MAX_LABEL = settings['max_label']
        return

    # def load_data_in_IPS_format(self, data_path=None, file_prefix=None, rank_cut=None):
    #     """
    #             Read dataset in ULTRA format including:
    #                 rank_list_size: the maximum number of documents for a query in the data.
    #                 features: the feature vectors of each query-document pair.
    #                 dids: the doc ids for each query-document pair.
    #                 initial_list: the initial ranking list for each query
    #                 qids: the query ids for each query.
    #                 labels: the relevance label for each query-document pair in the initial_list.
    #                 initial_scores: (if exists) the initial ranking scores in the initial list for each query-document pair.
    #                 initial_list_lengths: the length of the initial list for each query.
    #
    #             Args:
    #                 data_path: (string) the root directory of the experimental dataset.
    #                 file_prefix: (string) the prefix of the data to process, e.g. 'train', 'valid', or 'test'.
    #                 rank_cut: (int) the maximum number of top documents considered in each list.
    #
    #             Returns:
    #                 None
    #             """
    #     print(
    #         'Read data from %s/%s in IPS format.' %
    #         (data_path, file_prefix))
    #     self.load_basic_data_information(data_path)
    #
    #     if file_prefix =='valid':
    #         feature_fin = open(data_path + 'valid/valid.feature')
    #     elif file_prefix =='test':
    #         feature_fin = open(data_path + 'test/test.feature')
    #     elif file_prefix =='test-on-train':
    #         feature_fin = open(data_path + 'test-on-train/test.feature')
    #     elif file_prefix =='training_valid':
    #         feature_fin = open(data_path + 'training_valid/train.feature')
    #     elif file_prefix =='training_test':
    #         feature_fin = open(data_path + 'training_test/train.feature')
    #     else:
    #         feature_fin = open(
    #             data_path +
    #             'train' +
    #             '/' +
    #             'train' +
    #             '.feature')
    #     for line in feature_fin:
    #         arr = line.strip().split(' ')
    #         self.dids.append(arr[0])
    #         # self.features.append(np.zeros(self.feature_size))
    #         self.features[arr[0]] = [0.0 for _ in range(self.feature_size)]
    #         for x in arr[1:]:
    #             if x != '':
    #                 arr2 = x.split(':')
    #                 feautre_idx = int(arr2[0])
    #                 if feautre_idx < self.feature_size:
    #                     if float(arr2[1]) < 1.7e+308:
    #                         self.features[arr[0]][int(feautre_idx)] = float(arr2[1])
    #         for rf_idx in self.removed_feature_ids:
    #             del self.features[arr[0]][rf_idx - 1]
    #     self.feature_size -= len(self.removed_feature_ids)
    #     feature_fin.close()
    #
    #     print('Feature reading finish.')
    #     # fo = open(data_path + 'feature.txt', 'w')
    #     # for key in self.features.keys():
    #     #     fo.write(str(key)+':'+str(self.features[key]))
    #     # fo.close()
    #
    #     if file_prefix =='valid':
    #         init_list_fin = open(data_path + 'valid/valid.init_list')
    #     elif file_prefix =='test':
    #         init_list_fin = open(data_path + 'test/test.init_list')
    #     elif file_prefix =='test-on-train':
    #         init_list_fin = open(data_path + 'test-on-train/test.init_list')
    #     elif file_prefix =='training_valid':
    #         init_list_fin = open(data_path + 'training_valid/training_valid.init_list')
    #     elif file_prefix =='training_test':
    #         init_list_fin = open(data_path + 'training_test/training_test.init_list')
    #     else:
    #         init_list_fin = open(
    #             data_path +
    #             'train' +
    #             '/' +
    #             'train' +
    #             '.init_list')
    #     for line in init_list_fin:
    #         arr = line.strip().split(':')
    #         self.qids.append(arr[0])
    #         if rank_cut:
    #             self.initial_list.append([x for x in arr[1].strip().split(' ')[:rank_cut]])
    #         else:
    #             self.initial_list.append([x for x in arr[1].strip().split(' ')])
    #         if len(self.initial_list[-1]) > self.rank_list_size:
    #             self.rank_list_size = len(self.initial_list[-1])
    #     init_list_fin.close()
    #
    #     print('List reading finish.')
    #     # fo = open(data_path + 'initial_list.txt', 'w')
    #     # for list in self.initial_list:
    #     #     fo.write(str(list))
    #     # fo.close()
    #
    #     if file_prefix =='valid':
    #         label_fin = open(data_path + 'valid/valid.labels')
    #     elif file_prefix =='test':
    #         label_fin = open(data_path + 'test/test.labels')
    #         #label_fin = open(data_path + 'test/test.init_list')
    #     elif file_prefix =='test-on-train':
    #         label_fin = open(data_path + 'test-on-train/test.labels')
    #     elif file_prefix =='training_valid':
    #         label_fin = open(data_path + 'training_valid/training_valid.labels')
    #     elif file_prefix =='training_test':
    #         label_fin = open(data_path + 'training_test/training_test.labels')
    #     else:
    #         label_fin = open(
    #             data_path +
    #             'train' +
    #             '/' +
    #             file_prefix +
    #             '.labels')
    #     for line in label_fin:
    #         self.labels.append(
    #             [float(x) for x in line.strip().split(':')[1].strip().split(' ')[:self.rank_list_size]])
    #     label_fin.close()
    #
    #     if os.path.isfile(data_path + file_prefix + '/' +
    #                       file_prefix + '.intial_scores'):
    #         with open(data_path + file_prefix + '/' + file_prefix + '.initial_scores') as fin:
    #             for line in fin:
    #                 self.initial_scores.append(
    #                     [float(x) for x in line.strip().split(' ')[1:]])
    #     print('Label reading finish.')
    #     # fo = open(data_path + 'labels.txt', 'w')
    #     # fo.write(str(self.labels))
    #     # fo.close()
    #     self.initial_list_lengths = [
    #         len(self.initial_list[i]) for i in range(len(self.initial_list))]
    #     self.remove_invalid_data()
    #     print('Data reading finish!')
    #     return

    def load_data_in_user_bias_format(self, data_path=None, file_prefix=None, rank_cut=None, train_dataset = None):
        """
                    rank_list_size: the maximum number of documents for a query in the data.
                    features: the feature vectors of each query-document pair.
                    dids: the doc ids for each query-document pair.
                    initial_list: the initial ranking list for each query
                    qids: the query ids for each query.
                    labels: the relevance label for each query-document pair in the initial_list.
                    initial_scores: (if exists) the initial ranking scores in the initial list for each query-document pair.
                    initial_list_lengths: the length of the initial list for each query.

                Args:
                    data_path: (string) the root directory of the experimental dataset.
                    file_prefix: (string) the prefix of the data to process, e.g. 'train', 'valid', or 'test'.
                    rank_cut: (int) the maximum number of top documents considered in each list.

                Returns:
                    None
                """
        print(
            'Read data from %s/%s in user_bias format.' %
            (data_path, file_prefix))
        self.load_basic_data_information(data_path)

        if file_prefix == 'valid':
            feature_fin = open(data_path + 'valid/valid.feature')
            # feature_fin = open(data_path + 'valid' +train_dataset.strip('train') + '/valid.feature')
        elif file_prefix == 'test':
            feature_fin = open(data_path + 'test/test.feature')
        elif file_prefix.strip().split('_')[0] == 'valid':
            feature_fin = open(data_path + 'valid' + train_dataset.strip('train') + '/valid.feature')
        else:
            if train_dataset != None:
                feature_fin = open(data_path + train_dataset + '/train.feature')
            else:
                feature_fin = open(data_path + 'train/train.feature')
        for line in feature_fin:
            arr = line.strip().split(' ')
            self.dids.append(arr[0])
            # self.features.append(np.zeros(self.feature_size))
            self.features[arr[0]] = [0.0 for _ in range(self.feature_size)]
            for x in arr[1:]:
                if x != '':
                    arr2 = x.split(':')
                    feautre_idx = int(arr2[0])
                    if feautre_idx < self.feature_size:
                        if float(arr2[1]) < 1.7e+308:
                            self.features[arr[0]][int(feautre_idx)] = float(arr2[1])
            for rf_idx in self.removed_feature_ids:
                del self.features[arr[0]][rf_idx - 1]
        self.feature_size -= len(self.removed_feature_ids)
        feature_fin.close()
        print('Feature reading finish.')

        if file_prefix == 'valid':
            init_list_fin = open(data_path + 'valid/valid.init_list')
            # init_list_fin = open(data_path + 'valid' + train_dataset.strip('train') + '/valid.init_list')
        elif file_prefix == 'test':
            init_list_fin = open(data_path + 'test/test.init_list')
        elif file_prefix.strip().split('_')[0] == 'valid':
            init_list_fin = open(data_path + 'valid' + train_dataset.strip('train') + '/valid.init_list')
        else:
            if train_dataset != None:
                init_list_fin = open(data_path + train_dataset + '/train.init_list')
            else:
                init_list_fin = open(data_path +'train/train.init_list')
        for line in init_list_fin:
            arr = line.strip().split(':')
            self.qids.append(arr[0])
            if rank_cut:
                self.initial_list.append([x for x in arr[1].strip().split(' ')[:rank_cut]])
            else:
                self.initial_list.append([x for x in arr[1].strip().split(' ')])
            if len(self.initial_list[-1]) > self.rank_list_size:
                self.rank_list_size = len(self.initial_list[-1])
        init_list_fin.close()
        print('List reading finish.')

        if file_prefix == 'valid':
            label_fin = open(data_path + 'valid/valid.labels')
            # label_fin = open(data_path + 'valid' + train_dataset.strip('train') + '/valid_{}.labels'.format(file_prefix))
        elif file_prefix == 'test':
            label_fin = open(data_path + 'test/test.labels')
        elif file_prefix.strip().split('_')[0] == 'valid':
            label_fin = open(
                data_path + 'valid' + train_dataset.strip('train') + '/{}.labels'.format(file_prefix))
        else:
            if train_dataset != None:
                label_fin = open(data_path + train_dataset + '/' + file_prefix + '.labels')
            else:
                label_fin = open(data_path  + 'train/' + file_prefix + '.labels')
        for line in label_fin:
            self.labels.append(
                [float(x) for x in line.strip().split(':')[1].strip().split(' ')[:self.rank_list_size]])
        label_fin.close()

        # if os.path.isfile(data_path + file_prefix + '/' +
        #                   file_prefix + '.intial_scores'):
        #     with open(data_path + file_prefix + '/' + file_prefix + '.initial_scores') as fin:
        #         for line in fin:
        #             self.initial_scores.append(
        #                 [float(x) for x in line.strip().split(' ')[1:]])
        print('Label reading finish.')
        self.initial_list_lengths = [
            len(self.initial_list[i]) for i in range(len(self.initial_list))]
        self.remove_invalid_data()
        print('Data reading finish!')
        return


    def remove_invalid_data(self):
        """
        Remove query lists with no relevant items or less than 2 items

        self.feature_size = -1
        self.rank_list_size = -1
        self.removed_feature_ids = []
        self.features = []
        self.dids = []
        self.initial_list = []
        self.qids = []
        self.labels = []
        self.initial_scores = []
        self.initial_list_lengths = []

        Returns:
            None
        """

        # Find invalid queries and documents
        new_qids = []
        new_initial_list = []
        new_labels = []

        for i in range(len(self.qids)):
            if len(self.initial_list[i]) < 2 or sum(self.labels[i]) <= 0:
                continue
            else:
                new_qids.append(self.qids[i])
                new_initial_list.append(self.initial_list[i])
                new_labels.append(self.labels[i])
        self.qids = new_qids[:]
        self.initial_list = new_initial_list[:]
        self.labels = new_labels[:]



        # invalid_qidx = []
        # for i in range(len(self.qids)):
        #     qidx = len(self.qids) - 1 - i
        #     if len(self.initial_list[qidx]) < 2 or sum(self.labels[qidx]) <= 0:
        #         invalid_qidx.append(qidx)
        # print('Remove %d invalid queries.' % len(invalid_qidx))

        ''' need to maintain the features and dids to avoid wrong index.
        invalid_didx = []
        for qidx in invalid_qidx:
            for idx in self.initial_list[qidx]:
                if idx >= 0:
                    invalid_didx.append(idx)
        invalid_didx = sorted(invalid_didx, reverse=True)
        '''

        # Remove invalid queries and documents
        # for qidx in invalid_qidx:
            # del self.qids[qidx]
            # del self.initial_list[qidx]
            # del self.labels[qidx]
            # if len(self.initial_scores) > 0:
            #     del self.initial_scores[qidx]

        # for didx in invalid_didx:
        #    del self.features[didx]
        #    del self.dids[didx]

        # Recompute list lengths and maximum rank list size
        self.initial_list_lengths = [
            len(self.initial_list[i]) for i in range(len(self.initial_list))]
        for i in range(len(self.initial_list_lengths)):
            x = self.initial_list_lengths[i]
            if self.rank_list_size < x:
                self.rank_list_size = x
        return

    def remove_invalid_ULTRE_data(self):
        """
        Remove query lists with no relevant items or less than 2 items

        self.feature_size = -1
        self.rank_list_size = -1
        self.removed_feature_ids = []
        self.features = []
        self.dids = []
        self.initial_list = []
        self.qids = []
        self.initial_scores = []
        self.initial_list_lengths = []

        Returns:
            None
        """

        # Find invalid queries and documents
        invalid_qidx = []
        for i in range(len(self.qids)):
            qidx = len(self.qids) - 1 - i
            if len(self.initial_list[qidx]) < 2:
                invalid_qidx.append(qidx)
        print('Remove %d invalid queries.' % len(invalid_qidx))

        ''' need to maintain the features and dids to avoid wrong index.
        invalid_didx = []
        for qidx in invalid_qidx:
            for idx in self.initial_list[qidx]:
                if idx >= 0:
                    invalid_didx.append(idx)
        invalid_didx = sorted(invalid_didx, reverse=True)
        '''

        # Remove invalid queries and documents
        for qidx in invalid_qidx:
            del self.qids[qidx]
            del self.initial_list[qidx]
            del self.labels[qidx]
            if len(self.initial_scores) > 0:
                del self.initial_scores[qidx]

        # Recompute list lengths and maximum rank list size
        self.initial_list_lengths = [
            len(self.initial_list[i]) for i in range(len(self.initial_list))]
        for i in range(len(self.initial_list_lengths)):
            x = self.initial_list_lengths[i]
            if self.rank_list_size < x:
                self.rank_list_size = x
        return

    def pad(self, rank_list_size, pad_tails=True):
        """
        Pad a rank list with zero feature vectors when it is shorter than the required rank list size.

        Args:
            rank_list_size: (int) the required size of a ranked list
            pad_tails: (bool) Add padding vectors to the tails of each list (True) or the heads of each list (False)

        Returns:
            None
        """
        self.rank_list_size = rank_list_size
        # self.features.append(
        #     [0.0 for _ in range(self.feature_size)])  # vector for pad

        self.features[-1] = [0.0 for _ in range(self.feature_size)]  # vector for pad

        for i in range(len(self.initial_list)):
            if len(self.initial_list[i]) < self.rank_list_size:
                if pad_tails:  # pad tails
                    self.initial_list[i] += ['-1'] * \
                        (self.rank_list_size - len(self.initial_list[i]))
                else:    # pad heads
                    self.initial_list[i] = ['-1'] * (self.rank_list_size - len(
                        self.initial_list[i])) + self.initial_list[i]


def merge_Summary(summary_list, weights):
    merged_values = {}
    weight_sum_map = {}
    for i in range(len(summary_list)):
        summary = summary_list[i]
        for metric in summary.keys():
            if metric not in merged_values:
                merged_values[metric] = 0.0
                weight_sum_map[metric] = 0.0
            merged_values[metric] += summary[metric] * weights[i]
            weight_sum_map[metric] += weights[i]
    for k in merged_values:
        merged_values[k] /= max(0.0000001, weight_sum_map[k])
    return merged_values


def read_data(data_path, file_prefix, click_model_dir, rank_cut=None, train_dataset=None):
    data = Raw_data(data_path, file_prefix, click_model_dir, rank_cut, train_dataset)
    return data


def generate_ranklist(data, rerank_lists):
    """
        Create a reranked lists based on the data and rerank documents ids.

        Args:
            data: (Raw_data) the dataset that contains the raw data
            rerank_lists: (list<list<int>>) a list of rerank list in which each
                            element represents the original rank of the documents
                            in the initial list.

        Returns:
            qid_list_map: (map<list<int>>) a map of qid with the reranked document id list.
    """
    if len(rerank_lists) != len(data.initial_list):
        raise ValueError("The number of queries in rerank ranklists number must be equal to the initial list,"
                         " %d != %d." % (len(rerank_lists)), len(data.initial_list))
    qid_list_map = {}
    for i in range(len(data.qids)):
        if len(rerank_lists[i]) != len(data.initial_list[i]):
            raise ValueError("The number of docs in each rerank ranklists must be equal to the initial list,"
                             " %d != %d." % (len(rerank_lists[i]), len(data.initial_list[i])))
        # remove duplicated docs and organize rerank list
        index_list = []
        index_set = set()
        for idx in rerank_lists[i]:
            if idx not in index_set:
                index_set.add(idx)
                index_list.append(idx)
        # doc idxs that haven't been observed in the rerank list will be put at
        # the end of the list
        for idx in range(len(rerank_lists[i])):
            if idx not in index_set:
                index_list.append(idx)
        # get new ranking list
        qid = data.qids[i]
        did_list = []
        new_list = [data.initial_list[i][idx] for idx in index_list]
        # remove padding documents
        for ni in new_list:
            if ni >= 0:
                did_list.append(data.dids[ni])
        qid_list_map[qid] = did_list
    return qid_list_map


def generate_ranklist_by_scores(data, rerank_scores):
    """
        Create a reranked lists based on the data and rerank scores.

        Args:
            data: (Raw_data) the dataset that contains the raw data
            rerank_scores: (list<list<float>>) a list of rerank list in which each
                            element represents the reranking scores for the documents
                            on that position in the initial list.

        Returns:
            qid_list_map: (map<list<int>>) a map of qid with the reranked document id list.
    """
    if len(rerank_scores) != len(data.initial_list):
        raise ValueError("Rerank ranklists number must be equal to the initial list,"
                         " %d != %d." % (len(rerank_scores)), len(data.initial_list))
    qid_list_map = {}
    for i in range(len(data.qids)):
        scores = rerank_scores[i]
        rerank_list = sorted(
            range(
                len(scores)),
            key=lambda k: scores[k],
            reverse=True)
        if len(rerank_list) != len(data.initial_list[i]):
            raise ValueError("Rerank ranklists length must be equal to the gold list,"
                             " %d != %d." % (len(rerank_scores[i]), len(data.initial_list[i])))
        # remove duplicate and organize rerank list
        index_list = []
        index_set = set()
        for idx in rerank_list:
            if idx not in index_set:
                index_set.add(idx)
                index_list.append(idx)
        # doc idxs that haven't been observed in the rerank list will be put at
        # the end of the list
        for idx in range(len(rerank_list)):
            if idx not in index_set:
                index_list.append(idx)
        # get new ranking list
        qid = data.qids[i]
        did_list = []
        # remove padding documents
        for idx in index_list:
            ni = data.initial_list[i][idx]
            ns = scores[idx]
            if ni >= 0:
                did_list.append((data.dids[ni], ns))
        qid_list_map[qid] = did_list
    return qid_list_map


def output_ranklist(data, rerank_scores, output_path, file_name='test'):
    """
        Create a trec format rank list by reranking the initial list with reranking scores.

        Args:
            data: (Raw_data) the dataset that contains the raw data
            rerank_scores: (list<list<float>>) a list of rerank list in which each
                            element represents the reranking scores for the documents
                            on that position in the initial list.
            output_path: (string) the path for the output
            file_name: (string) the name of the output set, e.g., 'train', 'valid', 'text'.

        Returns:
            None
    """
    qid_list_map = generate_ranklist_by_scores(data, rerank_scores)
    fout = open(output_path + file_name + '.ranklist', 'w')
    for qid in data.qids:
        for i in range(len(qid_list_map[qid])):
            fout.write(qid + ' Q0 ' + qid_list_map[qid][i][0] + ' ' + str(i + 1)
                       + ' ' + str(qid_list_map[qid][i][1]) + ' Model\n')
    fout.close()
