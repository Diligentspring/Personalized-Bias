from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from ultra.learning_algorithm.base_algorithm import BaseAlgorithm
import ultra.utils

import numpy as np

def cross_entropy_loss(pre, tar):
    return -(torch.log(pre) * tar + torch.log(1 - pre) * (1 - tar)).sum(dim=-1).mean()

class DenoisingNet(nn.Module):
    def __init__(self, input_vec_size, output_vec_size):
        super(DenoisingNet, self).__init__()
        self.input_vec_size = input_vec_size
        self.linear_layer = nn.Linear(input_vec_size, output_vec_size)
        self.elu_layer = nn.ELU()
        self.propensity_net = nn.Sequential(self.linear_layer, self.elu_layer)
        self.list_size = input_vec_size

    def forward(self, input_list):
        input = F.one_hot(input_list, num_classes=self.input_vec_size)
        input = input.float()
        return torch.sigmoid(self.propensity_net(input))

class Context(BaseAlgorithm):
    def __init__(self, data_set, exp_settings):
        """Create the model.

        Args:
            data_set: (Raw_data) The dataset used to build the input layer.
            exp_settings: (dictionary) The dictionary containing the model settings.
        """
        print('Build Context algorithm.')

        self.hparams = ultra.utils.hparams.HParams(
            # learning_rate=0.05,                 # Learning rate.
            learning_rate=exp_settings['ln'],
            max_gradient_norm=5.0,            # Clip gradients to this norm.
            # Set strength for L2 regularization.
            l2_loss=0.0,
            grad_strategy='ada',            # Select gradient strategy
        )
        print(exp_settings['learning_algorithm_hparams'])

        self.cuda = torch.device('cuda')
        self.is_cuda_avail = torch.cuda.is_available()
        self.writer = SummaryWriter()
        self.train_summary = {}
        self.eval_summary = {}
        self.hparams.parse(exp_settings['learning_algorithm_hparams'])
        self.exp_settings = exp_settings
        if 'selection_bias_cutoff' in self.exp_settings.keys():
            self.rank_list_size = self.exp_settings['selection_bias_cutoff']
            # self.propensity_para = [torch.tensor([0.0]) for i in range(self.rank_list_size)]

        self.propensity_model = DenoisingNet(self.rank_list_size, self.rank_list_size)
        self.ave_ranking_model = DenoisingNet(self.rank_list_size, 1)

        self.max_candidate_num = exp_settings['max_candidate_num']
        self.feature_size = data_set.feature_size

        if self.is_cuda_avail:
            self.propensity_model = self.propensity_model.to(device=self.cuda)
            self.ave_ranking_model = self.ave_ranking_model.to(device=self.cuda)
        self.letor_features_name = "letor_features"
        self.letor_features = None
        self.docid_inputs_name = []  # a list of top documents
        self.labels_name = []  # the labels for the documents (e.g., clicks)
        self.docid_inputs = []  # a list of top documents
        self.labels = []  # the labels for the documents (e.g., clicks)
        for i in range(self.max_candidate_num):
            self.docid_inputs_name.append("docid_input{0}".format(i))
            self.labels_name.append("label{0}".format(i))

        # with torch.no_grad():
        #     self.propensity = (torch.ones([1, self.rank_list_size]) * 0.9)
        #     if self.is_cuda_avail:
        #         self.propensity = self.propensity.to(device=self.cuda)

        self.learning_rate = float(self.hparams.learning_rate)
        self.global_step = 0
        # self.sigmoid_prob_b = (torch.ones([1]) - 1.0)
        # if self.is_cuda_avail:
        #     self.sigmoid_prob_b = self.sigmoid_prob_b.to(device=self.cuda)

        # Select optimizer
        self.optimizer_func = torch.optim.Adagrad
        # tf.train.AdagradOptimizer
        if self.hparams.grad_strategy == 'sgd':
            self.optimizer_func = torch.optim.SGD

    def train(self, input_feed):
        """Run a step of the model feeding the given inputs for training process.

        Args:
            input_feed: (dictionary) A dictionary containing all the input feed data.

        Returns:
            A triple consisting of the loss, outputs (None if we do backward),
            and a tf.summary containing related information about the step.

        """
        self.ave_ranking_model.train()
        self.propensity_model.train()
        self.create_input_feed(input_feed, self.rank_list_size)

        # print(self.rank_list_size)

        test_input = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int).to(device=self.cuda)
        self.backup_propensities = self.propensity_model(test_input)

        qids = input_feed['qids']
        user_ids = []
        for qid in qids:
            user_id = int(qid.strip().split(',')[1])
            user_ids.append(user_id)
        user_ids = torch.tensor(user_ids, dtype=int).to(device=self.cuda)

        propensities = self.propensity_model(user_ids)
        ave_relevance = self.ave_ranking_model(user_ids)
        ave_relevances = torch.cat([ave_relevance] * self.rank_list_size, dim=1)

        # train_output = self.ranking_model(self.model,
        #                                   self.rank_list_size)
        # train_output = torch.sigmoid(train_output)

        self.loss = cross_entropy_loss(propensities * ave_relevances, self.labels)

        opt = self.optimizer_func(self.ave_ranking_model.parameters(), self.learning_rate)
        opt.zero_grad(set_to_none=True)

        opt_propensity = self.optimizer_func(self.propensity_model.parameters(), self.learning_rate)
        opt_propensity.zero_grad(set_to_none=True)

        self.loss.backward()

        if self.loss == 0:
            for name, param in self.model.named_parameters():
                print(name, param)
        if self.hparams.max_gradient_norm > 0:
            nn.utils.clip_grad_norm_(self.propensity_model.parameters(), self.hparams.max_gradient_norm)
            nn.utils.clip_grad_norm_(self.ave_ranking_model.parameters(), self.hparams.max_gradient_norm)

        opt.step()
        opt_propensity.step()

        self.global_step += 1
        print('Loss %f at global step %d' % (self.loss, self.global_step))
        return self.loss, None, self.train_summary

    def validation(self, input_feed):
        return
