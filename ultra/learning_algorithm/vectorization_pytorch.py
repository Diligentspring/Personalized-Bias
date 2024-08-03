"""Training and testing the Vectorization for unbiased learning to rank.

See the following paper for more information on the Vectorization.

    * Mouxiang Chen, Chenghao Liu, Zemin Liu, Jianling Sun. 2022. Scalar is Not Enough: Vectorization-based Unbiased Learning to Rank. In Proceedings of SIGKDD '22.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import zip

import ultra.utils
from ultra.learning_algorithm.base_algorithm import BaseAlgorithm

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import numpy as np

# class DenoisingNet(nn.Module):
#     def __init__(self, input_vec_size, dimension):
#         super(DenoisingNet, self).__init__()
#         self.linear_layer_0 = nn.Linear(input_vec_size, 10)
#         self.linear_layer = nn.Linear(10, dimension)
#         self.elu_layer_0 = nn.ELU()
#         self.elu_layer = nn.ELU()
#         self.propensity_net = nn.Sequential(self.linear_layer_0, self.elu_layer_0, self.linear_layer, self.elu_layer)
#         self.list_size = input_vec_size
#
#     def forward(self, input_list):
#         # input_list   j是list_size个batch_size * feature_size
#         output_propensity_list = []
#         for i in range(self.list_size):
#             # Add position information (one-hot vector)
#             click_feature = [
#                 torch.unsqueeze(
#                     torch.zeros_like(
#                         input_list[i]), -1) for _ in range(self.list_size)]
#             click_feature[i] = torch.unsqueeze(
#                 torch.ones_like(input_list[i]), -1)
#             # Predict propensity with a simple network
#             # print(torch.cat(click_feature, 1).shape) 256*10
#             output_propensity_list.append(
#                 self.propensity_net(
#                     torch.cat(
#                         click_feature, 1))) #list, 10个256*dimension
#         return output_propensity_list

class PropensityModel(nn.Module):
    def __init__(self, list_size, dimension):
        super(PropensityModel, self).__init__()
        self._propensity_model = nn.Parameter(torch.ones(1, list_size, dimension))

    def forward(self, relevance):
        batch_size = relevance.shape[0]
        return self._propensity_model.repeat(batch_size, 1, 1)  # (B, T, d)

class DNN_density(nn.Module):
    def __init__(self, dimension, feature_size):
        super(DNN_density, self).__init__()
        self.dimension = dimension
        self.linear_layer_0 = nn.Linear(feature_size, 256)
        self.linear_layer_1 = nn.Linear(256, 64)
        self.linear_layer = nn.Linear(64, dimension * 2)
        self.elu_layer_0 = nn.ELU()
        self.elu_layer_1 = nn.ELU()
        self.elu_layer = nn.ELU()
        self.propensity_net = nn.Sequential(self.linear_layer_0, self.elu_layer_0, self.linear_layer_1, self.elu_layer_1, self.linear_layer, self.elu_layer)
        # nn.init.ones_(self.linear_layer_0.weight)
        # nn.init.ones_(self.linear_layer_1.weight)
        # nn.init.ones_(self.linear_layer.weight)
        nn.init.uniform_(self.linear_layer_0.weight, 0, 1)
        nn.init.uniform_(self.linear_layer_1.weight, 0, 1)
        nn.init.uniform_(self.linear_layer.weight, 0, 1)

    def forward(self, input_list):
        """ Create the DNN model

        Args:
            input_list: (list<tf.tensor>) A list of tensors containing the features
                        for a list of documents.
            noisy_params: (dict<parameter_name, tf.variable>) A dictionary of noisy parameters to add.
            noise_rate: (float) A value specify how much noise to add.
            is_training: (bool) A flag indicating whether the model is running in training mode.

        Returns:
            A list of tf.Tensor containing the ranking scores for each instance in input_list.
        """
        input_data = torch.cat(input_list, dim=0)  # 2560 * feature_size
        input_data = input_data.to(dtype=torch.float32)
        if torch.cuda.is_available():
            input_data = input_data.to(device=torch.device('cuda'))
        # print(input_data.shape)
        output_data = self.propensity_net(input_data)  # 2560*6
        output_shape = input_list[0].shape[0]
        x = torch.split(output_data, output_shape, dim=0) # 一个列表，包含10个256*6
        x = torch.stack(x, dim=0)
        x = x.permute(1,0,2)
        mean = x[:, :, :self.dimension]
        log_var = x[:, :, self.dimension:]
        return mean, log_var

class Vectorization(BaseAlgorithm):
    def __init__(self, data_set, exp_settings):
        """Create the model.
        Args:
            data_set: (Raw_data) The dataset used to build the input layer.
            exp_settings: (dictionary) The dictionary containing the model settings.
        """
        print('Build Vectorization')

        self.hparams = ultra.utils.hparams.HParams(
            learning_rate=0.01,  # Learning rate.
            max_gradient_norm=5.0,  # Clip gradients to this norm.
            l2_loss=0.0,  # Set strength for L2 regularization.
            grad_strategy='ada',  # Optimizer
            dimension=3,  # Vector dimension
            pretrain_ranker_step=500,
            # The step for freezing the observation model and the base model
            prob_l2_loss=0.001,  # L2 regularization for the base model
            affine=0
            # 0/1, indicates whether to run in the Affine mode (Ali Vardasbi,
            # Harrie Oosterhuis, and Maarten de Rijke. When Inverse Propensity
            # Scoring does not Work: Affine Corrections for Unbiased Learning
            # to Rank. CIKM 2020)
        )
        print(exp_settings['learning_algorithm_hparams'])
        self.cuda = torch.device('cuda')
        self.is_cuda_avail = torch.cuda.is_available()
        self.writer = SummaryWriter()
        self.train_summary = {}
        self.eval_summary = {}
        self.hparams.parse(exp_settings['learning_algorithm_hparams'])
        print("hparam", self.hparams.to_json())
        self.exp_settings = exp_settings
        if self.exp_settings['ranking_model_hparams'].strip() != '':
            self.exp_settings['ranking_model_hparams'] += ","
        self.exp_settings['ranking_model_hparams'] += (
                "output_size=" + str(self.hparams.dimension))
        self.max_candidate_num = exp_settings['max_candidate_num']
        self.feature_size = data_set.feature_size
        if 'selection_bias_cutoff' in exp_settings.keys():
            self.rank_list_size = self.exp_settings['selection_bias_cutoff']
        self.model = self.create_model(self.feature_size)
        self.propensity_model = PropensityModel(self.rank_list_size, self.hparams.dimension)
        self.density_model = DNN_density(self.hparams.dimension, self.feature_size)

        if self.is_cuda_avail:
            self.model = self.model.to(device=self.cuda)
            self.propensity_model = self.propensity_model.to(device=self.cuda)
            self.density_model = self.density_model.to(device=self.cuda)

        self.letor_features_name = "letor_features"
        self.letor_features = None
        self.docid_inputs_name = []  # a list of top documents
        self.labels_name = []  # the labels for the documents (e.g., clicks)
        self.docid_inputs = []  # a list of top documents
        self.labels = []  # the labels for the documents (e.g., clicks)
        for i in range(self.max_candidate_num):
            self.docid_inputs_name.append("docid_input{0}".format(i))
            self.labels_name.append("label{0}".format(i))

        self.learning_rate = float(self.hparams.learning_rate)
        self.global_step = 0

        self.optimizer_func = torch.optim.Adagrad
        if self.hparams.grad_strategy == 'sgd':
            self.optimizer_func = torch.optim.SGD

    def train(self, input_feed):
        self.model.train()
        self.propensity_model.train()
        self.density_model.train()
        self.create_input_feed(input_feed, self.max_candidate_num)
        self.rank_list_size = self.exp_settings['selection_bias_cutoff']

        rel_vector, propensity_vector, base_vector = self.build_models(
            self.rank_list_size)  # [B, T, d]
        # print(rel_vector)
        # print(propensity_vector)
        # print(base_vector)
        click = self.combine_vector(
            rel_vector, propensity_vector)  # [B, T]
        # print(click.shape)
        # print(self.labels.shape)
        trained_labels = self.labels[:,:self.rank_list_size]  # (B, T)
        # print(trained_labels.shape)
        self.supervise_loss = self.softmax_loss(click, trained_labels)
        self.base_vector_loss = self.build_observation_density_loss(
            propensity_vector)
        print(self.supervise_loss)
        print(self.base_vector_loss)
        self.loss = self.supervise_loss + self.base_vector_loss

        self.build_update(self.loss)

        print(" Loss %f at Global Step %d: " % (self.loss.item(), self.global_step))
        self.train_summary['supervise_loss'] = self.supervise_loss.item()
        self.train_summary['base_vector_loss'] = self.base_vector_loss.item()
        self.train_summary['loss'] = self.loss.item()
        self.global_step += 1
        return self.loss, None, self.train_summary

    def validation(self, input_feed, is_online_simulation=False):
        self.model.eval()
        self.density_model.eval()
        self.create_input_feed(input_feed, self.max_candidate_num)
        with torch.no_grad():
            self.output_tuple = self.build_models(self.max_candidate_num,
                                              is_predict=True)
            relevance, _, base_vector = self.output_tuple  # (B, T, d)
            if self.hparams.affine == 1:
                self.output = relevance[:, :, 0]
            else:
                self.output = self.combine_vector(relevance, base_vector)

        if not is_online_simulation:
            pad_removed_output = self.remove_padding_for_metric_eval(
                self.docid_inputs, self.output)
            # reshape from [max_candidate_num, ?] to [?, max_candidate_num]
            for metric in self.exp_settings['metrics']:
                topn = self.exp_settings['metrics_topn']
                metric_values = ultra.utils.make_ranking_metric_fn(
                    metric, topn)(self.labels, pad_removed_output, None)
                for topn, metric_value in zip(topn, metric_values):
                    self.create_summary('%s_%d' % (metric, topn),
                                        '%s_%d' % (metric, topn), metric_value.item(), False)
        return None, self.output, self.eval_summary  # no loss, outputs, summary.


    # def evaluate(self, output):
    #     # output: (B, T)
    #     # label: (B, T)
    #     label = self.labels
    #     pad_removed_output = self.remove_padding_for_metric_eval(
    #         self.docid_inputs, output)
    #     reshaped_labels = tf.transpose(tf.convert_to_tensor(label))
    #     for metric in self.exp_settings['metrics']:
    #         for topn in self.exp_settings['metrics_topn']:
    #             metric_value = ultra.utils.make_ranking_metric_fn(
    #                 metric, topn)(reshaped_labels, pad_removed_output, None)
    #             item_name = '%s_%d' % (metric, topn)
    #             tf.summary.scalar(
    #                 item_name, metric_value, collections=['eval'])

    def combine_vector(self, v1, v2, keepdims=False):
        return torch.sum(v1 * v2, dim=-1, keepdim=keepdims)

    # def estimate_output(self, output_tuple):
    #     relevance, _, base_vector = output_tuple  # (B, T, d)
    #     if self.hparams.affine == 1:
    #         output = relevance[:, :, 0]
    #     else:
    #         output = self.combine_vector(relevance, base_vector)
    #     self.evaluate(output)
    #     return output

    def build_update(self, loss):
        #######
        density_model_params = self.density_model.parameters()
        propensity_model_params = self.propensity_model.parameters()
        ranking_model_params = self.model.parameters()
        # Select optimizer

        if self.hparams.l2_loss > 0:
            for p in density_model_params:
                loss += self.hparams.l2_loss * F.mse_loss(p, torch.zeros_like(p))
            for p in propensity_model_params:
                loss += self.hparams.l2_loss * F.mse_loss(p, torch.zeros_like(p))
            for p in ranking_model_params:
                loss += self.hparams.l2_loss * F.mse_loss(p, torch.zeros_like(p))

        opt_density = self.optimizer_func(self.density_model.parameters(), self.learning_rate)
        opt_propensity = self.optimizer_func(self.propensity_model.parameters(), self.learning_rate)
        opt_ranker = self.optimizer_func(self.model.parameters(), self.learning_rate)

        opt_density.zero_grad()
        opt_propensity.zero_grad()
        opt_ranker.zero_grad()

        loss.backward()

        if self.hparams.max_gradient_norm > 0:
            nn.utils.clip_grad_norm_(self.density_model.parameters(), self.hparams.max_gradient_norm)
            nn.utils.clip_grad_norm_(self.propensity_model.parameters(), self.hparams.max_gradient_norm)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.hparams.max_gradient_norm)

        opt_density.step()
        opt_propensity.step()
        opt_ranker.step()

        total_norm = 0

        for p in density_model_params:
            if p.grad != None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        for p in propensity_model_params:
            if p.grad != None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        for p in ranking_model_params:
            if p.grad != None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        self.norm = total_norm

    def build_models(self, T, is_predict=False, **kwargs):
        # Return (B, T, d)
        x = self.get_ranking_scores(self.model, self.docid_inputs[:T], **kwargs)  # (T, B, S_max) # 一个列表，包含T个batch_size*dimention
        relevance = torch.stack(x, dim=0)  # (T, B, xxx)
        if len(relevance.shape) == 2:
            relevance = torch.unsqueeze(relevance, -1)
        relevance = relevance.permute(1, 0, 2)
        if relevance.shape[-1] < self.hparams.dimension:
            raise ValueError('Vectorization requires the ranking model output size >= ' +
                             str(self.hparams.dimension) +
                             ", but get " + str(relevance.shape[-1])
                             + ". Please add 'output_size' in the hparams of this ranking model, "
                               "and adjust the size of build() method correspondingly.")
        relevance = relevance[:, :, :self.hparams.dimension]
        # (B, T, d)
        if self.hparams.affine == 1:
            relevance[:, :, 1:] = torch.ones_like(relevance[:, :, 1:])

        base_vector = self.get_base_vector_with_density(T)

        if not is_predict:
            propensity = self.propensity_model(relevance)
            # since observation model is initialized to ones,
            # we train relevance model first to ensure early stability.
            if self.hparams.affine == 0:
                if self.global_step >= self.hparams.pretrain_ranker_step:
                    propensity = propensity.detach()
        else:
            propensity = None
        return relevance, propensity, base_vector

    # def build_propensity_model(self, relevance, T, dimension):
    #     if not hasattr(
    #             self, "_propensity_model") or self._propensity_model is None:
    #         initializer = tf.initializers.constant(1.0)
    #         self._propensity_model = tf.get_variable("pbm_weight", (1, T, dimension),
    #                                                  initializer=initializer)
    #     batch_size = tf.shape(relevance)[0]
    #     return tf.tile(self._propensity_model, [batch_size, 1, 1])  # (B, T, d)

    def build_observation_density_loss(self, propensities):
        # propensities: (B, T, d)
        # return: loss
        input_feature_list = self.get_input_feature_list(self.docid_inputs[:self.rank_list_size])
        mean, log_var = self.density_model(input_feature_list) # (B, T, d)
        can_start_training = self.global_step >= self.hparams.pretrain_ranker_step
        if not can_start_training:
            mean = mean.detach()
            log_var = log_var.detach()
        # self.scalar(log_var, "log_var", train_only=True)
        propensities_cal = propensities.detach()
        mean_loss = torch.mean(
            torch.square(mean - propensities_cal) * torch.nan_to_num(torch.exp(-log_var))
        )
        var_loss = torch.nan_to_num(torch.mean(log_var))
        l2_loss = 0
        # for m in self._density_model:
        #     kernel = m.kernel
        #     l2_loss += tf.nn.l2_loss(kernel) * self.hparams.prob_l2_loss
        #     print(
        #         "Add density kernel L2 reg: " +
        #         str(kernel),
        #         self.hparams.prob_l2_loss)
        loss = torch.nan_to_num(mean_loss + var_loss + l2_loss)
        # self.scalar(mean_loss, "density_mean_loss", train_only=True)
        # self.scalar(var_loss, "density_var_loss", train_only=True)
        # self.scalar(l2_loss, "density_l2_loss", train_only=True)
        return loss

    # def get_base_vector_with_density(self, T):
    #     mean, log_var = self.density_model(self.docid_inputs[:T])  # (B, T, d)
    #     docid_inputs_tensor = tf.expand_dims(
    #         tf.transpose(tf.stack(self.docid_inputs[:T])), axis=-1)
    #     valid_flag = tf.where(
    #         tf.equal(
    #             docid_inputs_tensor,
    #             tf.cast(
    #                 tf.shape(
    #                     self.letor_features)[0],
    #                 tf.int64)),
    #         tf.zeros_like(docid_inputs_tensor, dtype=tf.float32),
    #         tf.ones_like(docid_inputs_tensor, dtype=tf.float32)
    #     )  # (B, T, 1)
    #     weight = tf.exp(-log_var) * valid_flag  # (B, T, d)
    #     base_vector = tf.reduce_mean(mean * weight, axis=1, keepdims=True) / \
    #         tf.reduce_mean(weight, axis=1, keepdims=True)  # (B, 1, d)
    #     return base_vector

    def get_input_feature_list(self, input_id_list):
        """Copy from base_algorithm.get_ranking_scores()
        """
        PAD_embed = np.zeros((1, self.feature_size), dtype=np.float32)
        letor_features = np.concatenate((self.letor_features, PAD_embed), axis=0)
        input_feature_list = []
        for i in range(len(input_id_list)):
            input_feature_list.append(torch.from_numpy(np.take(letor_features, input_id_list[i], 0)))
        return input_feature_list

    def get_base_vector_with_density(self, T):
        input_feature_list = self.get_input_feature_list(self.docid_inputs[:T])
        mean, log_var = self.density_model(input_feature_list)  # (B, T, d)

        # print(log_var)
        #
        docid_inputs_tensor = torch.unsqueeze(
            self.docid_inputs[:T].permute(1,0), dim=-1).to(device=torch.device('cuda')) #(B, T, 1)
        # print(docid_inputs_tensor)
        # print(docid_inputs_tensor.shape)
        #
        # print(torch.tensor(
        #             self.letor_features.shape[0],
        #             dtype=torch.long))
        valid_flag = torch.where(
            torch.eq(
                docid_inputs_tensor,
                torch.tensor(
                    self.letor_features.shape[0],
                    dtype=torch.long).to(device=torch.device('cuda'))),
            torch.zeros_like(docid_inputs_tensor, dtype=torch.float32).to(device=torch.device('cuda')),
            torch.ones_like(docid_inputs_tensor, dtype=torch.float32).to(device=torch.device('cuda'))
        )  # (B, T, 1)
        # print(docid_inputs_tensor.shape)

        weight = torch.exp(-log_var) * valid_flag # (B, T, d)
        weight = torch.nan_to_num(weight)
        # print(weight)
        base_vector = torch.mean(mean * weight, dim=1, keepdim=True) / \
                      torch.mean(weight, dim=1, keepdim=True)  # (B, 1, d)
        base_vector = torch.nan_to_num(base_vector)
        # print(base_vector)
        return base_vector

    # def build_observation_density_model(self, T):
    #     D = self.hparams.dimension
    #     features = self.get_input_feature_list(
    #         self.docid_inputs[:T])  # [T, (B, F)]
    #     features = tf.stop_gradient(
    #         tf.transpose(
    #             tf.stack(features), [
    #                 1, 0, 2]))  # (B, T, F)
    #     if not hasattr(self, "_density_model") or self._density_model is None:
    #         self._density_model = [
    #             tf.keras.layers.Dense(256, activation="elu"),
    #             tf.keras.layers.Dense(64, activation="elu"),
    #             tf.keras.layers.Dense(D * 2),
    #         ]
    #     x = features
    #
    #     with tf.name_scope('density'):
    #         for m in self._density_model:
    #             x = m(x)
    #     mean = x[:, :, :D]
    #     log_var = x[:, :, D:]
    #     return mean, log_var


