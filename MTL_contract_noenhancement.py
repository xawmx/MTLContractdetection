import numpy as np
import tensorflow as tf
from tensorflow.python.framework import function
from tensorflow.python.framework import dtypes
import numpy.matlib
import re
import os
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss

class MTDataset:
    def __init__(self, data, label, task_interval, num_class, batch_size):
        self.data = data
        self.data_dim = data.shape[1]
        self.label = np.reshape(label, [1, -1])
        self.task_interval = np.reshape(task_interval, [1, -1])
        self.num_task = task_interval.size - 1
        self.num_class = num_class
        self.batch_size = batch_size
        self.__build_index__()

    def __build_index__(self):
        index_list = []
        for i in range(self.num_task):
            start = self.task_interval[0, i]
            end = self.task_interval[0, i + 1]
            for j in range(self.num_class):
                index_list.append(np.arange(start, end)[np.where(self.label[0, start:end] == j)[0]])
        self.index_list = index_list
        self.counter = np.zeros([1, self.num_task * self.num_class], dtype=np.int32)

    def get_next_batch(self):
        sampled_data = np.zeros([self.batch_size * self.num_class * self.num_task, self.data_dim], dtype=np.float32)
        sampled_label = np.zeros([self.batch_size * self.num_class * self.num_task, self.num_class], dtype=np.int32)
        sampled_task_ind = np.zeros([1, self.batch_size * self.num_class * self.num_task], dtype=np.int32)
        sampled_label_ind = np.zeros([1, self.batch_size * self.num_class * self.num_task], dtype=np.int32)
        for i in range(self.num_task):
            for j in range(self.num_class):
                cur_ind = i * self.num_class + j
                task_class_index = self.index_list[cur_ind]
                sampled_ind = range(cur_ind * self.batch_size, (cur_ind + 1) * self.batch_size)
                sampled_task_ind[0, sampled_ind] = i
                sampled_label_ind[0, sampled_ind] = j
                sampled_label[sampled_ind, j] = 1
                if task_class_index.size < self.batch_size:
                    sampled_data[sampled_ind, :] = self.data[np.concatenate((task_class_index, task_class_index[
                        np.random.randint(0, high=task_class_index.size, size=self.batch_size - task_class_index.size)])), :]
                elif self.counter[0, cur_ind] + self.batch_size < task_class_index.size:
                    sampled_data[sampled_ind, :] = self.data[task_class_index[self.counter[0, cur_ind]:self.counter[0, cur_ind] + self.batch_size], :]
                    self.counter[0, cur_ind] = self.counter[0, cur_ind] + self.batch_size
                else:
                    sampled_data[sampled_ind, :] = self.data[task_class_index[-self.batch_size:], :]
                    self.counter[0, cur_ind] = 0
                    np.random.shuffle(self.index_list[cur_ind])
        return sampled_data, sampled_label, sampled_task_ind, sampled_label_ind

class MTDataset_Split:
    def __init__(self, contract_data, contract_label, num_class):
        # self.data = np.vstack((contract_data, func_data))
        self.data = contract_data
        self.contract_num = contract_data.shape[0]
        # self.func_num = func_data.shape[0]
        self.task_interval = [0]
        self.task_interval.append(self.contract_num)
        # self.task_interval.append(self.contract_num + self.func_num)
        self.data_dim = contract_data.shape[1]
        self.contract_label = np.reshape(contract_label, [1, -1])
        # self.func_label = np.reshape(func_label, [1, -1])
        # self.label = np.hstack((self.contract_label, self.func_label))
        self.label = contract_label
        self.num_task = 1
        self.num_class = 2
        self.__build_index__()

    def __build_index__(self):
        index_list = []
        self.num_class_ins = np.zeros([self.num_task, self.num_class])
        for i in range(self.num_task):
            start = self.task_interval[i]
            end = self.task_interval[i + 1]
            for j in range(self.num_class):
                index_array = np.where(self.label[0, start:end] == j)[0]
                self.num_class_ins[i, j] = index_array.size
                index_list.append(np.arange(start, end)[index_array])
        self.index_list = index_list
    # index_list 表示不同任务中不同类型的索引值
    def split(self, train_size):
        if train_size < 1:
            train_num = np.ceil(self.num_class_ins * train_size).astype(np.int32)
        else:
            train_num = np.ones([self.num_task, self.num_class], dtype=np.int32) * train_size
            train_num = np.maximum(1, np.minimum(train_num, self.num_class_ins - 10))
            train_num = train_num.astype(np.int32)
        traindata = np.zeros([0, self.data_dim], dtype=np.float32)
        testdata = np.zeros([0, self.data_dim], dtype=np.float32)
        trainlabel = np.zeros([1, 0], dtype=np.int32)
        testlabel = np.zeros([1, 0], dtype=np.int32)
        train_task_interval = np.zeros([1, self.num_task + 1], dtype=np.int32)
        test_task_interval = np.zeros([1, self.num_task + 1], dtype=np.int32)
        for i in range(self.num_task):
            for j in range(self.num_class):
                cur_ind = i * self.num_class + j
                task_class_index = self.index_list[cur_ind]
                np.random.shuffle(task_class_index)
                train_index = task_class_index[0:train_num[i, j]]
                test_index = task_class_index[train_num[i, j]:]
                traindata = np.concatenate((traindata, self.data[train_index, :]), axis=0)
                trainlabel = np.concatenate((trainlabel, np.ones([1, train_index.size], dtype=np.int32) * j), axis=1)
                testdata = np.concatenate((testdata, self.data[test_index, :]), axis=0)
                testlabel = np.concatenate((testlabel, np.ones([1, test_index.size], dtype=np.int32) * j), axis=1)
            train_task_interval[0, i + 1] = trainlabel.size
            test_task_interval[0, i + 1] = testlabel.size
        return traindata, trainlabel, train_task_interval, testdata, testlabel, test_task_interval

def read_data_from_file(contract_filename, func_filename, contract_label_file, func_label_file, graph_edge_file):

    contract_file = open(contract_filename, 'r')
    # func_file = open(func_filename, 'r')
    contract_label_file = open(contract_label_file, 'r')
    # func_label_file = open(func_label_file, 'r')
    contract_edge_file = open(graph_edge_file, 'r')

    contract_contents = contract_file.readlines()
    # func_contents = func_file.readlines()
    contract_label_contents = contract_label_file.readlines()
    # func_label_contents = func_label_file.readlines()
    contract_edge_contents = contract_edge_file.readlines()
    contract_edge = []
    for i in range(len(contract_edge_contents)):
        contract_edge.append([int(n) for n in contract_edge_contents[i].split('\t') if n != '\n'])
    contract_edge = np.array(contract_edge)

    contract_subset = []
    edge_index = [i for i in range(contract_edge.shape[0])]
    for i in edge_index:
        link_index = []
        link_index.append(i)
        link_set = [n for n in range(len(contract_edge[i])) if contract_edge[i][n] == 1]
        link_index = link_index + link_set
        for n in link_set:
            subset_link = []
            if n in edge_index:
                subset_link = subset_link + [m for m in range(len(contract_edge[n])) if contract_edge[n][m] == 1]
                link_index = link_index + subset_link
                edge_index.remove(n)
        contract_subset.append(link_index)

    contract_subset_nodup = []
    for n in range(len(contract_subset)):
        contract_subset_nodup.append(list(set(contract_subset[n])))
    contract_file.close()
    # func_file.close()
    contract_label_file.close()
    # func_label_file.close()
    contract_edge_file.close()

    num_task = int(1)
    num_class = int(2)
    # temp_ind = re.split(',', contents[2])
    # temp_ind = [int(elem) for elem in temp_ind]
    # task_interval = np.reshape(np.array(temp_ind), [1, -1])
    # ===feature
    temp_contract_data = []
    # temp_func_data = []
    for pos in range(len(contract_contents)):
        temp_sub_data = contract_contents[pos].split()
        temp_sub_data = [float(elem) for elem in temp_sub_data]
        temp_contract_data.append(temp_sub_data)
    # for pos in range(len(func_contents)):
    #     temp_sub_data = func_contents[pos].split()
    #     temp_sub_data = [float(elem) for elem in temp_sub_data]
    #     temp_sub_data = temp_sub_data[0:250]
    #     temp_func_data.append(temp_sub_data)
    contract_data = np.array(temp_contract_data)
    # func_data = np.array(temp_func_data)

    # ===labels
    temp_contract_label = []
    # temp_func_label = []
    for pos in range(len(contract_label_contents)):
        temp_sub_data = contract_label_contents[pos].split()[1]
        temp_contract_label.append(int(temp_sub_data))
    # for pos in range(len(func_label_contents)):
    #     temp_sub_data = func_label_contents[pos].split()[1]
    #     temp_contract_label.append(int(temp_sub_data))
    temp_contract_label = np.reshape(np.array(temp_contract_label), [1, -1])
    # temp_func_label = np.reshape(np.array(temp_func_label), [1, -1])
    contract_label = temp_contract_label
    # func_label = temp_func_label

    # return contract_data, func_data, contract_label, func_label, num_task, num_class
    return contract_data, contract_label, num_task, num_class

def compute_train_loss(i, feature_representation, hidden_output_weight, inputs_data_label, inputs_task_ind, inputs_num_ins_per_task, train_loss):
    train_loss += tf.div(tf.losses.softmax_cross_entropy(tf.expand_dims(inputs_data_label[i, :], 0),
                                                         tf.matmul(tf.expand_dims(feature_representation[inputs_task_ind[0, i]][i % (batch_size * inputs_data_label.shape[-1])][:], 0),
                                                                   hidden_output_weight[inputs_task_ind[0, i], :, :])),
                         tf.cast(inputs_num_ins_per_task[0, inputs_task_ind[0, i]], dtype=tf.float32))
    # tf.expand_dims 增加维度 tf.cast 转换类型
    # 第二项是输出标签
    return i + 1, feature_representation, hidden_output_weight, inputs_data_label, inputs_task_ind, inputs_num_ins_per_task, train_loss


def gradient_clipping_tf_false_consequence(optimizer, obj, gradient_clipping_threshold):
    gradients, variables = zip(*optimizer.compute_gradients(obj))
    gradients = [None if gradient is None else tf.clip_by_value(gradient, gradient_clipping_threshold,
                                                                tf.negative(gradient_clipping_threshold)) for gradient in gradients]
    train_step = optimizer.apply_gradients(zip(gradients, variables))
    return train_step


def gradient_clipping_tf(optimizer, obj, option, gradient_clipping_threshold):
    train_step = tf.cond(tf.equal(option, 0), lambda: optimizer.minimize(obj),
                         lambda: gradient_clipping_tf_false_consequence(optimizer, obj, gradient_clipping_threshold))
    train_step = tf.group(train_step)
    return train_step


def generate_label_task_ind(label, task_interval, num_class):
    num_task = task_interval.size - 1
    num_ins = label.size
    label_matrix = np.zeros((num_ins, num_class), dtype=np.int32)
    label_matrix[range(num_ins), label] = 1
    task_ind = np.zeros((1, num_ins), dtype=np.int32)
    for i in range(num_task):
        task_ind[0, task_interval[0, i]: task_interval[0, i + 1]] = i
    return label_matrix, task_ind


def compute_errors(hidden_rep, hidden_output_weight, task_ind, label, num_task):
    # num_total_ins = hidden_rep.shape[0]
    # num_ins = np.zeros([1, num_task])
    # TP = np.zeros([1, num_task])
    # FN = np.zeros([1, num_task])
    # FP = np.zeros([1, num_task])
    # errors = np.zeros([1, num_task + 1])
    # acc = np.zeros([1, num_task + 1])
    # for i in range(num_total_ins):
    #     probit = np.matmul(hidden_rep[i, :], hidden_output_weight[task_ind[0, i], :, :])
    #     num_ins[0, task_ind[0, i]] += 1
    #     if np.argmax(probit) != label[0, i]:
    #         errors[0, task_ind[0, i]] += 1
    #         if np.argmax(probit) == 0:
    #             FN[0, task_ind[0, i]] += 1
    #         if np.argmax(probit) == 1:
    #             FP[0, task_ind[0, i]] += 1
    #     else:
    #         acc[0, task_ind[0, i]] += 1
    #         if np.argmax(probit) == 1:
    #             TP[0, task_ind[0, i]] += 1
    # for i in range(num_task):
    #     errors[0, i] = errors[0, i] / num_ins[0, i]
    #     acc[0, i] = acc[0, i] / num_ins[0, i]
    # errors[0, num_task] = np.mean(errors[0, 0: num_task])
    # acc[0, num_task] = np.mean(acc[0, 0: num_task])
    # recall = TP[0, 0] / (TP[0, 0] + FN[0, 0])
    # # recall1 = TP[0, 1] / (TP[0, 1] + FN[0, 1])
    # precesion0 = TP[0, 0] / (TP[0, 0] + FP[0, 0])
    # # precesion1 = TP[0, 1] / (TP[0, 1] + FP[0, 1])
    # f1 = []
    # # f10 = 2 * recall0 * precesion0 / (recall0 + precesion0)
    # # f11 = 2 * recall1 * precesion1 / (recall1 + precesion1)
    # f10 = recall
    # # f11 = recall1
    # f1.append(f10)
    # # f1.append(f11)
    num_total_ins = hidden_rep.shape[0]
    num_ins = np.zeros([1, num_task])
    TP = np.zeros([1, num_task])
    FN = np.zeros([1, num_task])
    FP = np.zeros([1, num_task])
    errors = np.zeros([1, num_task + 1])
    loss = np.zeros([1, num_task + 1])
    acc = np.zeros([1, num_task + 1])
    label1 = []
    label2 = []
    aim1 = []
    aim2 = []
    for i in range(num_total_ins):
        # if label[0, i] == 0:
        #     temp_label = np.array([1, 0], dtype=np.float32).reshape((1, 2))
        # else:
        #     temp_label = np.array([0, 1], dtype=np.float32).reshape((1, 2))
        probit = np.matmul(hidden_rep[i, :], hidden_output_weight[task_ind[0, i], :, :]).reshape((1, 2))
        if task_ind[0, i] == 0:
            if label[0, i] == 0:
                label1.append([1.0, 0.0])
            else:
                label1.append([0.0, 1.0])
            aim1.append(probit[0].tolist())
        else:
            if label[0, i] == 0:
                label2.append([1.0, 0.0])
            else:
                label2.append([0.0, 1.0])
            aim2.append(probit[0].tolist())
        num_ins[0, task_ind[0, i]] += 1
        if np.argmax(probit) != label[0, i]:
            errors[0, task_ind[0, i]] += 1
            if np.argmax(probit) == 0:
                FN[0, task_ind[0, i]] += 1
            if np.argmax(probit) == 1:
                FP[0, task_ind[0, i]] += 1
        else:
            acc[0, task_ind[0, i]] += 1
            if np.argmax(probit) == 1:
                TP[0, task_ind[0, i]] += 1

    for i in range(len(label1)):
        loss[0, 0] += log_loss(label1[i][:], aim1[i][:])
    for i in range(len(label2)):
        loss[0, 1] += log_loss(label2[i][:], aim2[i][:])
    loss[0, 0] = loss[0, 0] / len(label1)
    loss[0, 1] = loss[0, 0] / len(label)
    for i in range(num_task):
        errors[0, i] = errors[0, i] / num_ins[0, i]
        acc[0, i] = acc[0, i] / num_ins[0, i]
    errors[0, num_task] = np.mean(errors[0, 0: num_task])
    acc[0, num_task] = np.mean(acc[0, 0: num_task])
    # loss[0, num_task] = np.mean(loss[0, 0: num_task])

    recall0 = TP[0, 0] / (TP[0, 0] + FN[0, 0])
    # recall1 = TP[0, 1] / (TP[0, 1] + FN[0, 1])
    precesion0 = TP[0, 0] / (TP[0, 0] + FP[0, 0])
    # precesion1 = TP[0, 1] / (TP[0, 1] + FP[0, 1])
    f1 = []
    f10 = 2 * recall0 * precesion0 / (recall0 + precesion0)
    # f11 = 2 * recall1 * precesion1 / (recall1 + precesion1)
    # f10 = recall0
    # f11 = recall1
    f1.append(recall0)
    # f1.append(f11)
    return errors, acc, f1


def change_datastruct(hidden_features, num_task):
    return tf.reshape(hidden_features, [num_task, -1, hidden_features.shape[-1]])


def compute_pairwise_dist_tf(data):
    # 输入的是一个task中每一条数据对应的特征，二维矩阵
    sq_data_norm = tf.reduce_sum(tf.square(data), axis=1)
    # 计算平方后的行和
    sq_data_norm = tf.reshape(sq_data_norm, [-1, 1])
    # 排成列
    dist_matrix = sq_data_norm - 2 * tf.matmul(data, data, transpose_b=True) + tf.matrix_transpose(sq_data_norm)
    # 二范数，返回特征之间的二范数
    return dist_matrix


def compute_pairwise_dist_np(data):
    sq_data_norm = np.sum(data ** 2, axis=1)
    sq_data_norm = np.reshape(sq_data_norm, [-1, 1])
    dist_matrix = sq_data_norm - 2 * np.dot(data, data.transpose()) + sq_data_norm.transpose()
    return dist_matrix


def compute_adjacency_matrix(hidden_features, inputs_data_label, num_task):
    new_hidden_features = change_datastruct(hidden_features, num_task)
    new_inputs_data_label = change_datastruct(inputs_data_label, num_task)
    adjacency_matrixs = []
    for i in range(num_task):
        dist_matrix = -compute_pairwise_dist_tf(new_hidden_features[i])
        # 得到特征间的距离
        sign_matrix = 2 * tf.matmul(new_inputs_data_label[i], tf.matrix_transpose(new_inputs_data_label[i])) - 1
        # matrix_transpose转置，得到标签相乘的结果，这里应该是在计算邻接矩阵的链接关系
        adjacency_matrix = tf.exp(dist_matrix) * sign_matrix
        # 计算邻接矩阵
        adjacency_matrixs.append(adjacency_matrix)
        # 按层添加邻接矩阵关系
    adjacency_matrixs = tf.stack(adjacency_matrixs)
    # 拼接起来
    return adjacency_matrixs


def activate_function(temp, activate_op):
    if activate_op == 1:
        return tf.tanh(temp)
    elif activate_op == 2:
        return tf.nn.relu(temp)
    elif activate_op == 3:
        return tf.nn.elu(temp)
    else:
        return


def get_normed_distance_tf(data):
    norminator = tf.matmul(data, tf.transpose(data))
    square = tf.reshape(tf.sqrt(tf.reduce_sum(tf.square(data), 1)), [norminator.shape[0], 1])
    denorminator = tf.matmul(square, tf.transpose(square))
    return norminator/denorminator


def get_normed_distance_np(data):
    norminator = np.matmul(data, np.transpose(data))
    square = np.reshape(np.sqrt(np.sum(np.square(data), 1)), [norminator.shape[0], 1])
    denorminator = np.matmul(square, np.transpose(square))
    return norminator/denorminator


def GAT(attention_weight, embedding_vectors):
    transformaed_embedding_vectors = tf.matmul(embedding_vectors, attention_weight)
    attention_values = tf.nn.softmax(get_normed_distance_tf(transformaed_embedding_vectors))
    return attention_values


def get_feature_representation(inputs, input_hidden_weights, hidden_features, adjacency_matrix, num_task, num_class, activate_op,
                            first_task_att_w, first_class_att_w, task_attention_weight, class_attention_weight, inputs_data_label):
    hidden_representation = activate_function(
        tf.add(change_datastruct(tf.matmul(inputs, input_hidden_weights), num_task),
               tf.matmul(adjacency_matrix, change_datastruct(hidden_features, num_task))), activate_op)
    # activate选择归一化函数 input_hidden_weights 公式1W, 这一行代码实际上就是公式2，加上了图相关信息，得到了几个任务中的表示，对应
    new_inputs_data_label = change_datastruct(inputs_data_label, num_task)
    # 将标签也按任务数分好
    new_adjacency_matrix = []
    # 新的邻接矩阵，指task之间的, 这里替换作为类别特征之间的近似度关系
    for i in range(num_task):
        dist_matrix = -compute_pairwise_dist_tf(hidden_representation[i])
        sign_matrix = 2 * tf.matmul(new_inputs_data_label[i], tf.matrix_transpose(new_inputs_data_label[i])) - 1
        adjacency_matrix = tf.exp(dist_matrix) * sign_matrix
        new_adjacency_matrix.append(adjacency_matrix)
    new_adjacency_matrix = tf.stack(new_adjacency_matrix)
    new_hidden_representation = activate_function(
        tf.add(change_datastruct(tf.matmul(inputs, input_hidden_weights), num_task),
               tf.matmul(new_adjacency_matrix, hidden_representation)), activate_op)
    # 得到更新后的邻接矩阵，重新计算每个数据的隐藏表征
    task_embedding_vectors = tf.reduce_max(new_hidden_representation, 1)
    # 每个task的表征
    task_attention_values = GAT(first_task_att_w, task_embedding_vectors)
    # 计算任务注意力值
    new_task_embedding_vectors = tf.tanh(tf.matmul(task_attention_values, tf.matmul(task_embedding_vectors, first_task_att_w)))

    task_attention_values = GAT(task_attention_weight, new_task_embedding_vectors)
    new_task_embedding_vectors = tf.tanh(tf.matmul(task_attention_values, tf.matmul(new_task_embedding_vectors, task_attention_weight)))

    class_embedding_vectors = []
    for i in range(num_task):
        class_hidden_rep = tf.reshape(new_hidden_representation[i], [num_class, -1, new_hidden_representation.shape[-1]])
        for j in range(num_class):
            class_embedding_vectors.append(tf.reduce_max(class_hidden_rep[j], 0))
    class_embedding_vectors = tf.stack(class_embedding_vectors)
    # 得到类嵌入表达式
    class_attention_values = GAT(first_class_att_w, class_embedding_vectors)
    # 计算类之间的注意力，这里是考虑到没有链接关系时
    new_class_embedding_vectors = tf.tanh(tf.matmul(class_attention_values, tf.matmul(class_embedding_vectors, first_class_att_w)))
    class_attention_values = GAT(class_attention_weight, new_class_embedding_vectors)
    new_class_embedding_vectors = tf.tanh(tf.matmul(class_attention_values, tf.matmul(new_class_embedding_vectors, class_attention_weight)))
    # 得到类的表征
    feature_representations = []
    for i in range(num_task):
        feature_representation = []
        feature_representation_1 = tf.concat([
            hidden_features[i * batch_size * num_class: (i + 1) * batch_size * num_class],
            tf.stack([new_task_embedding_vectors[i] for _ in range(num_class * batch_size)])], 1)
        for j in range(num_class):
            feature_representation_2 = tf.concat([
                feature_representation_1[j * batch_size: (j + 1) * batch_size],
                tf.stack([new_class_embedding_vectors[i * num_task + j] for _ in range(batch_size)])], 1)
            feature_representation.append(feature_representation_2)
        feature_representations.append(feature_representation)
    # 得到链接后特征，主要是隐式表征链接task表征链接类表征
    feature_representations = tf.stack(feature_representations)
    # 实际上返回的是最后增强的特征
    return tf.reshape(feature_representations, [num_task, -1, hidden_features.shape[-1] + F_pie_t + F_pie_c])


def np_softmax(x):
    x = x - np.max(x)
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x


def get_embedding_vec(traindata, input_hidden_weights, first_task_att_w, first_class_att_w, task_attention_weight,
                      class_attention_weight, train_hidden_features, train_label_matrix, train_task_ind, train_num_ins_per_task, num_task,  num_class):
    inputs = [[] for _ in range(num_task)]
    features = [[] for _ in range(num_task)]
    labels = [[] for _ in range(num_task)]
    for i in range(traindata.shape[0]):
        inputs[train_task_ind[0, i]].append(traindata[i])
        features[train_task_ind[0, i]].append(train_hidden_features[i])
        labels[train_task_ind[0, i]].append(train_label_matrix[i])
    task_embedding_vectors = []
    class_embedding_vectors = []
    for i in range(num_task):
        dist_matrix = -compute_pairwise_dist_np(np.stack(features[i]))
        sign_matrix = 2 * np.matmul(np.stack(labels[i]), np.transpose(np.stack(labels[i]))) - 1
        adjacency_matrix = np.exp(dist_matrix) * sign_matrix
        new_features = np.tanh(np.add(np.matmul(np.stack(inputs[i]), input_hidden_weights),
                                      np.matmul(adjacency_matrix, np.stack(features[i]))))
        new_dist_matrix = -compute_pairwise_dist_np(new_features)
        new_adjacency_matrix = np.exp(new_dist_matrix) * sign_matrix
        new_features = np.tanh(np.add(np.matmul(np.stack(inputs[i]), input_hidden_weights),
                                      np.matmul(new_adjacency_matrix, new_features)))

        task_embedding_vector = np.max(new_features, 0)
        task_embedding_vectors.append(task_embedding_vector)
        inputs_class = [[] for _ in range(num_class)]
        features_class = [[] for _ in range(num_class)]
        labels_class = [[] for _ in range(num_class)]
        for j in range(len(inputs[i])):
            inputs_class[int(np.argmax(labels[i][j]))].append(inputs[i][j])
            features_class[int(np.argmax(labels[i][j]))].append(features[i][j])
            labels_class[int(np.argmax(labels[i][j]))].append(labels[i][j])
        for j in range(num_class):
            dist_matrix = -compute_pairwise_dist_np(np.stack(features_class[j]))
            sign_matrix = 2 * np.matmul(np.stack(labels_class[j]), np.transpose(np.stack(labels_class[j]))) - 1
            adjacency_matrix = np.exp(dist_matrix) * sign_matrix
            new_features = np.tanh(np.add(np.matmul(np.stack(inputs_class[j]), input_hidden_weights),
                                          np.matmul(adjacency_matrix, np.stack(features_class[j]))))
            class_embedding_vector = np.max(new_features, 0)
            class_embedding_vectors.append(class_embedding_vector)
    task_attention_values = np_softmax(get_normed_distance_np(np.stack(task_embedding_vectors)))
    new_task_embedding_vectors = np.tanh(np.matmul(task_attention_values, np.matmul(task_embedding_vectors, first_task_att_w)))
    task_attention_values = np_softmax(get_normed_distance_np(np.stack(new_task_embedding_vectors)))
    new_task_embedding_vectors = np.tanh(np.matmul(task_attention_values, np.matmul(new_task_embedding_vectors, task_attention_weight)))

    class_attention_values = np_softmax(get_normed_distance_np(np.stack(class_embedding_vectors)))
    new_class_embedding_vectors = np.tanh(np.matmul(class_attention_values, np.matmul(class_embedding_vectors, first_class_att_w)))
    class_attention_values = np_softmax(get_normed_distance_np(np.stack(new_class_embedding_vectors)))
    new_class_embedding_vectors = np.tanh(np.matmul(class_attention_values, np.matmul(new_class_embedding_vectors, class_attention_weight)))

    return new_task_embedding_vectors, new_class_embedding_vectors


def get_new_hidden_features(test_hidden_rep, task_embedding_vectors, class_embedding_vectors, hidden_output_weight, test_task_ind, num_task, num_class):
    temp_test_hidden_rep = []
    task_embedding_vectors = [[1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6]]
    class_embedding_vectors = [[1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6], [1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6]]
    task_embedding_vectors = np.array(task_embedding_vectors)
    class_embedding_vectors = np.array(class_embedding_vectors)

    for i in range(test_hidden_rep.shape[0]):
        temp = np.concatenate([test_hidden_rep[i], task_embedding_vectors[test_task_ind[0, i]]], 0)
        temp_test_hidden_rep.append(temp)
    temp_test_hidden_rep = np.stack(temp_test_hidden_rep)
    test_hidden_rep = []
    for i in range(len(temp_test_hidden_rep)):
        task_id = test_task_ind[0, i]
        probits_softmax = []
        for j in range(num_class):
            temp = np.concatenate([temp_test_hidden_rep[i], class_embedding_vectors[task_id * num_task + j]], 0)
            probit_softmax = np_softmax(np.matmul(temp, hidden_output_weight[task_id]))
            probits_softmax.append(probit_softmax)
        probits_softmax = np.stack(probits_softmax)
        diagonal = []
        for j in range(num_class):
            diagonal.append(probits_softmax[j][j])
        class_id = np.argmax(diagonal)
        test_hidden_rep.append(np.concatenate([temp_test_hidden_rep[i], class_embedding_vectors[task_id * num_class + class_id]], 0))
    test_hidden_rep = np.stack(test_hidden_rep)
    return test_hidden_rep


def MTL_GRAPH(traindata, trainlabel, train_task_interval, dim, num_class, num_task, hidden_dim, batch_size, reg_para,
         max_epoch, testdata, testlabel, test_task_interval, activate_op):
    print('MTL_GRAPH is running...')
    train_lose_graph = []
    train_lose_graph_f = []
    inputs = tf.placeholder(tf.float32, shape=[None, dim])
    # 输入，现在只能确定dim
    inputs_data_label = tf.placeholder(tf.float32, shape=[None, num_class])
    # 数据标签，能确定种类
    inputs_task_ind = tf.placeholder(tf.int32, shape=[1, None])
    # 任务索引
    inputs_num_ins_per_task = tf.placeholder(tf.int32, shape=[1, None])
    # 每个任务的
    input_hidden_weights = tf.Variable(tf.truncated_normal([dim, hidden_dim], dtype=tf.float32, stddev=1e-1))
    # 用来计算特征的隐式表达，即公式1中的W
    hidden_features = activate_function(tf.matmul(inputs, input_hidden_weights), activate_op)
    # 隐式表达
    adjacency_matrix = compute_adjacency_matrix(hidden_features, inputs_data_label, num_task)
    # 计算邻接矩阵，我们这里不采用,处理为通过外来矩阵进行运算
    first_task_att_w = tf.Variable(tf.truncated_normal(
        [hidden_dim, GAT_hidden_dim], dtype=tf.float32, stddev=1e-1))
    # 用来提取注意力特征，即后面算距离中d=We(i, t)中的W
    first_class_att_w = tf.Variable(tf.truncated_normal(
        [hidden_dim, GAT_hidden_dim], dtype=tf.float32, stddev=1e-1))
    # 类比上式
    task_attention_weight = tf.Variable(tf.truncated_normal(
        [GAT_hidden_dim, F_pie_t], dtype=tf.float32, stddev=1e-1))
    # 目前还不清楚，具体方法
    class_attention_weight = tf.Variable(tf.truncated_normal(
        [GAT_hidden_dim, F_pie_c], dtype=tf.float32, stddev=1e-1))

    feature_representation = get_feature_representation(inputs, input_hidden_weights, hidden_features, adjacency_matrix,
                                               num_task, num_class, activate_op, first_task_att_w, first_class_att_w, task_attention_weight, class_attention_weight, inputs_data_label)
    # 这里的特征表征是指增强后特征形式
    hidden_output_weight = tf.Variable(tf.truncated_normal(
        [num_task, hidden_dim + F_pie_t + F_pie_c, num_class], dtype=tf.float32, stddev=1e-1))
    # 多任务层，分别分类
    train_loss = tf.Variable(0.0, dtype=tf.float32)
    # 训练损失
    _, _, _, _, _, _, train_loss = tf.while_loop(
        cond=lambda i, j1, j2, j3, j4, j5, j6: tf.less(i, tf.shape(inputs_task_ind)[1]), body=compute_train_loss,
        loop_vars=(tf.constant(0, dtype=tf.int32), feature_representation, hidden_output_weight,
                   inputs_data_label, inputs_task_ind, inputs_num_ins_per_task, train_loss))
    ########xinzeng
    # train_lose_graph.append(train_loss)
    # while_loop 执行，当less返回为真时，计算损失，loop的参数，1.i 第i个数据，feature_representation,每个数据链接增强特征后的特征
    # hidden_output_weight 可能是在提取完特征以后，最后的一层来作为multi-task
    # inputs_data_label 输入特征
    # inputs_task_ind 任务索引
    # inputs_num_ins_per_task 任务内数据索引
    # train_loss 训练损失
    obj = train_loss + reg_para * (tf.square(tf.norm(input_hidden_weights))+tf.square(tf.norm(hidden_output_weight)))
    # tf.norm计算范数
    learning_rate = tf.placeholder(tf.float32)
    # 学习率
    gradient_clipping_threshold = tf.placeholder(tf.float32)
    # 梯度裁剪阈值，一旦梯度超过该值，则设置为该值
    optimizer = tf.train.AdamOptimizer(learning_rate)

    gradient_clipping_option = tf.placeholder(tf.int32)
    train_step = gradient_clipping_tf(optimizer, obj, gradient_clipping_option, gradient_clipping_threshold)
    init_op = tf.global_variables_initializer()
    # global_variables_initializer将tf.variable声明的变量进行初始化
    with tf.Session() as sess:
        max_iter_epoch = numpy.ceil(traindata.shape[0] / (batch_size * num_task * num_class)).astype(
            np.int32)
        Iterator = MTDataset(traindata, trainlabel, train_task_interval, num_class, batch_size)
        sess.run(init_op)
        # 初始化参数
        train_label_matrix, train_task_ind = generate_label_task_ind(trainlabel, train_task_interval, num_class)
        for iter in range(max_iter_epoch * max_epoch):
            sampled_data, sampled_label, sampled_task_ind, _ = Iterator.get_next_batch()
            num_iter = iter // max_iter_epoch
            # 取整
            train_step.run(feed_dict={d1: d2 for d1, d2 in
                                      zip([learning_rate, gradient_clipping_option, gradient_clipping_threshold, inputs,
                                           inputs_data_label, inputs_task_ind, inputs_num_ins_per_task],
                                          [0.0002, 0, -5., sampled_data, sampled_label, sampled_task_ind,
                                           np.ones([1, num_task]) * (batch_size * num_class)])})
            # 0.02 / (1 + num_iter)
            # if iter % max_iter_epoch == 0 and num_iter % 1 == 0:
            if iter % 4 == 0 and num_iter % 1 == 0:
                train_hidden_features = hidden_features.eval(feed_dict={inputs: traindata, inputs_task_ind: train_task_ind})
                task_embedding_vectors, class_embedding_vectors = get_embedding_vec(traindata, input_hidden_weights.eval(), first_task_att_w.eval(), first_class_att_w.eval(), task_attention_weight.eval(), class_attention_weight.eval(),
                                    train_hidden_features, train_label_matrix, train_task_ind, np.reshape(
                                   train_task_interval[0, 1:] - train_task_interval[0, 0:num_task], [1, -1]), num_task, num_class)
                _, test_task_ind = generate_label_task_ind(testlabel, test_task_interval, num_class)
                test_hidden_rep = hidden_features.eval(feed_dict={inputs: testdata, inputs_task_ind: test_task_ind})
                new_test_hidden_rep = get_new_hidden_features(test_hidden_rep, task_embedding_vectors, class_embedding_vectors, hidden_output_weight.eval(), test_task_ind, num_task, num_class)
                # new_test_hidden_rep = test_hidden_rep
                # new_test_hidden_rep = get_new_hidden_features(test_hidden_rep, task_embedding_vectors, task_embedding_vectors, hidden_output_weight.eval(), test_task_ind, num_task, num_class)
                test_errors, test_acc, test_f1 = compute_errors(new_test_hidden_rep, hidden_output_weight.eval(), test_task_ind, testlabel, num_task)
                print('epoch = %g, test_errors = %s, test_acc = %s, test_f1 = %s' % (num_iter, test_errors, test_acc, test_f1))
                test_c = []
                test_f = []
                test_c.append(test_errors[0][0])
                test_c.append(test_acc[0][0])
                test_f.append(test_errors[0][1])
                test_f.append(test_acc[0][1])

                train_lose_graph.append(test_c)
                train_lose_graph_f.append(test_f)
    return test_errors, train_lose_graph, train_lose_graph_f


def main_process(contract_filename, func_filename, graph_label_file, func_label_file, graph_edge_file, train_size, hidden_dim, batch_size, reg_para, max_epoch, use_gpu, gpu_id='0', activate_op=1):
    if use_gpu == 1:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    else:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    contract_data, contract_label, num_task, num_class = read_data_from_file(contract_filename, func_filename, graph_label_file, func_label_file, graph_edge_file)
    # data_split = MTDataset_Split(data, label, task_interval, num_class)
    data_split = MTDataset_Split(contract_data, contract_label, num_class)
    dim = contract_data.shape[1]
    traindata, trainlabel, train_task_interval, testdata, testlabel, test_task_interval = data_split.split(train_size)
    # traindata 样本个数*特征维度 trainlabel 1*样本个数 train_task_interval一个任务中样本个数
    error, train_loss_pic, train_loss_pic_f = MTL_GRAPH(traindata, trainlabel, train_task_interval, dim, num_class, num_task, hidden_dim,
                 batch_size, reg_para, max_epoch, testdata, testlabel, test_task_interval, activate_op)
    # zuotu
    y_train_loss = train_loss_pic
    y_train_loss_f = train_loss_pic_f
    y_loss_c = []
    y_acc_c = []
    y_loss_f = []
    y_acc_f = []
    for i in range(150):
        y_loss_c.append(y_train_loss[i][0])
        y_acc_c.append(y_train_loss[i][1])
        y_loss_f.append(y_train_loss_f[i][0])
        y_acc_f.append(y_train_loss_f[i][1])

    x_train_loss = range(len(y_loss_c))
    x_train_loss = [i for i in x_train_loss]
    ax1 = plt.subplot(2, 1, 1)
    ax1.set_title('Contract Detection')
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('accuracy')
    # plt.plot(x_train_loss, y_loss_c, color='blue', linewidth=1, linestyle="solid", label='loss')
    plt.plot(x_train_loss, y_acc_c, color='red', linewidth=1, linestyle="solid")
    plt.legend()

    ax1 = plt.subplot(2, 1, 2)
    # ax1.set_title('Function Detection')
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('loss')
    plt.plot(x_train_loss, y_loss_c, color='blue', linewidth=1, linestyle="solid")
    # plt.plot(x_train_loss, y_acc_f, color='red', linewidth=1, linestyle="solid", label='accuracy')
    plt.legend()

    # ax1 = plt.subplot(2, 2, 2)
    # ax1.set_title('Function Detection')
    # ax1.set_xlabel('epochs')
    # ax1.set_ylabel('accuracy')
    # # plt.plot(x_train_loss, y_loss_f, color='blue', linewidth=1, linestyle="solid", label='loss')
    # plt.plot(x_train_loss, y_acc_f, color='red', linewidth=1, linestyle="solid")
    # plt.legend()
    #
    # ax1 = plt.subplot(2, 2, 4)
    # # ax1.set_title('Function Detection')
    # ax1.set_xlabel('epochs')
    # ax1.set_ylabel('loss')
    # plt.plot(x_train_loss, y_loss_f, color='blue', linewidth=1, linestyle="solid")
    # # plt.plot(x_train_loss, y_acc_f, color='red', linewidth=1, linestyle="solid", label='accuracy')
    # plt.legend()

    fig = plt.gcf()
    # fig.subplots_adjust(left=0.09)
    # fig.subplots_adjust(right=0.99)
    plt.show()
    return error


graph_feature_file = './data/graph_feature.txt'
func_feature_file = './data/func_feature.txt'
graph_label_file = './data/graph_index.txt'
func_label_file = './data/func_index.txt'
graph_edge_file = './data/graph_edge.txt'
max_epoch = 200
use_gpu = 1
gpu_id = '2'
hidden_dim = 300
batch_size = 64
reg_para = 0.2
train_size = 0.7
activate_op = 1
GAT_hidden_dim = 16
F_pie_t = 16
F_pie_c = 16

mean_errors = main_process(graph_feature_file, func_feature_file, graph_label_file, func_label_file, graph_edge_file, train_size, hidden_dim, batch_size, reg_para, max_epoch, use_gpu, gpu_id, activate_op)

print('final test_errors = ', mean_errors)
