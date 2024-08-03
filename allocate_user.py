import random
import numpy as np
import sys

random.seed(1)
np.random.seed(1)

train_label_path = sys.argv[1]
train_label_file = open(train_label_path, 'r')
train_qid_list = []
for line in train_label_file:
    train_qid_list.append(line.strip().split(':')[0])
train_label_file.close()

num_user = sys.argv[2] # simulated user num
num_qid = len(train_qid_list)

query_distribution = []
for _ in range(num_user):
    probs = np.zeros(num_qid)
    non_zero_indices = np.random.choice(num_qid, int(num_qid * 0.5), replace=False)
    non_zero_probs = np.random.rand(len(non_zero_indices))
    probs[non_zero_indices] = non_zero_probs
    probs /= np.sum(probs)
    query_distribution.append(probs)

user_qid_matrix = np.zeros((num_user, num_qid))

betas = num_user -1 - np.array(range(num_user))
user_counts = np.array([1.25 ** beta for beta in betas])
user_counts = user_counts / sum(user_counts)

session_num = int(sys.argv[3]) #simulated session num
user_session_num = np.random.multinomial(session_num, user_counts)

for i in range(num_user):
    user_qid_matrix[i] = user_qid_matrix[i] + np.random.multinomial(int(user_session_num[i]), query_distribution[i])


qid_initlist_dic = {}
initial_ranked_list_path = sys.argv[4]
train_initlist_file = open(initial_ranked_list_path, 'r')
for line in train_initlist_file:
    qid = line.strip().split(':')[0]
    initlist = line.strip().split(':')[1]
    qid_initlist_dic[qid] = initlist
train_initlist_file.close()

user_train_initlist_file = open('user_train_s{}.init_list'.format(session_num), 'w')
qid_session_count = np.zeros(num_qid)
for j in range(num_qid):
    for i in range(num_user):
        session = user_qid_matrix[i][j]
        qid = train_qid_list[j]
        for _ in range(int(session)):
            user_train_initlist_file.write(qid+'_'+str(int(qid_session_count[j]))+','+str(i)+':'+qid_initlist_dic[qid]+'\n')
            qid_session_count[j] += 1
user_train_initlist_file.close()

