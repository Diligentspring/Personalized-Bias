import sys
import random
import numpy as np

random.seed(1)

initlist_path = sys.argv[1]
epsilon = 0.1

# 得到relevance
label_path = sys.argv[2]
label_file = open(label_path, "r")
label_dic = {}
for line in label_file:
    if(line != ''):
        qid = line.strip().split(":")[0]
        labels = line.strip().split(":")[1].strip().split(" ")
        for i in range(len(labels)):
            label_dic[qid] = labels
label_file.close()
# print(label_dic)

# 得到user_exams
num_user = int(sys.argv[3])
position = 10
alphas = 1 / (np.array(range(position)) + 1)

user_exams = np.zeros((num_user, position))
if num_user == 5:
    user_etas = [2.5, 2, 1, 0.8, 0]
elif num_user == 10:
    user_etas = [2.5, 2, 1.8, 1.5, 1.2, 1, 0.8, 0.5, 0.2, 0]
elif num_user == 20:
    user_etas = [2.5, 2.4, 2.2, 2, 1.9, 1.8, 1.6, 1.5, 1.4, 1.2, 1.1, 1, 0.9, 0.8, 0.6, 0.5, 0.4, 0.2, 0.1, 0]

for i in range(num_user):
    exams = np.power(alphas, user_etas[i])
    user_exams[i] = exams
# print(user_exams)

def generate_clicks(user, relevances):
    exams = user_exams[int(user)]
    clicks = []
    for i in range(len(relevances)):
        click_prob = exams[i] * (epsilon + (1 - epsilon) * 0.25 * relevances[i])
        if random.random() < click_prob:
            clicks.append(1)
        else:
            clicks.append(0)
    return clicks

initlist_file = open(initlist_path, 'r')
gen_label_file = open('user_click.labels'.format(session_num), 'w')

for line in initlist_file:
    if line != '':
        query_info = line.strip().split(':')[0]
        gen_label_file.write(query_info+':')
        qid = query_info.strip().split('_')[0]
        user = query_info.strip().split(',')[1]
        docs = line.strip().split(':')[1].strip().split(' ')
        labels = [int(l) for l in label_dic[qid]]

        click_list = generate_clicks(user, labels)
        click_list = [str(c) for c in click_list]
        gen_label_file.write(' '.join(click_list) + '\n')
initlist_file.close()
gen_label_file.close()