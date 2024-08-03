import numpy as np
import json
import sys

# get relevance
label_path = sys.argv[1]
label_file = open(label_path, "r")
label_dic = {}
train_qid_list = []
for line in label_file:
    if(line != ''):
        qid = line.strip().split(":")[0]
        train_qid_list.append(qid)
        labels = line.strip().split(":")[1].strip().split(" ")[:10]
        label_dic[qid] = [int(l) for l in labels]
label_file.close()

num_user = int(sys.argv[2])
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

# click_file = open('user_exam_train.labels' ,'r')
click_file_path = sys.argv[3]
click_file = open(click_file_path, 'r')
query_user_dic = {}
query_clicks = {}
click_pos = np.zeros(position)
user_counts = [0 for _ in range(num_user)]
for line in click_file:
    query_info = line.strip().split(':')[0]
    qid = query_info.strip().split('_')[0]
    user = int(query_info.strip().split(',')[1])
    clicks = line.strip().split(':')[1].strip().split(' ')

    for i in range(len(clicks)):
        if clicks[i] == '1':
            click_pos[i] += 1
    clicks = [int(c) for c in clicks]
    user_counts[user] += 1
    if qid not in query_user_dic.keys():
        query_user_dic[qid] = [user]
        query_clicks[qid] = [clicks]
    else:
        query_user_dic[qid].append(user)
        query_clicks[qid].append(clicks)
click_file.close()

# print(user_counts)
# print(sum(user_counts))

# click-through rate
click_pos = click_pos / sum(user_counts)
print('click_pos:')
print(click_pos)

# ave_exams for IPSPBM
ave_exams_IPSPBM = []
for i in range(position):
    exams = user_exams[:, i]
    # print(exams)
    ave_exam = np.inner(exams, user_counts) / sum(user_counts)
    ave_exams_IPSPBM.append(ave_exam)
ave_exams_IPSPBM = np.array(ave_exams_IPSPBM)
# print(ave_exams_IPSPBM)

# obtain query_ave_exams for straightforward
query_ave_pw_dic = {}
count = 0
for k in query_user_dic.keys():
    users = query_user_dic[k]
    ave_exams = np.zeros(position)
    for u in users:
        ave_exams += user_exams[int(u)]
    ave_exams = ave_exams / len(users)
    if count < 100:
        print(ave_exams)
        count += 1
    ave_pw = 1.0 / ave_exams
    query_ave_pw_dic[k] = list(ave_pw)

naive_file = open('naive.labels', 'w')
IPS_PBM_file = open('IPS-PBM.labels', 'w')
user_aware_file = open('user-aware.labels', 'w')
straightforward_file = open('straightforward.labels', 'w')

for qid in train_qid_list:
    if qid not in query_user_dic.keys():
        clicks = [str(0) for _ in range(10)]
        naive_file.write(qid + ':' + ' '.join(clicks) + '\n')
        IPS_PBM_file.write(qid + ':' + ' '.join(clicks) + '\n')
        user_aware_file.write(qid + ':' + ' '.join(clicks) + '\n')
        straightforward_file.write(qid + ':' + ' '.join(clicks) + '\n')
    else:
        clicks = query_clicks[qid]
        users = query_user_dic[qid]
        labels = label_dic[qid] # skyline
        num_session = len(users)

        position_for_this_session = len(clicks[0])
        raw_click = np.zeros(position_for_this_session)
        straightforward = np.zeros(position_for_this_session)
        for i in range(num_session):
            click = np.array(clicks[i])
            raw_click = raw_click + click
            straightforward += click / user_exams[users[i]][:position_for_this_session]
        ctr = raw_click / num_session
        IPS_PBM = ctr / ave_exams_IPSPBM[:position_for_this_session]
        user_aware = ctr * np.array(query_ave_pw_dic[qid])[:position_for_this_session]
        straightforward = straightforward / num_session

        ctr = [str(c) for c in ctr]
        IPS_PBM = [str(p) for p in IPS_PBM]
        user_aware = [str(u) for u in user_aware]
        straightforward = [str(u) for u in straightforward]

        naive_file.write(qid + ':' + ' '.join(ctr) + '\n')
        IPS_PBM_file.write(qid + ':' + ' '.join(IPS_PBM) + '\n')
        user_aware_file.write(qid + ':' + ' '.join(user_aware) + '\n')
        straightforward_file.write(qid + ':' + ' '.join(straightforward) + '\n')




