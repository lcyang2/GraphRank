import sys
sys.path.append('..')

import os
import os.path as osp
import time
import pickle
import copy

import torch
import torch.nn.functional as F


import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm

import sklearn
from xgboost import XGBClassifier



def make_y_error(out, y_class):
    y_error = torch.zeros(size=[y_class.shape[0]], dtype=torch.int64)
    for i in tqdm(range(y_class.shape[0])):
        if out[i].max(-1)[1].item() != y_class[i]:
            y_error[i] = 1
        else :
            y_error[i] = 0
    return y_error

def atrc(all_failure_mask, select_test_idx, faliure_num):
    
    TRC_list = np.zeros(shape=faliure_num)
    detach_now = 0
    for i in (range(faliure_num)):
        if all_failure_mask[select_test_idx[i]]:
            detach_now += 1
        TRC_list[i] = detach_now / (i+1)
    
    TRC_select = []
    for i in range(faliure_num):
        TRC_select.append(TRC_list[i])
    ATRC = np.sum(TRC_select) / len(TRC_select)
    
    return ATRC
    

def make_test(select):
    datasets = ["data_ogbn-products", "data_Reddit", "data_Flickr"]
    models = ["/EnGCN", "/SIGN", "/GraphSAGE", "/ClusterGCN"]
    muta_datasets = ["products", "reddit", "flickr"]
    muta_models = ["engcn", "sign", "graphsage", "clustergcn"]
    
    dataset_name = datasets[select[0]]
    model_name = models[select[1]]
        
    print("dataset : {}, model : {}".format(dataset_name, model_name))
    

    out = torch.load("./my_data/" + dataset_name + model_name +'/out/out')  
    out_pred = out.argmax(axis=1)
    
    out_mean = torch.torch.zeros_like(torch.load("./my_data/" + dataset_name + model_name +'/dropout/out_9'))
    for i in tqdm(range(0, 10)):      # get probabilistic output attributes
        out_mean += torch.load("./my_data/" + dataset_name + model_name +'/dropout/out_{}'.format(i))
    out_mean /= 10
    
    logpx = F.log_softmax(out_mean, dim=1).detach().numpy()
    uncertainty = -np.sum(np.multiply(logpx, np.exp(logpx)), axis = 1)
    uncertainty = torch.tensor(uncertainty).unsqueeze(dim=1)

    x_out = torch.load("./my_data/" + dataset_name + '/data_x/x_out')     # get graph node attributes
    
    split_masks = torch.load("./my_data/" + dataset_name + '/split/my_split')
    
    deg = torch.load("./my_data/" + dataset_name + '/deg/deg')    # get graph structure attributes
    
    y_class = torch.load("./my_data/" + dataset_name + '/y/y_class')
    y_error = torch.load("./my_data/" + dataset_name + model_name + '/y_error')
    print(1- y_error[split_masks['test']].sum() / split_masks['test'].sum())
    

    HE = torch.load("./my_data/" + dataset_name + model_name + '/out/HE')
    x_HE = torch.load("./my_data/" + dataset_name + '/data_x/x_HE')
    

    out = F.softmax(out, dim=1)
    x_out = F.softmax(x_out, dim=1)
    DeepGini = 1 - torch.sum(torch.pow(out, 2), dim=1)
    DeepGini = DeepGini.unsqueeze(dim=1)
    
    num_node = out.shape[0]


    out_last = torch.zeros(size=(out.shape[0],0), dtype=torch.float32)
    
    out_last = torch.cat((out_last, out), dim=1)
    out_last = torch.cat((out_last, x_out), dim=1)
    out_last = torch.cat((out_last, HE), dim=1)
    out_last = torch.cat((out_last, x_HE), dim=1)
    out_last = torch.cat((out_last, deg), dim=1)
    out_last = torch.cat((out_last, uncertainty), dim=1) 
    out_last = torch.cat((out_last, DeepGini), dim=1)
   
    out_agg = copy.deepcopy(out_last)

 
    T = torch.load("./my_data/" + dataset_name +'/edge/T') 
    print(len(T))
    
    for i in tqdm(range(len(T))):
        if len(T[i]) == 0:
            continue
        feature_tmp = out_last[T[i]]
        out_agg[i] += torch.sum(feature_tmp, dim=0)

    for i in tqdm(range(len(T))):
        out_agg[i] = out_agg[i] / (len(T[i])+1)
        
    torch.save(out_agg, "./my_data/" + dataset_name + model_name + '/out_agg')
    print(out_agg.min(), out_agg.max())
    out_agg = torch.load("./my_data/" + dataset_name + model_name + '/out_agg')
    
    feature_np = torch.cat((out_last, out_agg), dim=1)

    test_failure_mask = y_error[split_masks['test']]
    all_failure_mask = y_error
    print(all_failure_mask.shape)
    print(test_failure_mask.shape)

    label_np = np.zeros(shape=num_node)
    for i in range(num_node):
        if y_error[i] == 1:
            label_np[i] = 1
        else:
            label_np[i] = 0

    feature_np = torch.tensor(feature_np, dtype=torch.float)     
    label_np = torch.from_numpy(label_np)


    ite_num = 10
    failure_number = int(test_failure_mask.sum())
    budget_one = int(failure_number / ite_num)
    print(failure_number, budget_one)

    train_idx = split_masks["train"].nonzero(as_tuple=False).view(-1)
    valid_idx = split_masks["valid"].nonzero(as_tuple=False).view(-1)
    test_idx = split_masks["test"].nonzero(as_tuple=False).view(-1)

    select_test_idx = np.array([], dtype=np.int16)
    valid_idx_tmp = copy.deepcopy(valid_idx)
    test_idx_tmp = copy.deepcopy(test_idx)
    

    for i in range(ite_num):
        x_train = feature_np[valid_idx]
        y_train = label_np[valid_idx]
        x_test = feature_np[test_idx]
        y_test = label_np[test_idx]

        print(i, 'start train')
        
        model = XGBClassifier()
       
        model.fit(x_train, y_train)
        y_pred_test = model.predict_proba(x_test)[:, 1]
        print(y_pred_test.shape)
        rank_idx = y_pred_test.argsort()[::-1].copy()
        
        if i != (ite_num-1):
            select_idx = test_idx[rank_idx[:budget_one]]
            select_test_idx = np.concatenate((select_test_idx, select_idx), axis=0)
        else:
            select_idx = test_idx[rank_idx]
            select_test_idx = np.concatenate((select_test_idx, select_idx), axis=0)
            
        print(select_test_idx.shape)
        print(np.unique(select_test_idx).shape)
        
        test_idx = np.setdiff1d(test_idx, select_idx)
        valid_idx = np.concatenate((valid_idx, select_idx), axis=0)
        print(test_idx.shape)
        print(valid_idx.shape)


    valid_idx = valid_idx_tmp
    test_idx = test_idx_tmp
    print(valid_idx.shape, test_idx.shape)
    
    tree_atrc = atrc(all_failure_mask, select_test_idx, failure_number)

    seed = 0
    path_save = './result_ite/save/{}/{}_{}_{}.pt'.format(seed, dataset_name[5:], model_name[1:], 'xgb')

    if not os.path.exists('./result_ite/save/{}'.format(seed)):
        os.mkdir('./result_ite/save/{}'.format(seed))
    pickle.dump([select_test_idx], open(path_save, 'wb'))
    
    atrc_list = [dataset_name[5:] + '_' + model_name[1:], tree_atrc]
    columns = ['subject', 'xgb']
    df_apfd = pd.DataFrame(data=[atrc_list], columns=columns)
    df_apfd.to_csv('result_ite/GraphRank_{}_{}.csv'.format(seed, 'xgb'), mode='a', header=False, index=False)
        


if __name__ == "__main__":
    for select in [[2,0],[2,1],[2,2],[2,3]]:
        make_test(select)
        print()
