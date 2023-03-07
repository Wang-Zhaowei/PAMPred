from sklearn.svm import SVC
import numpy as np
import random
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import math
import ipso
#np.set_printoptions(suppress=True)


def get_subset_weights(factors):

    indexes = np.argsort(factors).tolist()[:0]
    t_factors = []
    for i in range(len(factors)):
        if i in indexes:
            t_factors.append(0)
        else:
            t_factors.append((1+factors[i])*math.log(1+factors[i]))
    weights = []
    for factor in t_factors:
        weight = factor/sum(t_factors)
        weights.append(weight)
    return weights


def get_learner_weights(factors, num):
    indexes = np.argsort(factors).tolist()[:num]
    t_factors = []
    for i in range(len(factors)):
        if i in indexes:
            t_factors.append(0)
        else:
            t_factors.append(factors[i])
    weights = []
    for factor in t_factors:
        weight = factor/sum(t_factors)
        weights.append(weight)
    #return weights, sum(t_factors)/len(t_factors)
    return weights


def get_pruning_num(weights, val_probs, y_val):
    pre_auc = 0
    count = 0
    for i in range(len(weights)):
        updated_weights = get_learner_weights(weights, i)
        prob = 0  
        for j in range(len(weights)):
            prob += val_probs[j]*updated_weights[j]
        next_auc = roc_auc_score(y_val[:,-1], prob)
        if i != len(weights)-1:
            if next_auc >= pre_auc:
                pre_auc = next_auc
                count = 0
            else:
                count += 1
                if count == 2:
                    return i-2
        else:
            if next_auc >= pre_auc:
                return len(weights)-2
            elif next_auc < pre_auc and count == 0:
                return len(weights)-2
            elif next_auc < pre_auc and count == 1:
                return len(weights)-3


def get_bce(y_true, prob):
   ce_loss = y_true*(np.log(prob))+(1-y_true)*(np.log(1-prob))
   total_ce = np.sum(ce_loss)
   bce = - total_ce/y_true.shape[0]
   return 1/bce


def ensem_learning_val(train_data, train_label, prop, test_data, test_label):
    factors = []
    val_probs = []
    val_num = int(train_data.shape[0]*prop)
    
    aac_train = train_data[:,:20]
    qso_train = train_data[:,20:70]
    cks_train = train_data[:,70:220]
    ctd_train = train_data[:,220:]
    train1 = np.concatenate((aac_train,cks_train),axis=1)
    train2 = np.concatenate((aac_train,ctd_train),axis=1)
    train3 = np.concatenate((qso_train,cks_train),axis=1)
    train4 = np.concatenate((qso_train,ctd_train),axis=1)
    train5 = train_data

    aac_test = test_data[:,:20]
    qso_test = test_data[:,20:70]
    cks_test = test_data[:,70:220]
    ctd_test = test_data[:,220:]
    test1 = np.concatenate((aac_test,cks_test),axis=1)
    test2 = np.concatenate((aac_test,ctd_test),axis=1)
    test3 = np.concatenate((qso_test,cks_test),axis=1)
    test4 = np.concatenate((qso_test,ctd_test),axis=1)
    test5 = test_data

    train_datasets = [train1,train2,train3,train4,train5]
    test_datasets = [test1,test2,test3,test4,test5]
    for data in train_datasets:
        X_train, X_val, y_train, y_val = data[val_num:,:], data[:val_num,:], train_label[val_num:,:], train_label[:val_num,:]
        print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)
        
        clf1 = lgb.LGBMClassifier()
        clf1.fit(X_train, y_train[:,-1])
        prob1 = clf1.predict_proba(X_val)[:,-1]
        val_probs.append(prob1)
        auc1 = roc_auc_score(y_val[:,-1], prob1)
        factors.append(auc1)
        
        clf2 = SVC(probability=True)
        clf2.fit(X_train, y_train[:,-1])
        prob2 = clf2.predict_proba(X_val)[:,-1]
        val_probs.append(prob2)
        auc2 = roc_auc_score(y_val[:,-1], prob2)
        factors.append(auc2)
        
    optimasation = ipso.get_optimasation_function(val_probs, y_val[:,-1])
    lb, ub = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    weights = []
    seed_range = np.arange(0,1)
    # Optimization loop
    for s in seed_range:
        np.random.seed(s)
        random.seed(s)
        optx, fopt = ipso.pso(optimasation, lb, ub, swarmsize=100, seed=s, maxiter=100)
        weights = ipso.extract_weight(optx)
        #clf_weights = optx
        print("Round {}: Completed".format(s+1))
    
    pruning_num1 = get_pruning_num(weights, val_probs, y_val)
    clf_weights = get_learner_weights(weights, pruning_num1)
    print(clf_weights)
    subset_prob = 0
    for i in range(len(clf_weights)):
        subset_prob += val_probs[i]*clf_weights[i]

    subset_bce = get_bce(y_val[:,-1], subset_prob)

    test_probs = []
    count = 0
    for data_index in range(len(test_datasets)):
        train_data = train_datasets[data_index]
        test_data = test_datasets[data_index]

        for i in range(2):
            if i == 0:
                clf = lgb.LGBMClassifier()
                clf.fit(train_data, train_label[:,-1])
                test_prob = clf.predict_proba(test_data)[:,-1]
            else:
                clf = SVC(probability=True)
                clf.fit(train_data, train_label[:,-1])
                test_prob = clf.predict_proba(test_data)[:,-1]
            count += 1
            test_probs.append(test_prob)

    ensem_test_prob = 0
    for i in range(len(test_probs)):
        ensem_test_prob += test_probs[i]*clf_weights[i]
    return ensem_test_prob, subset_prob, subset_bce