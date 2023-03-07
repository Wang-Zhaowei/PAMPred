import data_processing
from sklearn.model_selection import KFold
import numpy as np
import ensemble_learning
import test_scores as score
import random


def PAMPred():
    posi_file = './dataset/Positive dataset.fasta'
    unlabeled_file = './dataset/Negative dataset.fasta'
    posi_samples, unlabeled_samples = data_processing.get_dataset(posi_file, unlabeled_file)
    posi_num = len(posi_samples)
    unlabeled_train_samples, unlabeled_cv_samples = data_processing.generating_unlabeled_cv_data(unlabeled_samples, posi_num)
    posi_samples = np.array(posi_samples)
    unlabeled_cv_samples = np.array(unlabeled_cv_samples)
    ave_acc = 0
    ave_prec = 0
    ave_recall = 0
    ave_mcc = 0
    ave_f1_score = 0
    ave_auc = 0
    ave_aupr = 0
    
    metric_list = []
    n_fold = 10
    cluster_num = 23
    fold_index = 0
    Kfold = KFold(n_splits=n_fold, shuffle=True)
    for train_index,test_index in Kfold.split(posi_samples):
        posi_train, posi_test = posi_samples[train_index], posi_samples[test_index]
        unlabeled_train, unlabeled_test = unlabeled_cv_samples[train_index], unlabeled_cv_samples[test_index]
        data_test = np.concatenate((posi_test[:,:-2], unlabeled_test[:,:-2]))
        y_train = np.concatenate((posi_train[:,-2:], unlabeled_train[:,-2:]))
        y_test = np.concatenate((posi_test[:,-2:], unlabeled_test[:,-2:]))
        
        test_probs = []
        subset_probs = []
        subset_bces = []
        nega_num = len(unlabeled_train_samples)
        nega_train_samples = data_processing.spliting_by_clustering(unlabeled_train_samples, cluster_num)
        for count in range(5):
            nega_train, nega_train_samples, nega_num= data_processing.sampling_from_clusters(nega_train_samples, posi_train.shape[0], nega_num)
            train_samples = posi_train.tolist()+nega_train
            random.shuffle(train_samples)
            train_samples = np.array(train_samples)
            data_train = train_samples[:,:-2]
            y_train = train_samples[:,-2:]
            print(data_train.shape, y_train.shape)
            print(data_test.shape, y_test.shape)

            y_test_prob, subset_prob, subset_auc = ensemble_learning.ensem_learning_val(data_train, y_train, 0.2, data_test, y_test)

            test_probs.append(y_test_prob)
            subset_probs.append(subset_prob)
            subset_bces.append(subset_auc)
            tp, fp, tn, fn, acc, precision, recall, MCC, f1_score, AUC, AUPR = score.calculate_performace(y_test_prob, y_test[:,-1])
            print('\nSubset'+str(count+1)+'\ntp = \t'+ str(tp) + '\t fp = \t'+ str(fp) + '\t tn = \t'+ str(tn)+ '\t fn = \t'+ str(fn)+'\n  Acc = \t', acc, '\n  prec = \t', precision, '\n  recall = \t', recall, '\n  MCC = \t', MCC, '\n  f1_score = \t', f1_score, '\n  AUC = \t', AUC, '\n  AUPR = \t', AUPR)
        metric_list.append([acc, precision, recall, MCC, f1_score, AUC, AUPR])
        
        subset_weights = ensemble_learning.get_subset_weights(subset_bces)

        final_prob = 0
        for i in range(len(test_probs)):
            final_prob += test_probs[i]*subset_weights[i]
        
        tp, fp, tn, fn, acc, prec, recall, MCC, f1_score, AUC, AUPR = score.calculate_performace(final_prob, y_test[:,-1])
        print('\n------------------ Fold ', fold_index+1, '----------------------\ntp = \t'+ str(tp) + '\t fp = \t'+ str(fp) + '\t tn = \t'+ str(tn)+ '\t fn = \t'+ str(fn)+'\n  Acc = \t', acc, '\n  prec = \t', prec, '\n  recall = \t', recall, '\n  MCC = \t', MCC, '\n  f1_score = \t', f1_score, '\n  AUC = \t', AUC, '\n  AUPR = \t', AUPR)
        fold_index += 1

        ave_acc += acc
        ave_prec += prec
        ave_recall += recall
        ave_mcc += MCC
        ave_f1_score += f1_score
        ave_auc += AUC
        ave_aupr += AUPR

    fw = open('./Experimental results/Ten-fold cross-validation results.txt', 'a+')
    ave_acc /= n_fold
    ave_prec /= n_fold
    ave_recall /= n_fold
    ave_mcc /= n_fold
    ave_f1_score /= n_fold
    ave_auc /= n_fold
    ave_aupr /= n_fold
    ####
    print('\n Acc = \t'+ str(ave_acc)+'\n prec = \t'+ str(ave_prec)+ '\n recall = \t'+str(ave_recall)+ '\n MCC = \t'+str(ave_mcc)+'\n f1_score = \t'+str(ave_f1_score)+'\n AUC = \t'+ str(ave_auc) + '\n AUPR =\t'+str(ave_aupr)+'\n')
    fw.write('Acc\t'+ str(ave_acc)+'\tprec\t'+ str(ave_prec)+ '\trecall\t'+str(ave_recall)+ '\tMCC\t'+str(ave_mcc)+'\tf1_score\t'+str(ave_f1_score)+'\tAUC\t'+ str(ave_auc) + '\tAUPR\t'+str(ave_aupr)+'\n')
    fw.close()


PAMPred()
