#!/usr/bin/env python
#_*_coding:utf-8_*_

from collections import Counter
import math, random
import numpy as np
from sklearn.cluster import MiniBatchKMeans


def AAC(sequence, **kw):
	# AA = kw['order'] if kw['order'] != None else 'ACDEFGHIKLMNPQRSTVWY'
	AA = 'ACDEFGHIKLMNPQRSTVWY'

	count = Counter(sequence)
	for key in count:
		count[key] = count[key]/len(sequence)
	code = []
	for aa in AA:
		code.append(count[aa])
	return code


def QSOrder(sequence, nlag=5, w=0.1, **kw):
	dataFile = './dataset/Schneider-Wrede.txt'
	dataFile1 = './dataset/Grantham.txt'

	AA = 'ACDEFGHIKLMNPQRSTVWY'
	AA1 = 'ARNDCQEGHILKMFPSTWYV'

	DictAA = {}
	for i in range(len(AA)):
		DictAA[AA[i]] = i

	DictAA1 = {}
	for i in range(len(AA1)):
		DictAA1[AA1[i]] = i

	with open(dataFile) as f:
		records = f.readlines()[1:]
	AADistance = []
	for i in records:
		array = i.rstrip().split()[1:] if i.rstrip() != '' else None
		AADistance.append(array)
	AADistance = np.array(
		[float(AADistance[i][j]) for i in range(len(AADistance)) for j in range(len(AADistance[i]))]).reshape((20, 20))

	with open(dataFile1) as f:
		records = f.readlines()[1:]
	AADistance1 = []
	for i in records:
		array = i.rstrip().split()[1:] if i.rstrip() != '' else None
		AADistance1.append(array)
	AADistance1 = np.array(
		[float(AADistance1[i][j]) for i in range(len(AADistance1)) for j in range(len(AADistance1[i]))]).reshape(
		(20, 20))

	code = []
	arraySW = []
	arrayGM = []
	for n in range(1, nlag + 1):
		arraySW.append(
			sum([AADistance[DictAA[sequence[j]]][DictAA[sequence[j + n]]] ** 2 for j in range(len(sequence) - n)]))
		arrayGM.append(sum(
			[AADistance1[DictAA1[sequence[j]]][DictAA1[sequence[j + n]]] ** 2 for j in range(len(sequence) - n)]))
	myDict = {}
	for aa in AA1:
		myDict[aa] = sequence.count(aa)
	for aa in AA1:
		code.append(myDict[aa] / (1 + w * sum(arraySW)))
	for aa in AA1:
		code.append(myDict[aa] / (1 + w * sum(arrayGM)))
	for num in arraySW:
		code.append((w * num) / (1 + w * sum(arraySW)))
	for num in arrayGM:
		code.append((w * num) / (1 + w * sum(arrayGM)))
	return code


def generateGroupPairs(groupKey):
    gPair = {}
    for key1 in groupKey:
        for key2 in groupKey:
            gPair[key1+'.'+key2] = 0
    return gPair


def cksaagp(sequence, gap = 5, **kw):
	group = {'alphaticr': 'GAVLMI',
				'aromatic': 'FYW',
				'postivecharger': 'KRH',
				'negativecharger': 'DE',
				'uncharger': 'STCPNQ'}

	AA = 'ARNDCQEGHILKMFPSTWYV'

	groupKey = group.keys()

	index = {}
	for key in groupKey:
		for aa in group[key]:
			index[aa] = key

	gPairIndex = []
	for key1 in groupKey:
		for key2 in groupKey:
			gPairIndex.append(key1+'.'+key2)
			
	code = []
	for g in range(gap + 1):
		gPair = generateGroupPairs(groupKey)
		sum = 0
		for p1 in range(len(sequence)):
			p2 = p1 + g + 1
			if p2 < len(sequence) and sequence[p1] in AA and sequence[p2] in AA:
				gPair[index[sequence[p1]]+'.'+index[sequence[p2]]] = gPair[index[sequence[p1]]+'.'+index[sequence[p2]]] + 1
				sum = sum + 1

		if sum == 0:
			for gp in gPairIndex:
				code.append(0)
		else:
			for gp in gPairIndex:
				code.append(gPair[gp] / sum)
	return code


def CTDC_Count(seq1, seq2):
	sum = 0
	for aa in seq1:
		sum = sum + seq2.count(aa)
	return sum


def CTDC(sequence, **kw):
	group1 = {
		'hydrophobicity_PRAM900101': 'RKEDQN',
		'hydrophobicity_ARGP820101': 'QSTNGDE',
		'hydrophobicity_ZIMJ680101': 'QNGSWTDERA',
		'hydrophobicity_PONP930101': 'KPDESNQT',
		'hydrophobicity_CASG920101': 'KDEQPSRNTG',
		'hydrophobicity_ENGD860101': 'RDKENQHYP',
		'hydrophobicity_FASG890101': 'KERSQD',
		'normwaalsvolume': 'GASTPDC',
		'polarity':        'LIFWCMVY',
		'polarizability':  'GASDT',
		'charge':          'KR',
		'secondarystruct': 'EALMQKRH',
		'solventaccess':   'ALFCGIVW'
	}
	group2 = {
		'hydrophobicity_PRAM900101': 'GASTPHY',
		'hydrophobicity_ARGP820101': 'RAHCKMV',
		'hydrophobicity_ZIMJ680101': 'HMCKV',
		'hydrophobicity_PONP930101': 'GRHA',
		'hydrophobicity_CASG920101': 'AHYMLV',
		'hydrophobicity_ENGD860101': 'SGTAW',
		'hydrophobicity_FASG890101': 'NTPG',
		'normwaalsvolume': 'NVEQIL',
		'polarity':        'PATGS',
		'polarizability':  'CPNVEQIL',
		'charge':          'ANCQGHILMFPSTWYV',
		'secondarystruct': 'VIYCWFT',
		'solventaccess':   'RKQEND'
	}

	property = (
	'hydrophobicity_PRAM900101', 'hydrophobicity_ARGP820101', 'hydrophobicity_ZIMJ680101', 'hydrophobicity_PONP930101',
	'hydrophobicity_CASG920101', 'hydrophobicity_ENGD860101', 'hydrophobicity_FASG890101', 'normwaalsvolume',
	'polarity', 'polarizability', 'charge', 'secondarystruct', 'solventaccess')

	code = []
	for p in property:
		c1 = CTDC_Count(group1[p], sequence) / len(sequence)
		c2 = CTDC_Count(group2[p], sequence) / len(sequence)
		c3 = 1 - c1 - c2
		code = code + [c1, c2, c3]
	return code

def CTDT(sequence, **kw):
	group1 = {
		'hydrophobicity_PRAM900101': 'RKEDQN',
		'hydrophobicity_ARGP820101': 'QSTNGDE',
		'hydrophobicity_ZIMJ680101': 'QNGSWTDERA',
		'hydrophobicity_PONP930101': 'KPDESNQT',
		'hydrophobicity_CASG920101': 'KDEQPSRNTG',
		'hydrophobicity_ENGD860101': 'RDKENQHYP',
		'hydrophobicity_FASG890101': 'KERSQD',
		'normwaalsvolume': 'GASTPDC',
		'polarity':        'LIFWCMVY',
		'polarizability':  'GASDT',
		'charge':          'KR',
		'secondarystruct': 'EALMQKRH',
		'solventaccess':   'ALFCGIVW'
	}
	group2 = {
		'hydrophobicity_PRAM900101': 'GASTPHY',
		'hydrophobicity_ARGP820101': 'RAHCKMV',
		'hydrophobicity_ZIMJ680101': 'HMCKV',
		'hydrophobicity_PONP930101': 'GRHA',
		'hydrophobicity_CASG920101': 'AHYMLV',
		'hydrophobicity_ENGD860101': 'SGTAW',
		'hydrophobicity_FASG890101': 'NTPG',
		'normwaalsvolume': 'NVEQIL',
		'polarity':        'PATGS',
		'polarizability':  'CPNVEQIL',
		'charge':          'ANCQGHILMFPSTWYV',
		'secondarystruct': 'VIYCWFT',
		'solventaccess':   'RKQEND'
	}
	group3 = {
		'hydrophobicity_PRAM900101': 'CLVIMFW',
		'hydrophobicity_ARGP820101': 'LYPFIW',
		'hydrophobicity_ZIMJ680101': 'LPFYI',
		'hydrophobicity_PONP930101': 'YMFWLCVI',
		'hydrophobicity_CASG920101': 'FIWC',
		'hydrophobicity_ENGD860101': 'CVLIMF',
		'hydrophobicity_FASG890101': 'AYHWVMFLIC',
		'normwaalsvolume': 'MHKFRYW',
		'polarity':        'HQRKNED',
		'polarizability':  'KMHFRYW',
		'charge':          'DE',
		'secondarystruct': 'GNPSD',
		'solventaccess':   'MSPTHY'
	}

	code = []
	aaPair = [sequence[j:j + 2] for j in range(len(sequence) - 1)]
	for p in property:
		c1221, c1331, c2332 = 0, 0, 0
		for pair in aaPair:
			if (pair[0] in group1[p] and pair[1] in group2[p]) or (pair[0] in group2[p] and pair[1] in group1[p]):
				c1221 = c1221 + 1
				continue
			if (pair[0] in group1[p] and pair[1] in group3[p]) or (pair[0] in group3[p] and pair[1] in group1[p]):
				c1331 = c1331 + 1
				continue
			if (pair[0] in group2[p] and pair[1] in group3[p]) or (pair[0] in group3[p] and pair[1] in group2[p]):
				c2332 = c2332 + 1
		code = code + [c1221/len(aaPair), c1331/len(aaPair), c2332/len(aaPair)]

	return code

def CTDD_Count(aaSet, sequence):
	number = 0
	for aa in sequence:
		if aa in aaSet:
			number = number + 1
	cutoffNums = [1, math.floor(0.25 * number), math.floor(0.50 * number), math.floor(0.75 * number), number]
	cutoffNums = [i if i >=1 else 1 for i in cutoffNums]

	code = []
	for cutoff in cutoffNums:
		myCount = 0
		for i in range(len(sequence)):
			if sequence[i] in aaSet:
				myCount += 1
				if myCount == cutoff:
					code.append((i + 1) / len(sequence) * 100)
					break
		if myCount == 0:
			code.append(0)
	return code


def CTDD(sequence, **kw):
	group1 = {
		'hydrophobicity_PRAM900101': 'RKEDQN',
		'hydrophobicity_ARGP820101': 'QSTNGDE',
		'hydrophobicity_ZIMJ680101': 'QNGSWTDERA',
		'hydrophobicity_PONP930101': 'KPDESNQT',
		'hydrophobicity_CASG920101': 'KDEQPSRNTG',
		'hydrophobicity_ENGD860101': 'RDKENQHYP',
		'hydrophobicity_FASG890101': 'KERSQD',
		'normwaalsvolume': 'GASTPDC',
		'polarity':        'LIFWCMVY',
		'polarizability':  'GASDT',
		'charge':          'KR',
		'secondarystruct': 'EALMQKRH',
		'solventaccess':   'ALFCGIVW'
	}
	group2 = {
		'hydrophobicity_PRAM900101': 'GASTPHY',
		'hydrophobicity_ARGP820101': 'RAHCKMV',
		'hydrophobicity_ZIMJ680101': 'HMCKV',
		'hydrophobicity_PONP930101': 'GRHA',
		'hydrophobicity_CASG920101': 'AHYMLV',
		'hydrophobicity_ENGD860101': 'SGTAW',
		'hydrophobicity_FASG890101': 'NTPG',
		'normwaalsvolume': 'NVEQIL',
		'polarity':        'PATGS',
		'polarizability':  'CPNVEQIL',
		'charge':          'ANCQGHILMFPSTWYV',
		'secondarystruct': 'VIYCWFT',
		'solventaccess':   'RKQEND'
	}
	group3 = {
		'hydrophobicity_PRAM900101': 'CLVIMFW',
		'hydrophobicity_ARGP820101': 'LYPFIW',
		'hydrophobicity_ZIMJ680101': 'LPFYI',
		'hydrophobicity_PONP930101': 'YMFWLCVI',
		'hydrophobicity_CASG920101': 'FIWC',
		'hydrophobicity_ENGD860101': 'CVLIMF',
		'hydrophobicity_FASG890101': 'AYHWVMFLIC',
		'normwaalsvolume': 'MHKFRYW',
		'polarity':        'HQRKNED',
		'polarizability':  'KMHFRYW',
		'charge':          'DE',
		'secondarystruct': 'GNPSD',
		'solventaccess':   'MSPTHY'
	}

	property = (
	'hydrophobicity_PRAM900101', 'hydrophobicity_ARGP820101', 'hydrophobicity_ZIMJ680101', 'hydrophobicity_PONP930101',
	'hydrophobicity_CASG920101', 'hydrophobicity_ENGD860101', 'hydrophobicity_FASG890101', 'normwaalsvolume',
	'polarity', 'polarizability', 'charge', 'secondarystruct', 'solventaccess')

	code = []
	for p in property:
		code = code + CTDD_Count(group1[p], sequence) + CTDD_Count(group2[p], sequence) + CTDD_Count(group3[p], sequence)

	return code


def get_dataset(posi_file, unlabeled_file):
	posi_samples = []
	with open(posi_file, "r") as lines:
		for data in lines:
			line = data.strip()
			if line[0] == '>':
				name = line[1:]
			else:
				sequence = line
				aac_fea = AAC(sequence)
				qso_fea = QSOrder(sequence)
				cks_fea = cksaagp(sequence)
				c_fea = CTDC(sequence)
				t_fea = CTDT(sequence)
				d_fea = CTDD(sequence)
				ctd_fea = c_fea + t_fea + d_fea
				posi_sample = aac_fea + qso_fea + cks_fea + ctd_fea + [0, 1]
				posi_samples.append(posi_sample)

	unlabeled_samples = []
	with open(unlabeled_file, "r") as lines:
		for data in lines:
			line = data.strip()
			if line[0] == '>':
				name = line[1:]
			else:
				sequence = line
				aac_fea = AAC(sequence)
				qso_fea = QSOrder(sequence)
				cks_fea = cksaagp(sequence)
				c_fea = CTDC(sequence)
				t_fea = CTDT(sequence)
				d_fea = CTDD(sequence)
				ctd_fea = c_fea + t_fea + d_fea
				unlabeled_sample = aac_fea + qso_fea + cks_fea + ctd_fea + [1, 0]
				unlabeled_samples.append(unlabeled_sample)

	random.shuffle(posi_samples)
	random.shuffle(unlabeled_samples)
	return posi_samples, unlabeled_samples


def generating_unlabeled_cv_data(unlabeled_data, posi_num):
	unlabeled_train_data = unlabeled_data[posi_num:]
	unlabeled_cv_data = unlabeled_data[:posi_num]
	return unlabeled_train_data, unlabeled_cv_data


def generating_subset(posi_train_data, unlabelled_train_data):
	random.shuffle(unlabelled_train_data)
	nega_samples = unlabelled_train_data[:len(posi_train_data)]
	rest_samples = unlabelled_train_data[len(posi_train_data):]
	train_data = posi_train_data.tolist() + nega_samples
	random.shuffle(train_data)
	return np.array(train_data), rest_samples
	

def spliting_by_clustering(unlabelled_train_data, group_num):
	unlabelled_train_data = np.array(unlabelled_train_data)
	kmeans = MiniBatchKMeans(n_clusters=group_num)
	kmeans.fit(unlabelled_train_data)
	labels = kmeans.labels_
	unlabelled_new_train_data = {}
	for i in range(group_num):
		group_list = []
		unlabelled_new_train_data[i] = group_list
		
	count = 0
	for label in labels:
		unlabelled_new_train_data[label].append(unlabelled_train_data[count])
		count = count + 1
	return unlabelled_new_train_data
        

def sampling_from_clusters(unlabelled_new_train_data, posi_num, nega_num):
	X_nega_train = []
	nega_train_data ={}
	nega_train_num = 0
	for key in unlabelled_new_train_data:
		sample_num = int(len(unlabelled_new_train_data[key])/nega_num * posi_num)
		group_samples = unlabelled_new_train_data[key][:sample_num]
		nega_train_data[key] = unlabelled_new_train_data[key][sample_num:]
		nega_train_num += len(nega_train_data[key])
		X_nega_train = X_nega_train + group_samples
	return X_nega_train,nega_train_data,nega_train_num
