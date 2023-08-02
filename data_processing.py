#!/usr/bin/env python
#_*_coding:utf-8_*_

from collections import Counter
import math, random
import numpy as np
from sklearn.cluster import KMeans


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


def construct_kmer():
	ntarr = ('D', 'E', 'K', 'R', 'A', 'N', 'C', 'Q', 'G', 'H', 'I', 'L', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V')

	kmerArray = []

	for n in range(20):
		str1 = ntarr[n]
		for m in range(20):
			str2 = str1 + ntarr[m]
			kmerArray.append(str2)
	return kmerArray


def GTPC(sequence, **kw):
	group = {
		'alphaticr': 'GAVLMI',
		'aromatic': 'FYW',
		'postivecharger': 'KRH',
		'negativecharger': 'DE',
		'uncharger': 'STCPNQ'
	}

	groupKey = group.keys()
	baseNum = len(groupKey)
	triple = [g1+'.'+g2+'.'+g3 for g1 in groupKey for g2 in groupKey for g3 in groupKey]

	index = {}
	for key in groupKey:
		for aa in group[key]:
			index[aa] = key

	code = []
	myDict = {}
	for t in triple:
		myDict[t] = 0

	sum = 0
	for j in range(len(sequence) - 3 + 1):
		myDict[index[sequence[j]]+'.'+index[sequence[j+1]]+'.'+index[sequence[j+2]]] = myDict[index[sequence[j]]+'.'+index[sequence[j+1]]+'.'+index[sequence[j+2]]] + 1
		sum = sum +1

	if sum == 0:
		for t in triple:
			code.append(0)
	else:
		for t in triple:
			code.append(myDict[t]/sum)

	return code


# single nucleic ggap
def g_gap_single(seq, ggaparray, g):
	# seq length is fix =23

	rst = np.zeros((400))
	for i in range(len(seq) - 1 - g):
		str1 = seq[i]
		str2 = seq[i + 1 + g]
		idx = ggaparray.index(str1 + str2)
		rst[idx] += 1

	for j in range(len(ggaparray)):
		rst[j] = rst[j] / (len(seq) - 1 - g)  # l-1-g

	return rst

def GGAP(sequence, **kw):
	kmerArray = construct_kmer()
	ggap = g_gap_single(sequence, kmerArray, 1)
	return ggap.tolist()


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


def Rvalue(aa1, aa2, AADict, Matrix):
	return sum([(Matrix[i][AADict[aa1]] - Matrix[i][AADict[aa2]]) ** 2 for i in range(len(Matrix))]) / len(Matrix)
	

def PAAC(sequence, lambdaValue=4, w=0.05, **kw):
	dataFile = './dataset/PAAC.txt'
	with open(dataFile) as f:
		records = f.readlines()
	AA = ''.join(records[0].rstrip().split()[1:])
	AADict = {}
	for i in range(len(AA)):
		AADict[AA[i]] = i
	AAProperty = []
	AAPropertyNames = []
	for i in range(1, len(records)):
		array = records[i].rstrip().split() if records[i].rstrip() != '' else None
		AAProperty.append([float(j) for j in array[1:]])
		AAPropertyNames.append(array[0])

	AAProperty1 = []
	for i in AAProperty:
		meanI = sum(i) / 20
		fenmu = math.sqrt(sum([(j-meanI)**2 for j in i])/20)
		AAProperty1.append([(j-meanI)/fenmu for j in i])

	encodings = []
	header = ['#']
	for aa in AA:
		header.append('Xc1.' + aa)
	for n in range(1, lambdaValue + 1):
		header.append('Xc2.lambda' + str(n))
	encodings.append(header)

	code = []
	theta = []
	for n in range(1, lambdaValue + 1):
		theta.append(
			sum([Rvalue(sequence[j], sequence[j + n], AADict, AAProperty1) for j in range(len(sequence) - n)]) / (
			len(sequence) - n))
	myDict = {}
	for aa in AA:
		myDict[aa] = sequence.count(aa)
	code = code + [myDict[aa] / (1 + w * sum(theta)) for aa in AA]
	code = code + [(w * j) / (1 + w * sum(theta)) for j in theta]
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

	groups = [group1, group2, group3]
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

	groups = [group1, group2, group3]
	property = (
	'hydrophobicity_PRAM900101', 'hydrophobicity_ARGP820101', 'hydrophobicity_ZIMJ680101', 'hydrophobicity_PONP930101',
	'hydrophobicity_CASG920101', 'hydrophobicity_ENGD860101', 'hydrophobicity_FASG890101', 'normwaalsvolume',
	'polarity', 'polarizability', 'charge', 'secondarystruct', 'solventaccess')

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

	groups = [group1, group2, group3]
	property = (
	'hydrophobicity_PRAM900101', 'hydrophobicity_ARGP820101', 'hydrophobicity_ZIMJ680101', 'hydrophobicity_PONP930101',
	'hydrophobicity_CASG920101', 'hydrophobicity_ENGD860101', 'hydrophobicity_FASG890101', 'normwaalsvolume',
	'polarity', 'polarizability', 'charge', 'secondarystruct', 'solventaccess')

	code = []
	for p in property:
		code = code + CTDD_Count(group1[p], sequence) + CTDD_Count(group2[p], sequence) + CTDD_Count(group3[p], sequence)

	return code


def get_dataset(posi_file, nega_file):
	posi_samples = []
	with open(posi_file, "r") as lines:
		for data in lines:
			line = data.strip()
			if line[0] == '>':
				name = line[1:]
			else:
				sequence = line
				aac_fea = AAC(sequence)
				gga_fea = GGAP(sequence)
				qso_fea = QSOrder(sequence)
				gtp_fea = GTPC(sequence)
				paac_fea = PAAC(sequence)
				c_fea = CTDC(sequence)
				t_fea = CTDT(sequence)
				d_fea = CTDD(sequence)
				ctd_fea = c_fea + t_fea + d_fea
				posi_sample = aac_fea + gga_fea + qso_fea + gtp_fea + paac_fea + ctd_fea + [1]
				posi_samples.append(posi_sample)

	nega_samples = []
	with open(nega_file, "r") as lines:
		for data in lines:
			line = data.strip()
			if line[0] == '>':
				name = line[1:]
			else:
				sequence = line
				aac_fea = AAC(sequence)
				gga_fea = GGAP(sequence)
				qso_fea = QSOrder(sequence)
				gtp_fea = GTPC(sequence)
				paac_fea = PAAC(sequence)
				c_fea = CTDC(sequence)
				t_fea = CTDT(sequence)
				d_fea = CTDD(sequence)
				ctd_fea = c_fea + t_fea + d_fea
				nega_sample = aac_fea + gga_fea + qso_fea  + gtp_fea + paac_fea + ctd_fea + [0]
				nega_samples.append(nega_sample)

	random.shuffle(posi_samples)
	random.shuffle(nega_samples)
	return np.array(posi_samples), np.array(nega_samples)


def spliting_by_clustering(nega_train_data, group_num):
	nega_train_data = np.array(nega_train_data)
	kmeans = KMeans(n_clusters=group_num)
	kmeans.fit(nega_train_data)
	labels = kmeans.labels_
	nega_new_train_data = {}
	for i in range(group_num):
		group_list = []
		nega_new_train_data[i] = group_list
		
	count = 0
	for label in labels:
		nega_new_train_data[label].append(nega_train_data[count])
		count = count + 1
	return nega_new_train_data


def sampling_from_clusters(nega_new_train_data, posi_num, nega_num):
	X_nega_train = []
	nega_train_data ={}
	nega_train_num = 0
	rest_nega = []
	for key in nega_new_train_data:
		sample_num = round(len(nega_new_train_data[key])/nega_num * posi_num)
		group_samples = nega_new_train_data[key][:sample_num]
		nega_train_data[key] = nega_new_train_data[key][sample_num:]
		nega_train_num += len(nega_train_data[key])
		rest_nega += nega_train_data[key]
		X_nega_train = X_nega_train + group_samples
	rest_nega = np.array(rest_nega)
	return X_nega_train,rest_nega,nega_train_num