# PAMPred
PAMPred: A hierarchical evolutionary ensemble framework for identifying plant antimicrobial peptides

## Introduction
Motivation:

Antimicrobial peptides (AMPs) play a crucial role in plant immune regulation, growth and development stages, which have attracted significant attentions in recent years. As the wet-lab experiments are laborious and cost-prohibitive, it is indispensable to develop computational methods to discover novel plant AMPs accurately.

Results:

In this study, we presented a hierarchical evolutionary ensemble framework, named PAMPred, which consisted of a multi-level heterogeneous architecture to identify plant AMPs. Specifically, to address the existing class imbalance problem, a cluster-based resampling method was adopted to build multiple balanced subsets. Then, several peptide features including sequence information-based and physicochemical properties-based features were fed into the different types of basic learners to increase the ensemble diversity. For boosting the predictive capability of PAMPred, the improved particle swarm optimization (PSO) algorithm and dynamic ensemble pruning strategy were used to optimise the weights at different levels adaptively. Furthermore, extensive ten-fold cross-validation and independent testing experiments demonstrated that PAMPred achieved excellent prediction performance and generalization ability, and outperformed the state-of-the-art methods. The results also indicated that the proposed method could serve as an effective auxiliary tool to identify plant AMPs, which is conducive to explore the immune regulatory mechanism of plants.

## Dataset

In this study, the plant AMP sequences (positive samples) are collected from PlantPepDB database. For the negative samples, there are few experimentally validated plant non-AMPs reported in literatures and public repositories. Therefore, in addition to PlantPepDB, we downloaded plant non-AMP sequences from Uniprot database. To avoid the classification bias caused by sequence homology and redundancy, the CD-HIT program  with a threshold of 0.9 was applied on both the positive and negative datasets. After preprocessing, the final dataset included 379 positive samples and 4115 negative samples, which was further divided into the benchmark training dataset and the independent testing dataset to evaluate the prediction performance and generalization ability of PAMPred and other available methods.

## Usage

~~~python
python PAMPred.py
~~~

## Citation

Z Wang, J Meng\* et al. PAMPred: A hierarchical evolutionary ensemble framework for identifying plant antimicrobial peptides. Computers in Biology and Medicine, 2023.
