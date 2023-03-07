# PAMPred
Combining the weighted ensemble learning with particle swarm optimization to predict plant antimicrobial non-conventional peptides

## Introduction
Motivation:

Accumulating evidences suggested that non-conventional peptides play a crucial role in biological immune regulation, growth and development stages. Recently, the study of antimicrobial activity of plant non-conventional peptides has attracted significant attentions in the bioinformatics field. However, there is no computational method proposed to identify antimicrobial non-conventional peptides.

Results:

In this study, a novel weighted ensemble learning framework combined with an improved particle swarm optimization (PSO) algorithm, named PAMPred, was presented for antimicrobial non-conventional peptide prediction. To address the existing imbalance classification problem, a cluster-based resampling method was developed to construct multiple balanced subsets. Moreover, the improved particle swarm optimization (PSO) algorithm and the dynamic ensemble pruning approach were applied to optimise the weights and combination of basic learners in the framework to boost the predictive capability.

## Dataset

In this paper, the experimentally verified potential antimicrobial non-conventional peptides of the maize inbred line B73 derived from the studies of Professor Wu Liuji and her colleagues. For the negative samples, we firstly downloaded the RNA sequences of the maize inbred line B73 including microRNAs (miRNAs), long non-coding RNAs (lncRNAs), etc from the Ensembl Plants website (http://ftp.ensemblgenomes.org/pub/plants/release-41/fasta/zea_mays). Then, the small ORFs between the start and the stop codons upon the ncRNAs, which have been considered not to be translated into peptides, were mined and translated into small peptides.

## Usage

~~~python
python PAMPred.py
~~~

## Citation

Z Wang, J Meng\* and Y Luan. Combining the weighted ensemble learning with particle swarm optimization to predict plant antimicrobial non-conventional peptides. Applied Soft Computing, 2023.
