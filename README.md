Please look at the proposal pdf for more information. We aim to evaluate each feature interaction detection/scoring method as follows:
1. We finetune some models (BERT, RoBERTa) for a sentiment classification task (imdb, sst-2).
2. We take the test set, calculate the scores using each method (Integrated Directed Gradients, Integrated Hessians, Asymmetric Shapely Interaction Value, T-NID, Shapely Interaction Value)
3. We iteratively remove the top k features (k = 1 - 5) in the test set as identified by these methods and calculate AOPC and LOR.
4. We iteratively remove the top k features in the train set as identified in these methods, finetune fresh models on these sets, and observe the drop in prediction confidence by measuring AUC.
