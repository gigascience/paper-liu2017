This repository houses the source code from Liu et al. paper.
folder discriptions are as follows:

./cv_score: data and code for the "Overall Performance" part in the paper,which contains five folders named according to the dataset, each contains three files:
1. main.py: generates the auc score of gbdt-space lasso and XGBoost for comparison
2. gbdtLr.py: our method
3. a txt or csv file contains the test data

./train_test_split: data and code for the "Case Study" part in the paper,which contains two folders named according to the dataset, each contains four files:
1. train_test_split.py: generates the roc curve and auc score
2. plot_bar_case.py: generates the Feature coefficient analysis
3. gbdtLr.py: our method
3. a txt or csv file contains the test data

source URL for each data set:
1. Breast Cancer: http://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Original%29
2. diabetes: https://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes
3. fertility: https://archive.ics.uci.edu/ml/datasets/Fertility
4. heart disease: https://archive.ics.uci.edu/ml/datasets/Heart+Disease
5. liver disorder: https://archive.ics.uci.edu/ml/datasets/ILPD+%28Indian+Liver+Patient+Dataset%29