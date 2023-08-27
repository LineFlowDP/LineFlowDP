
# This is a repository for "LineFlowDP: A Deep Learning-Based Two-Phase Approach for Line-Level Defect Prediction"
  
Preparation of "README.md" file referenced from DeepLineDP[github](https://github.com/awsm-research/DeepLineDP), Thanks!🥰🥰🥰

  
## Environment Configuration
  
The environment we used has been exported to the "environment.yaml" file, from which you can create the environment necessary to replicate the results of this paper.
  
## Datasets

The datasets are obtained from Wattanakriengkrai et. al. The datasets contain 32 software releases across 9 software projects. The datasets that we used in our experiment can be found in this [github](https://github.com/awsm-research/line-level-defect-prediction).

The file-level datasets (in the File-level directory) contain the following columns

 - `File`: A file name of source code
 - `Bug`: A label indicating whether source code is clean or defective
 - `SRC`: A content in source code file

The line-level datasets (in the Line-level directory) contain the following columns
 - `File`: A file name of source code
 - `Line_number`: A line number where source code is modified
 - `SRC`: An actual source code that is modified

For each software project, we use the oldest release to train LineFlowDP models. The subsequent release is used as validation sets. The other releases are used as test sets.

For example, there are 5 releases in ActiveMQ (e.g., R1, R2, R3, R4, R5), R1 is used as training set, R2 is used as validation set, and R3 - R5 are used as test sets.

## Data Prepation

