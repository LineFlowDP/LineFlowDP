
# This is a repository for "LineFlowDP: A Deep Learning-Based Two-Phase Approach for Line-Level Defect Prediction"
  
Preparation of "README.md" file referenced from DeepLineDP at this [github](https://github.com/awsm-research/DeepLineDP), Thanks!🥰🥰🥰

  
## Environment Configuration
  
1. clone the github repository by using the following command:

		git clone https://github.com/LineFlowDP/LineFlowDP.git

2. download the dataset from this [github](https://github.com/awsm-research/line-level-defect-prediction) and keep it in `datasets/original/`

3. use the following command to install required libraries in conda environment

		conda env create -f requirements.yml
		conda activate LineFlowDP_env

4. install PyTorch library by following the instruction from this [link](https://pytorch.org/) (the installation instruction may vary based on OS and CUDA version)
  
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

### Data Preprocess

Put the downloaded dataset into `datasets/original/` folder, then run **"preprocess_data.py"** to get the preprocessed data, the preprocessed data will be stored in `datasets/preprocessed_data/` folder.

		python preprocess_data.py

### Getting Java Source Files

Run the following command to get the Java source code in csv files, the Java source code will be saved in the `sourcecode/` folder.
		
		python extract_Java_source_code.py

### Extracting Program Dependency Graphs

We use the PropertyGraph tool to extract program dependency graphs (PDG) in this repository  [github](https://github.com/Zanbrachrissik/PropertyGraph), Thanks to them for providing such a handy tool!

PDG files are formatted as *.dot*. Then put the PDG file in the `sourcecode/[PROJECT]/[VERSION]/PDG/` folder, such as `sourcecode/activemq/activemq-5.0.0/PDG/`.

In the PDG file, it will contain data flow and control flow information, in order to refine the nodes and edges, run the following command to extract the information in the dot file, in which the information of the edges will be saved in `sourcecode/[PROJECT]/[VERSION]/[FILE_NAME]_pdg.txt`, to distinguish the data flow edges and refined control flow edges will be saved in `sourcecode/[PROJECT]/[VERSION]/[FILE_NAME]_edge_label.txt`, and refined node types will be saved in a csv file in the `used_file_data` folder.

		python dot_to_txt.py

### Flow Information Extension & Word Embedding

 - `Flow Information Extension`: Run the command in the `do2vec` folder to get the lines of code after the flow information extension, and merge these statements into a corpus for the whole project for subsequent training of the do2vec model.
 
		python flow_information_extension.py
 
 - `Word Embedding`:Run **train_doc2vec_model.py** in the `do2vec` folder to train the corresponding doc2vec model of the project and save it locally, and then run word **embedding.py** in the `do2vec` folder to get the word vectors of the utterances that have been expanded with the stream information and save it as a *.txt* file.
 
		python train_doc2vec_model.py
		python embedding.py

### Get the TUDataset raw file format

In order to realize the training process afterwards, the data obtained after the above steps are needed to processing into a file in TUDataset raw format.

Run the command to get the TUDataset raw files for software projects:

		python TUDataset_raw.py

TUDataset raw file will contain the following six files: DS_A, DS_edge_labels, DS_graph_indicator, DS_graph_labels, DS_node_attributes, DS_node_labels.

 - `DS_A`: The edges in the graphs. (The adjacency matrix of the graph.)
 - `DS_edge_labels`: Indicating that the edge corresponds to the data flow edge or the refined control flow edge.
 - `DS_graph_indicator`: Indicating which graph/file this edge corresponds to.
 - `DS_graph_labels`: Indicating whether the graph/file has defects, 0 for no defects, 1 for defects.
 - `DS_node_attributes`: Word vectors extracted from Doc2Vec model after Flow Information Extension.
 - `DS_node_labels`: Indicating the type of node after refinement.
 
And the TUDataset raw file will be stored in `sript/data/[PROJECT]/[VERSION]/raw/`.
 
Then run **TUDataset_processed.py** to get the processed TUDataset data, it will contain `"data.pt"`,`"pre_filter.pt"` and `"pre_transform.pt"` three files.

		python TUDataset_processed.py

## Training Graph Convolutional Network Model

To leverage the information in our refined program dependency graph, we constructed a Relational Graph Convolutional Network model.



