import numpy as np
import os
import warnings
from itertools import chain
from script.my_util import *

warnings.filterwarnings("ignore")

save_path = './data/'
source_code_path = '../sourcecode/'
used_file_data_path = '../used_file_data/'


def readData(path, file):
    data = pd.read_csv(path + file, encoding='latin', keep_default_na=False)
    return data


def toTUDA(line_data: pd.DataFrame, project, release, method='lineflow'):
    save_pre = save_path + project + '/' + release + '/' + 'raw/'
    if not os.path.exists(save_pre):
        os.makedirs(save_pre)

    graph_labels = []
    node_labels = []
    graph_indicators = []
    graph_indicators_i = 1
    node_attributes = []
    edge_labels = []
    DS_A = []
    max = 0
    flag_frist_dfg_is_null = False

    file_data = line_data.drop_duplicates('filename', keep='first')
    files_list = list(file_data['filename'])

    for file_data_index in file_data.index:
        file_name = file_data.loc[file_data_index, 'filename']
        current_file_data = line_data.loc[line_data['filename'] == file_name]
        folder = (source_code_path + project + '/' + release + '/' + file_name).replace('.java', '')

        if not file_data.loc[file_data_index, 'file-label']:
            graph_labels.append(0)
        else:
            graph_labels.append(1)

        node_label = list(line_data.loc[line_data['filename'] == file_name, 'node_label'])
        node_labels.append(node_label)

        if method == 'lineflow':
            word2vec_path = source_code_path + project.lower() + '/' + release + '/' + file_name.replace('.java',
                                                                                                         '') + '_lineflow.doc2vec'
        elif method == 'linenoflow':
            word2vec_path = source_code_path + project.lower() + '/' + release + '/' + file_name.replace('.java',
                                                                                                         '') + '_linenoflow.doc2vec'
        elif method == 'noflow':
            word2vec_path = source_code_path + project.lower() + '/' + release + '/' + file_name.replace('.java',
                                                                                                         '') + '_noflow.doc2vec'

        vector = np.loadtxt(word2vec_path, delimiter=',')
        if file_data_index == 0:
            node_attributes = vector
        else:
            node_attributes = np.vstack((node_attributes, vector))

        temp = [int(graph_indicators_i) for i in range(len(current_file_data))]
        graph_indicators.append(temp)
        graph_indicators_i += 1

        dfg_path = source_code_path + project + '/' + release + '/' + file_name.replace('.java',
                                                                                        '') + '_pdg.txt'
        max_line = len(current_file_data)
        dfg_line = np.loadtxt(dfg_path) + max
        max = max + max_line
        if file_data_index == 0:
            if len(dfg_line) == 0:
                flag_frist_dfg_is_null = True
                DS_A = [0, 0]
            else:
                DS_A = dfg_line
        else:
            try:
                DS_A = np.vstack((DS_A, dfg_line))
            except:
                DS_A = DS_A
        edge_label_path = source_code_path + project + '/' + release + '/' + file_name.replace('.java',
                                                                                               '') + '_edge_label.txt'
        edge_label = np.loadtxt(edge_label_path)
        if file_data_index == 0:
            edge_labels = edge_label
        else:
            edge_labels = np.hstack((edge_labels, edge_label))

    node_labels = list(chain.from_iterable(node_labels))
    graph_indicators = list(chain.from_iterable(graph_indicators))
    if flag_frist_dfg_is_null:
        DS_A = DS_A[1:]

    print(' graph_labels：', len(graph_labels))
    print(' node_labels：', len(node_labels))
    print(' node_attributes：', len(node_attributes))
    print(' graph_indicators：', len(graph_indicators))
    print(' edge_labels：', len(edge_labels))
    print(' DS_A：', len(DS_A))

    np.savetxt(save_pre + release + '_graph_labels.txt', graph_labels, fmt='%d')
    np.savetxt(save_pre + release + '_node_labels.txt', node_labels, fmt='%d')
    np.savetxt(save_pre + release + '_node_attributes.txt', node_attributes, fmt='%.8f', delimiter=',')
    np.savetxt(save_pre + release + '_graph_indicator.txt', graph_indicators, fmt='%d')
    np.savetxt(save_pre + release + '_A.txt', DS_A, fmt='%d', delimiter=',')
    np.savetxt(save_pre + release + '_edge_labels.txt', edge_labels, fmt='%d', delimiter=',')
    print('-' * 50, release, '_done', '-' * 50)


def main():
    for project in all_releases.keys():
        for release in all_releases[project]:
            line_data = readData(used_file_data_path, release + '.csv')
            # toTUDA(line_data=line_data, project=project, release=release, method='lineflow')
            # toTUDA(line_data=line_data, project=project, release=release, method='linenoflow')
            toTUDA(line_data=line_data, project=project, release=release, method='noflow')


if __name__ == '__main__':
    main()
