import time
import numpy as np
import warnings
import glob
from script.my_util import *

warnings.filterwarnings('ignore')

source_code_path = '../sourcecode/'
model_path = './model/'
used_file_path = '../used_file_data/'
save_path = './corpus/'


def readData(path, file):
    data = pd.read_csv(path + file, encoding='latin', keep_default_na=False)
    return data


def get_line_flow(project, release):
    data_all = readData(path=used_file_path, file=release + '.csv')
    files_list = data_all.drop_duplicates('filename', keep='first')
    files_list = list(files_list['filename'])
    for files_list_index in range(len(files_list)):
        file_name = files_list[files_list_index]
        print('file_name:', release + '_', file_name, 'progress:', files_list_index + 1, '/', len(files_list))
        folder = (source_code_path + project + '/' + release + '/' + file_name).replace('.java', '')
        file_df = data_all.loc[data_all['filename'] == file_name, :]
        edges = np.loadtxt(folder + '_pdg.txt', delimiter=' ')
        edge_labels = np.loadtxt(folder + '_edge_label.txt')
        try:
            sources = [node[0] for node in edges]
            targets = [node[1] for node in edges]
        except:
            sources = [edges[0]]
            targets = [edges[1]]
            edge_labels = [edge_labels]
        nodes = np.unique(edges.flatten())

        for file_line in range(len(file_df)):
            node = int(file_line + 1)
            line_original = file_df.loc[file_df['line_number'] == node]['code_line'].values
            code_extend = str(line_original[0])
            if node in nodes:
                control_forward = []
                data_forward = []
                control_backward = []
                data_backward = []

                for source_index in range(len(sources)):
                    if (sources[source_index] == node) and (edge_labels[source_index] == 1):
                        try:
                            control_forward.append(
                                file_df.loc[file_df['line_number'] == targets[source_index], 'code_line'].values[0])
                        except:
                            pass
                    if (sources[source_index] == node) and (edge_labels[source_index] == 2):
                        try:
                            data_forward.append(
                                file_df.loc[file_df['line_number'] == targets[source_index], 'code_line'].values[0])
                        except:
                            pass
                    if (targets[source_index] == node) and (edge_labels[source_index] == 1):
                        try:
                            control_backward.append(
                                file_df.loc[file_df['line_number'] == sources[source_index], 'code_line'].values[0])
                        except:
                            pass
                    if (targets[source_index] == node) and (edge_labels[source_index] == 2):
                        try:
                            data_backward.append(
                                file_df.loc[file_df['line_number'] == sources[source_index], 'code_line'].values[0])
                        except:
                            pass
                if len(control_forward) > 0:
                    code_extend = str(code_extend) + '\n' + str(control_forward[0])
                if len(data_forward) > 0:
                    code_extend = str(code_extend) + '\n' + str(data_forward[0])
                if len(control_backward) > 0:
                    code_extend = str(control_backward[0]) + '\n' + str(code_extend)
                if len(data_backward) > 0:
                    code_extend = str(data_backward[0]) + '\n' + str(code_extend)
            data_all.loc[
                (data_all['filename'] == file_name) & (data_all['line_number'] == file_line + 1), 'code_line'] = str(
                code_extend)
    data_all.to_csv(save_path + release + '_line_flow.csv', encoding='latin', na_rep=False, index=False)


def combine_code_line(project, path):
    path1 = glob.glob(path + '\\' + project + '*_line_flow.csv')
    print(path1)
    l = []

    for p in path1:
        csvname = p
        print(csvname)
        csvname = pd.read_csv(p, encoding='latin', keep_default_na=False)
        l.append(csvname)
        demomerge = pd.concat(l, axis=0, ignore_index=True, join='inner')
        demomerge.to_csv(save_path + project + '.csv', index=False, encoding='latin')  # 输出文件


def main():
    total_time = 0
    count = 0
    for project in all_releases.keys():
        for release in all_releases[project]:
            start_time = time.time()
            get_line_flow(project=project, release=release)
            path = '../doc2vec/corpus'
            end_time = time.time()
            elapsed_time = end_time - start_time
            total_time += elapsed_time
            count += 1
            print(f"Project: {project}, Release: {release}, Time: {elapsed_time} seconds")
    combine_code_line(project, path)
    average_time = total_time / count
    print(f"Average Time: {average_time} seconds")


if __name__ == '__main__':
    main()
