import time
import numpy as np
from my_util import *
import os

preprocessed_file_path = '../datasets/preprocessed_data/'
saved_file_path = '../used_file_data/'
source_code_path = '../sourcecode/'


def readData(path, file):
    data = pd.read_csv(path + file, encoding='latin', keep_default_na=False)
    return data


def get_line_num(E):
    result = []
    e = E[E.rfind('<'):E.rfind('>')][1:]
    if '...' in e:
        start_point = int(e[0:e.rfind('...')])
        end_point = int(e[e.rfind('.'):][1:])
        e = list(range(start_point, end_point + 1))
        result = e
    else:
        e = int(e)
        result.append(e)
    return result


def get_all_pdg(project, release):
    data_all = readData(path=preprocessed_file_path, file=release + '.csv')
    data = data_all.drop_duplicates('filename', keep='first')

    for index in data.index:
        file_name = str(data.loc[index, 'filename'])
        folder = (source_code_path + project + '/' + release + '/' + file_name).replace('.java', '')
        java_name = folder.split('/')[-1]
        pdg_path = source_code_path + project + '/' + release + '/PDG/' + java_name
        if not os.path.exists(folder):
            os.makedirs(folder)
        if not os.path.exists(saved_file_path):
            os.makedirs(saved_file_path)
        print(' Java file:', release + '_' + file_name, 'Progress:', index + 1, '/', len(data_all))

        if '.java' not in file_name:
            continue

        node_lines = {}
        edges = []
        try:
            with open(pdg_path + '_pdg.dot', encoding='utf-8') as f:
                contents = f.read().split('\n')
                for i in range(len(contents)):
                    content = contents[i]
                    if ('style = ' in content) & ('label = ' in content) \
                            & ('fillcolor = aquamarine' not in content) & (' -> ' not in content):
                        while ('fillcolor' not in content) | ('shape' not in content) | (not content.endswith('];')):
                            i += 1
                            if i >= len(contents):
                                break
                            content = content + '' + contents[i]
                        node = content.split(' ')[0]
                        lines = get_line_num(content)
                        node_lines[node] = lines
                        if 'shape = box' in content:
                            node_label = 1
                        elif 'shape = ellipse' in content:
                            node_label = 2
                        elif 'shape = diamond' in content:
                            node_label = 3
                        else:
                            node_label = 4
                        for line in lines:
                            data_all.loc[(data_all['filename'] == file_name) & (
                                    data_all['line_number'] == line), 'node_label'] = node_label

                    if ('->' in content) & ('[style =' in content) & ('label=' in content) & (content.endswith('"];')):
                        edge_source = content.split(' ')[0]
                        edge_target = content.split(' ')[2]
                        if 'style = dotted' in content:
                            edge_label_flag = 1
                        elif 'style = solid' in content:
                            edge_label_flag = 2
                        elif 'style = bold' in content:
                            edge_label_flag = 3
                        else:
                            edge_label_flag = 4
                        edges.append([edge_source, edge_target, edge_label_flag])

        except:
            with open(pdg_path + '_pdg.dot', encoding='ansi') as f:
                contents = f.read().split('\n')
                for i in range(len(contents)):
                    content = contents[i]
                    if ('style = ' in content) & ('label = ' in content) \
                            & ('fillcolor = aquamarine' not in content) & (' -> ' not in content):
                        while ('fillcolor' not in content) | ('shape' not in content) | (not content.endswith('];')):
                            i += 1
                            if i >= len(contents):
                                break
                            content = content + '' + contents[i]
                        node = content.split(' ')[0]
                        lines = get_line_num(content)
                        node_lines[node] = lines
                        if 'shape = box' in content:
                            node_label = 1
                        elif 'shape = ellipse' in content:
                            node_label = 2
                        elif 'shape = diamond' in content:
                            node_label = 3
                        else:
                            node_label = 4
                        for line in lines:
                            data_all.loc[(data_all['filename'] == file_name) & (
                                    data_all['line_number'] == line), 'node_label'] = node_label

                    if ('->' in content) & ('[style =' in content) & ('label=' in content) & (content.endswith('"];')):
                        edge_source = content.split(' ')[0]
                        edge_target = content.split(' ')[2]
                        if 'style = dotted' in content:
                            edge_label_flag = 1
                        elif 'style = solid' in content:
                            edge_label_flag = 2
                        elif 'style = bold' in content:
                            edge_label_flag = 3
                        else:
                            edge_label_flag = 4
                        edges.append([edge_source, edge_target, edge_label_flag])

        f.close()
        source = []
        target = []
        edge_label = []
        for edge in edges:
            if (edge[0] in node_lines.keys()) & (edge[1] in node_lines.keys()):
                for i in node_lines.get(edge[0]):
                    for j in node_lines.get(edge[1]):
                        source.append(i)
                        target.append(j)
                        edge_label.append(edge[2])
        pdg = np.vstack((source, target)).T
        np.savetxt(folder + '_pdg.txt', pdg, fmt='%d')
        np.savetxt(folder + '_edge_label.txt', edge_label, fmt='%d')

    data_all.loc[:, 'node_label'].fillna(4, axis=0, inplace=True)
    data_all.to_csv(saved_file_path + release + '.csv', encoding='latin', index=False)


def main():
    total_time = 0
    count = 0
    for project in all_releases.keys():
        for release in all_releases[project]:
            start_time = time.time()
            get_all_pdg(project=project, release=release)
            end_time = time.time()
            elapsed_time = end_time - start_time
            total_time += elapsed_time
            count += 1
            print(f"Project: {project}, Release: {release}, Time: {elapsed_time} seconds")
    average_time = total_time / count
    print(f"Average Time: {average_time} seconds")


if __name__ == '__main__':
    main()
