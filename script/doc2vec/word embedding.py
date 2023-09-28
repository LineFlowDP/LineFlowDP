import time
from gensim.models import Doc2Vec, doc2vec
import numpy as np
import warnings
from script.my_util import *

warnings.filterwarnings('ignore')

source_code_path = '../sourcecode/'
model_path = './model/'
used_file_path = '../used_file_data/'


def readData(path, file):
    data = pd.read_csv(path + file, encoding='latin')
    return data


def get_line_doc2vec(project, release, method='lineflow'):
    data_all = readData(path=used_file_path, file=release + '.csv')
    files_list = data_all.drop_duplicates('filename', keep='first')
    files_list = list(files_list['filename'])
    if method == 'lineflow':
        model = Doc2Vec.load(model_path + project + '_lineflow.d2v')
    elif method == 'linenoflow':
        model = Doc2Vec.load(model_path + project + '_linenoflow.d2v')
    elif method == 'noflow':
        model = Doc2Vec.load(model_path + project + '_noflow.d2v')
    print(model)
    doc_vec_index = 0
    for files_list_index in range(len(files_list)):
        file_name = files_list[files_list_index]
        print('file_name:', release + '_', file_name, 'progress:', files_list_index + 1, '/', len(files_list))
        folder = (source_code_path + project + '/' + release + '/' + file_name).replace('.java', '')
        java_name = folder.split('/')[-1]
        file_df = data_all.loc[data_all['filename'] == file_name, :]
        file_vectors = [model.docvecs[index] for index in range(doc_vec_index, (doc_vec_index + len(file_df)))]
        doc_vec_index += len(file_df)

        if method == 'lineflow':
            np.savetxt(folder + '_lineflow.doc2vec', file_vectors, fmt='%.8f', delimiter=',')
        elif method == 'linenoflow':
            np.savetxt(folder + '_linenoflow.doc2vec', file_vectors, fmt='%.8f', delimiter=',')
        elif method == 'noflow':
            np.savetxt(folder + '_noflow.doc2vec', file_vectors, fmt='%.8f', delimiter=',')


def main():
    total_time = 0
    count = 0
    for project in all_releases.keys():
        for release in all_releases[project]:
            start_time = time.time()
            get_line_doc2vec(project=project, release=release, method='lineflow')
            # get_line_doc2vec(project=project, release=release, method='linenoflow')
            # get_line_doc2vec(project=project, release=release, method='noflow')
            end_time = time.time()
            elapsed_time = end_time - start_time
            total_time += elapsed_time
            count += 1
            print(f"Project: {project}, Release: {release}, Time: {elapsed_time} seconds")
    average_time = total_time / count
    print(f"Average Time: {average_time} seconds")


if __name__ == '__main__':
    main()
