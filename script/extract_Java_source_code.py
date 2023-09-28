import time
from my_util import *
import os

used_file_path = '../datasets/preprocessed_data/'
source_code_path = '../sourcecode/'


def readData(path, file):
    data = pd.read_csv(path + file, encoding='latin', keep_default_na=False)
    return data


def get_java_code(project, release):
    # 读取数据
    data = readData(used_file_path, release + '.csv')
    java_files = data.drop_duplicates('filename', keep='last')
    java_files = list(java_files['filename'])
    for java_file in java_files:
        code = ''
        java_df = data[data['filename'] == java_file]
        for index in java_df.index:
            code = code + str(java_df.loc[index, 'code_line']) + '\n'
        file_name = str(java_df.loc[index, 'filename'])
        folder = (source_code_path + project + '/' + release + '/' + file_name).replace('.java', '')
        if not os.path.exists(folder):
            os.makedirs(folder)
        with open(source_code_path + project + '/' + release + '/' + file_name, 'w', encoding='utf-8') as f:
            f.write(code.strip())
            f.close()


def main():
    total_time = 0
    count = 0
    for project in all_releases.keys():
        for release in all_releases[project]:
            start_time = time.time()
            get_java_code(project=project, release=release)
            end_time = time.time()
            elapsed_time = end_time - start_time
            total_time += elapsed_time
            count += 1
            print(f"Project: {project}, Release: {release}, Time: {elapsed_time} seconds")
    average_time = total_time / count
    print(f"Average Time: {average_time} seconds")


if __name__ == '__main__':
    main()
