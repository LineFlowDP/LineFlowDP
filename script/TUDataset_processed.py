from script.my_util import *

data_path = './data/'


def tudataset(project, version):
    dataset_train = MYDataset(root=(data_path + project), name=version, use_node_attr=True)
    return dataset_train


if __name__ == '__main__':
    error_version = []
    for project in list(all_releases.keys()):
        cur_releases = all_releases[project]
        for release in cur_releases:
            try:
                tudataset(project=project, version=release)
                print(release, 'done')
            except:
                error_version.append(release)
                print(release, 'error')
    print(error_version)
