import os.path
from script.my_util import *
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import pandas as pd
import random
import warnings

warnings.filterwarnings('ignore')

file_path = './corpus/'
model_path = './model/'

if not os.path.exists(model_path):
    os.mkdir(model_path)


class DocumentDataset(object):
    def __init__(self, data: pd.DataFrame, column):
        document = data[column].apply(self.preprocess)
        self.documents = [TaggedDocument(text, [index]) for index, text in document.iteritems()]

    def preprocess(self, document):
        # return preprocess_string(remove_stopwords(document))
        return document

    def __iter__(self):
        for document in self.documents:
            yield document

    def tagged_documents(self, shuffle=False):
        if shuffle:
            random.shuffle(self.documents)
        return self.documents


# read data
def readData(path, file):
    data = pd.read_csv(path + file, encoding='latin')
    return data


def train_doc2vec(project, method='lineflow'):
    if method =='lineflow':
        data = readData(file_path, project + '.csv')
    elif method =='noflow':
        data = readData(file_path, project + '_noflow.csv')
    elif method == 'linenoflow':
        data = readData(file_path, project + '_linenoflow.csv')
    data['code_line'] = data['code_line'].astype(str)
    document_dataset = DocumentDataset(data, 'code_line')
    docVecModel = Doc2Vec(min_count=1,
                          window=5,
                          vector_size=100,
                          sample=1e-4,
                          negative=5,
                          workers=2,
                          )
    docVecModel.build_vocab(document_dataset.tagged_documents())
    print('training......')
    docVecModel.train(document_dataset.tagged_documents(shuffle=False),
                      total_examples=docVecModel.corpus_count,
                      epochs=20)
    if method == 'lineflow':
        docVecModel.save(model_path + project + '_lineflow.d2v')
    elif method == 'linenoflow':
        docVecModel.save(model_path + project + '_linenoflow.d2v')
    elif method == 'noflow':
        docVecModel.save(model_path + project + '_noflow.d2v')
    print('done!')


def main():
    for project in all_releases.keys():
        train_doc2vec(project=project, method='noflow')
        print('-' * 50, project + '_done', '-' * 50)


if __name__ == '__main__':
    main()
