# This file contain an example how to perform multi-task learning using the
# BiLSTM-CNN-CRF implementation.
# In the datasets variable, we specify two datasets: POS-tagging (unidep_pos) and conll2000_chunking.
# The network will then train jointly on both datasets.
# The network can on more datasets by adding more entries to the datasets dictionary.

from __future__ import print_function
import os
import logging
import sys
from neuralnets.BiLSTM import BiLSTM
from util.preprocessing import perpareDataset, loadDatasetPickle

from keras import backend as K
import argparse

# :: Change into the working dir of the script ::
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# :: Logging level ::
loggingLevel = logging.INFO
logger = logging.getLogger()
logger.setLevel(loggingLevel)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(loggingLevel)
formatter = logging.Formatter('%(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


######################################################
#
# Data preprocessing
#
######################################################
datasets = {
    'ampersand':
        {'columns': {1:'tokens', 2:'full_BIO'},
         'label': 'full_BIO',
         'evaluate': True,
         'commentSymbol': None},
    'ampersand1':
        {'columns': {1:'tokens', 3:'bt_BIO'},
         'label': 'bt_BIO',
         'evaluate': True,
         'commentSymbol': None},
    'ampersand2':
        {'columns': {1:'tokens', 4:'ds'},
         'label': 'ds',
         'evaluate': True,
         'commentSymbol': None},
}

embeddingsPath = 'komninos_english_embeddings.gz' #Word embeddings by Levy et al: https://levyomer.wordpress.com/2014/04/25/dependency-based-word-embeddings/

# :: Prepares the dataset to be used with the LSTM-network. Creates and stores cPickle files in the pkl/ folder ::
pickleFile = perpareDataset(embeddingsPath, datasets)


######################################################
#
# The training of the network starts here
#
######################################################


#Load the embeddings and the dataset
embeddings, mappings, data = loadDatasetPickle(pickleFile)

# Some network hyperparameters
params = {'classifier': ['CRF'], 'LSTM-Size': [100], 'dropout': (0.25, 0.25),
         'customClassifier': {'ampersand2': ['Softmax']}}

save_path = "models/[ModelName]_[DevScore]_[TestScore]_[Epoch].h5" 

if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--reload_wts', nargs='+', type=str, default=None, help='Paths of files to reload weights from.')
    args = parser.parse_args()
    
    if args.reload_wts is None:
        model.setMappings(mappings, embeddings)
        model.setDataset(datasets, data)
        model.modelSavePath = save_path
    
    else:
        model = BiLSTM.loadModel(args.reload_wts, params, embeddings, datasets, data, save_path)

    model.fit(epochs=500)
