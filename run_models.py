import random
random.seed(203)
import os
import warnings
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf
import pandas as pd
from ProgCAE import Process
from ProgCAE.utils import ClusterProcessor, do_km_plot
from ProgCAE.Survive_select import survive_select


def extract_feature(model, x):
    # Get a Keras backend function that extracts the output of the 'HiddenLayer' layer given an input
    f = tf.keras.backend.function(model.input, model.get_layer('HiddenLayer').output)
    # Reshape the input data to the expected shape
    x = tf.reshape(x, [-1, 1, x.shape[1], 1])
    # Use the backend function to extract the hidden layer output for the input data
    return f(x)


# Read in the data
survive = pd.read_table('example/example_sur.csv', sep=',', index_col=0)
CNV = pd.read_table('example/example_cnv.csv', sep=',', index_col=0)
miRNA = pd.read_table('example/example_miRNA.csv', sep=',', index_col=0)
RNA = pd.read_table('example/example_RNA.csv', sep=',', index_col=0)
Meth = pd.read_table('example/example_Meth.csv', sep=',', index_col=0)

# load models
CNV_model = tf.keras.models.load_model('models/cnv_model.h5')
RNA_model = tf.keras.models.load_model('models/rna_model.h5')
miRNA_model = tf.keras.models.load_model('models/mirna_model.h5')
Meth_model = tf.keras.models.load_model('models/meth_model.h5')

# Process the data
CNV_processor = Process.DataProcessor(CNV)
CNV = CNV_processor.sort_corr(2000)
RNA_processor = Process.DataProcessor(RNA)
RNA = RNA_processor.sort_corr(5000)
miRNA_processor = Process.DataProcessor(miRNA)
miRNA = miRNA_processor.sort_corr(300)
Meth_processor = Process.DataProcessor(Meth)
Meth = Meth_processor.sort_corr(1000)

CNV_feature = extract_feature(CNV_model, CNV)
RNA_feature = extract_feature(RNA_model, RNA)
miRNA_feature = extract_feature(miRNA_model, miRNA)
Meth_feature = extract_feature(Meth_model, Meth)

# Combine the features into a single dataframe
flatten = pd.concat([pd.DataFrame(CNV_feature),
                     pd.DataFrame(RNA_feature),
                     pd.DataFrame(miRNA_feature),
                     pd.DataFrame(Meth_feature)], axis=1)
SURVIVE_SELECT = survive_select(survive, flatten, 0.01)
# Save the dataframe to a file
SURVIVE_SELECT.to_csv('ProgCAE_features.csv')

# Perform clustering and survival analysis
# Initialize the cluster processor
cp = ClusterProcessor(SURVIVE_SELECT, survive)

# Compute indexes for clustering
cp.compute_indexes(5)

# Compute p-value and clusters for Log-Rank test
p_value, clusters = cp.LogRankp(nclusters=3)

# Plot survival curves using Kaplan-Meier method
do_km_plot(survive, pvalue=p_value.p_value, cindex=None, cancer_type='example', model_name='ProgCAE')
