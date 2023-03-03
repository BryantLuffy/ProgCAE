import random
random.seed(203)
import os
import warnings
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import argparse


if __name__ == '__main__':
    # Initialize the ArgumentParser object
    parser = argparse.ArgumentParser()

    # Add the required arguments to the ArgumentParser object
    parser.add_argument('--path1', '-p1', type=str, required=True, help='The first omics file name.')
    parser.add_argument('--path2', '-p2', type=str, required=True, help='The second omics file name.')
    parser.add_argument('--path3', '-p3', type=str, required=True, help='The third omics file name.')
    parser.add_argument('--path4', '-p4', type=str, required=True, help='The forth omics file name.')
    parser.add_argument('--path5', '-p5', type=str, required=True, help='The survival file name.')

    # Parse the arguments passed to the script
    args = parser.parse_args()


# Read in the data
survive = pd.read_table(args.path5, sep=',', index_col=0)
CNV = pd.read_table(args.path1, sep=',', index_col=0)
miRNA = pd.read_table(args.path2, sep=',', index_col=0)
RNA = pd.read_table(args.path3, sep=',', index_col=0)
Meth = pd.read_table(args.path4, sep=',', index_col=0)

# Import the necessary modules
from ProgCAE import Model
from ProgCAE import Process
from ProgCAE.utils import ClusterProcessor, do_km_plot
from ProgCAE.Survive_select import survive_select
# Process the data
CNV_processor = Process.DataProcessor(CNV)
CNV = CNV_processor.sort_corr(2000)
RNA_processor = Process.DataProcessor(RNA)
RNA = RNA_processor.sort_corr(5000)
miRNA_processor = Process.DataProcessor(miRNA)
miRNA = miRNA_processor.sort_corr(300)
Meth_processor = Process.DataProcessor(Meth)
Meth = Meth_processor.sort_corr(1000)

# Build the models
CNV_model = Model.ProgCAE(CNV.shape[1])
RNA_model = Model.ProgCAE(RNA.shape[1])
miRNA_model = Model.ProgCAE(miRNA.shape[1])
Meth_model = Model.ProgCAE(Meth.shape[1])

# Train the models
CNV_model.fit(CNV)
RNA_model.fit(RNA)
miRNA_model.fit(miRNA)
Meth_model.fit(Meth)

# Extract features from the models
CNV_feature = CNV_model.extract_feature(CNV)
RNA_feature = RNA_model.extract_feature(RNA)
miRNA_feature = miRNA_model.extract_feature(miRNA)
Meth_feature = Meth_model.extract_feature(Meth)

# Combine the features into a single dataframe
flatten = pd.concat([pd.DataFrame(CNV_feature),
                     pd.DataFrame(RNA_feature),
                     pd.DataFrame(miRNA_feature),
                     pd.DataFrame(Meth_feature)], axis=1)
SURVIVE_SELECT = survive_select(survive, flatten, 0.01)
# Save the dataframe to a file
flatten.to_csv('ProgCAE_features.csv')

# Perform clustering and survival analysis


# Initialize the cluster processor
cp = ClusterProcessor(SURVIVE_SELECT, survive)

# Compute indexes for clustering
cp.compute_indexes(5)

# Compute p-value and clusters for Log-Rank test
p_value, clusters = cp.LogRankp(nclusters=3)

# Plot survival curves using Kaplan-Meier method
do_km_plot(survive, pvalue=p_value.p_value, cindex=None, cancer_type='example', model_name='ProgCAE')
