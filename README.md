ProgCAE: A novel method to integrate multi-omics data for predicting cancer subtypes 
====
![image](https://github.com/BryantLuffy/ProgCAE/blob/master/shell.png)

Description
-----
Determining cancer subtypes and patient prognosis analysis are important tasks of cancer research. The massive amount of multi-omics data spawned by high-throughput sequencing technology is an important resource for cancer prognosis, and deep learning methods can be used to effectively integrate multi-omics data to identify more accurate cancer subtypes. <br>

ProgCAE, a prognostic model based on convolutional autoencoder (CAE) to predict patient survival subtypes using multi-omics data. 

ProgCAE can predict cancer subtypes with significant survival differences on 12 cancers and outperforms traditional statistical methods on most cancers. ProgCAE is highly robust, its predicted subtypes can be used to construct supervised classifiers.

Dependencies
-----
ProgCAE is implemented in Python 3.9, which also requires the installation of keras, lifelines, numpy, pandas, scikit-learn, scipy, tensorflow and other packages. Their specific versions are as follows.<br>
### packages
`keras`         `2.6.0`<br>
`lifelines`     `0.27.2`<br>
`numpy`         `1.23.3`<br>
`pandas`        `1.4.4`<br>
`scikit-learn`  `1.1.2`<br>
`scipy`         `1.9.1`<br>
`tensorflow`    `2.6.0`<br> 

Usage
-----
The input for ProgCAE consists of multiple sets of omics matrices and survival information, which must be in csv format. For a specific omics matrix, its rows should represent samples (patients), its columns should represent features (genes), and the first column should be the id of each patient. The first three columns of the survival information table should be the patient id, status, and time, respectively. The output includes a survival feature matrix, P values for different clustering numbers, and the Kalpan-Meier curve. To reproduce this method, you can use the following command:<br>
```
python Prog_CAE.py [-p1 OMIC1] [-p2 OMIC2] [-p3 OMIC3] [-p4 OMIC4] [-p5 Survive]
```
The option -p is used to specify the input file path. In this study, OMIC1-4 represent CNV, miRNA, RNA, and Methylation data, respectively.

Example
-----
Taking the example data as an example, after you have downloaded the project files, the usage is as follows:
```
python Prog_CAE.py -p1 example/example_cnv.csv -p2 example/example_miRNA.csv -p3 example/example_RNA.csv -p4 example/example_Meth.csv -p5 example/example_sur.csv
```
The program will take 2-3 minutes to run, and afterwards, you can obtain a survival feature matrix in CSV format and a KM curve graph (in TIFF format) when the number of clusters is 3 in the root directory.

