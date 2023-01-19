    import tensorflow as tf
    from keras.models import Sequential
    from keras.layers.convolutional import Conv2DTranspose
    from keras.layers.convolutional import Conv2D
    from keras.layers import Reshape
    import pandas as pd
    from keras.layers import Dense, Flatten
    import os
    import keras.backend as K

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    ## Load the sample data
    survive = pd.read_table('example_sur.csv', sep=',', index_col=0)
    CNV = pd.read_table('example_cnv.csv', sep=',', index_col=0)
    miRNA = pd.read_table('example_miRNA.csv', sep=',', index_col=0)
    RNA = pd.read_table('example_RNA.csv', sep=',', index_col=0)
    Meth = pd.read_table('example_Meth.csv', sep=',', index_col=0)


    ## Variance filter
    def std_filter(data, number):
        index = list(data.std().sort_values(ascending=False)[0:number].index)
        return data[index]


    ## normalization
    def MinmaxVARIABLES(data, number):
        data = (std_filter(data, number))
        data = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))
        return data


    ## Sort based on cumulative correlation coefficient
    def sort_corr(data, number):
        data = MinmaxVARIABLES(data, number)
        abs_data = data.corr().abs()
        cumprod_data = abs_data.cumprod() ** (1 / len(abs_data))
        cumprod_sort_index = pd.Series(cumprod_data.iloc[:, -1].sort_values(ascending=False).index)
        return data[cumprod_sort_index]


    ## base model
    def Conv_AE(input_shape, kernel_height1, kernel_height2, kernel_height3,
                stride1, stride2, stride3, filter1, filter2, filter3,
                hidden_dim):
        model = Sequential()
        ##three convolutional layers
        model.add(Conv2D(filter1, (1, kernel_height1), strides=(1, stride1), padding='same', activation='relu',
                         input_shape=(1, input_shape, 1)))
        model.add(Conv2D(filter2, (1, kernel_height2), strides=(1, stride2), activation='relu', padding='same'))
        model.add(Conv2D(filter3, (1, kernel_height3), strides=(1, stride3), activation='relu', padding='same'))
        ##fully connected layers
        model.add(Flatten())
        model.add(Dense(units=hidden_dim, name='HiddenLayer'))
        model.add(Dense(units=filter3 * int(input_shape / (stride3 * stride2 * stride1)), activation='relu'))
        model.add(Reshape((1, int(input_shape / (stride3 * stride2 * stride1)), filter3)))
        ##three de_conv layers
        model.add(Conv2DTranspose(filter2, (1, kernel_height3), strides=(1, stride3), padding='same', activation='relu'))
        model.add(Conv2DTranspose(filter1, (1, kernel_height2), strides=(1, stride2), activation='relu', padding='same'))
        model.add(Conv2DTranspose(1, (1, kernel_height1), strides=(1, stride1), activation='relu', padding='same'))
        model.summary()
        return model


    ## fitting model
    def fit_model(data, model, epochs, batch_size, validation_split, learningrate):
        model.compile(loss='mean_squared_error',
                      optimizer=tf.keras.optimizers.Adam(lr=learningrate))
        data_train = tf.reshape(data, [-1, 1, data.shape[1], 1])
        model.fit(data_train, data_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
        return model


    def extract_feature(x, model):
        f = K.function(model.input, model.get_layer('HiddenLayer').output)
        x = tf.reshape(x, [-1, 1, x.shape[1], 1])
        return f(x)


    ## Survival selection
    def survive_select(survive_data, data, p_thresh):
        from lifelines import CoxPHFitter
        survive_data = survive_data.iloc[:, 0:2]
        columns_names = survive_data.T.columns.values
        dat_trans = data.T
        dat_trans.columns = columns_names
        dat_trans = dat_trans.T
        list = []
        for i in range(dat_trans.shape[1]):
            survive_data['feature'] = dat_trans.iloc[:, i].values
            cpf = CoxPHFitter()
            cpf.fit(survive_data, 'OS.time', 'OS')
            if cpf.summary['p'].values <= p_thresh:
                list.append(i)
        return dat_trans.iloc[:, list]


    ##Cluster
    def KmeansCluster(data, nclusters):
        from sklearn.cluster import KMeans
        K_mod = KMeans(n_clusters=nclusters)
        K_mod.fit(data)
        clusters = K_mod.predict(data)
        return clusters


    ##LogRankp
    def LogRankp(sur_data, data, nclusters):
        from lifelines.statistics import multivariate_logrank_test
        clusters = KmeansCluster(data, nclusters)
        sur_data['Type'] = clusters
        pvalue = multivariate_logrank_test(sur_data['OS.time'], sur_data['Type'], sur_data['OS'])
        return pvalue, clusters


    def compute_indexs(sur_data, data, maxclusters):
        from sklearn import metrics
        from sklearn.cluster import KMeans
        for i in range(2, maxclusters, 1):
            estimator = KMeans(n_clusters=i)
            estimator.fit(data)
            pvalue, clusters = LogRankp(sur_data, data, i)

            print("Number of clusters: ", i)
            print("silhouette score：", metrics.silhouette_score(data, estimator.labels_, metric='euclidean'))
            print("P-values: ", pvalue.p_value)


    def do_km_plot(sur_data, pvalue, cindex, name, model):
        from lifelines import KaplanMeierFitter
        import matplotlib.pyplot as plt
        import numpy as np
        import seaborn as sns
        values = np.asarray(sur_data['Type'])
        events = np.asarray(sur_data['OS'])
        times = np.asarray(sur_data['OS.time'])
        sns.set(style='ticks', context='notebook', font_scale=1.5)
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines['bottom'].set_linewidth(1.5)  # 图框下边
        ax.spines['left'].set_linewidth(1.5)  # 图框左边
        kaplan = KaplanMeierFitter()
        for label in set(values):
            kaplan.fit(
                times[values == label],
                event_observed=events[values == label],
                label='cluster {0}'.format(label)
            )
            kaplan.plot_survival_function(ax=ax,
                                          ci_alpha=0)
            ax.legend(loc=1, frameon=False)
        if cindex == None:
            ax.set_xlabel('days', fontsize=20)
            ax.set_ylabel('Survival Probability', fontsize=20)
            ax.set_title('{1} \n Cancer: {0}    p-value.:{2: .1e} '.format(
                name, model, pvalue),
                fontsize=18,
                fontweight='bold')
        else:
            ax.set_xlabel('days', fontsize=20)
            ax.set_title('{1} \n Cancer: {0}  p-value.:{2: .1e}   Cindex: {3: .2f}'.format(
                name, model, pvalue, cindex),
                fontsize=18,
                fontweight='bold')

        fig.savefig('./' + str(name) + model + '.tiff', dpi=300)


    CNV = sort_corr(CNV, 2000)
    RNA = sort_corr(RNA, 5000)
    miRNA = sort_corr(miRNA, 300)
    Meth = sort_corr(Meth, 1000)

    ## All CAEs
    CNV_CAE = Conv_AE(CNV.shape[1], 16, 10, 8, 10, 5, 2, 32, 64, 128, 30)
    RNA_CAE = Conv_AE(RNA.shape[1], 40, 16, 8, 10, 5, 5, 32, 64, 128, 30)
    miRNA_CAE = Conv_AE(miRNA.shape[1], 10, 6, 5, 5, 3, 2, 32, 64, 128, 30)
    Meth_CAE = Conv_AE(Meth.shape[1], 20, 10, 5, 10, 5, 2, 32, 64, 128, 30)
    ## fit CAEs
    CNV_CAE = fit_model(CNV, CNV_CAE, 500, 32, 0.2, 0.0005)
    Gene_CAE = fit_model(RNA, RNA_CAE, 500, 32, 0.2, 0.0005)
    miRNA_CAE = fit_model(miRNA, miRNA_CAE, 500, 32, 0.2, 0.0005)
    Meth_CAE = fit_model(Meth, Meth_CAE, 500, 32, 0.2, 0.0005)
    ## extract hidden layers
    CNV_hidden_features = extract_feature(CNV, CNV_CAE)
    Gene_hidden_features = extract_feature(RNA, Gene_CAE)
    miRNA_hidden_features = extract_feature(miRNA, miRNA_CAE)
    Meth_hidden_features = extract_feature(Meth, Meth_CAE)
    ## concatenate all features
    flatten = pd.concat([pd.DataFrame(CNV_hidden_features),
                         pd.DataFrame(Gene_hidden_features),
                         pd.DataFrame(miRNA_hidden_features),
                         pd.DataFrame(Meth_hidden_features)], axis=1)
    flatten.to_csv('progcae_features.csv')

SURVIVE_SELECT = survive_select(survive, flatten, 0.01)
##choose the best k
compute_indexs(survive, SURVIVE_SELECT, 6)

p_value, clusters = LogRankp(survive, SURVIVE_SELECT, 2)
## km-plot
do_km_plot(survive, pvalue=p_value.p_value, cindex=None, name='example', model='ProgCAE')
