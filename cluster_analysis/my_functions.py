import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.metrics import accuracy_score

def pca_visual(X_data, Y_data, dict_CLnames, comp=False, clusters=None,):
    """
    Visualize the data easily with PCA
    
    Input: X_data: data from all the features that define Y_data
           Y_data: Target data
           dict_CLnames: dict. with encoding value with specific target class names. eg. {0:'ALL',1:'AML'} 
           comp: if want to compare with another dataset like clustered, put 'True'
           clusters: add the information from cluster classes
           
    """
    pca = PCA(2)  # project from 72 to 2 dimensions
    X_pca = pca.fit_transform(X_data)

    #encode class labels into numeric values
    le = preprocessing.LabelEncoder()
    label_encoder = le.fit(Y_data)
    y = label_encoder.transform(Y_data)

    Xax=X_pca[:,0] #First Principal Component
    Yax=X_pca[:,1] #Second Principal Component
    labels= y
    cdict={0:'red',1:'green'} #dict with colors
    labl=dict_CLnames
    labl_cl = {0:'cluster 1',1:'cluster 2'}
    if comp == False:
        fig,ax=plt.subplots(figsize=(7,5))
        fig.patch.set_facecolor('white')
        for l in np.unique(labels):
            ix=np.where(labels==l)
            ax.scatter(Xax[ix],Yax[ix],c=cdict[l],s=40, label=labl[l])
        # for loop ends
        plt.xlabel("First Principal Component",fontsize=14)
        plt.ylabel("Second Principal Component",fontsize=14)
        plt.legend()
        plt.show()
        
    if comp == True:
        fig,axs =plt.subplots(nrows=1, ncols=2, figsize=(15,5))
        fig.patch.set_facecolor('white')
        ax = axs[0]
        for l in np.unique(labels):
            ix=np.where(labels==l)
            ax.scatter(Xax[ix],Yax[ix],c=cdict[l],s=40, label=labl[l])
        # for loop ends
        ax.set_xlabel("First Principal Component",fontsize=14)
        ax.set_ylabel("Second Principal Component",fontsize=14)
        ax.set_title('Original data')
        ax.legend()

        
        ax = axs[1]
        for l in np.unique(clusters):
            ix=np.where(clusters==l)
            ax.scatter(Xax[ix],Yax[ix],c=cdict[l],s=40, label=labl_cl[l])
        # for loop ends
        ax.set_xlabel("First Principal Component",fontsize=14)
        ax.set_ylabel("Second Principal Component",fontsize=14)
        ax.set_title('Clustered data')
        ax.legend()
        plt.show()


def cluster(X, Y, cluster_type = 'K-means', data='orig'):
    """
    Calculate the Silhouette score and how many patients originally with ALL and AML are in each cluster.
    
    Input: X: Input data (ex. X_data)
           Y: Ouput data (ex. Y_data)
           cluster_type: 'K-means' or 'Hierarchical'
           data: which data (orig, var, pca)
           
    Output: the Silhouette score and how many patients ALL and AML are in each cluster.
    
    Adapted from: scikit-learn developers(2007-2019): "Selecting the number of clusters with silhouette analysis on KMeans clustering", https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html#sphx-glr-auto-examples-cluster-plot-kmeans-silhouette-analysis-py
    
    Modified by: João Vidigal 
    """
    range_n_clusters = [ 2, 3, 4, 5, 6]
    dic_sil = {}
    parameter_linkage = ['single','average','complete', 'ward']
    count=0
    sil_base = 0
    acc_val_base = 0
    if cluster_type == 'K-means':
        for n_clusters in range_n_clusters:

            # Initialize the clusterer with n_clusters value and a random generator
            # seed of 10 for reproducibility.
                clusterer = KMeans(n_clusters=n_clusters, random_state=10)
                cluster_labels = clusterer.fit_predict(X)
            # The silhouette_score gives the average value for all the samples.
            # This gives a perspective into the density and separation of the formed
            # clusters

                silhouette_avg = silhouette_score(X, cluster_labels)

                dic_sil[n_clusters] = silhouette_avg

                print("For n_clusters =", n_clusters,
                      "The average silhouette_score is :", np.round(silhouette_avg,3))

                #Combine the true labels and cluster labels into a dataframe
                df_cluster_list = pd.DataFrame({'clusters' : clusterer.labels_, 'labels' : Y})

                #Group by clusters and for each cluster get the fraction of each label
                df_cluster_list_1 = pd.get_dummies(df_cluster_list['labels'])
                df_cluster_list_2 = pd.concat([df_cluster_list, df_cluster_list_1], axis=1)
                ddf_cluster_list_3 = df_cluster_list_2[['clusters','ALL','AML']].groupby('clusters').sum()
                
                le = preprocessing.LabelEncoder()
                label_encoder = le.fit(df_cluster_list['labels'])
                y = label_encoder.transform(df_cluster_list['labels'])
                y_pre = df_cluster_list['clusters']
                acc_val = accuracy_score(y_pre, y)
                
                
                print("\n class predict : ", np.round(acc_val, 2)*100, '%\n')
                print("\n class predict : \n", ddf_cluster_list_3, '\n')

                
                info = f'{cluster_type}_{data}'
                df_sil = pd.DataFrame.from_dict(dic_sil, orient='index') 
                df_sil[info] = df_sil[0]
                df_sil = df_sil.drop(0, axis=1)
                
                    
                if silhouette_avg > sil_base:
                    df_bst_sil = df_cluster_list 
                    info_sil = f'{cluster_type} with k={n_clusters}'
                    sil_base = silhouette_avg

                if acc_val > acc_val_base:
                    df_bst_acc = df_cluster_list 
                    info_acc = f'{cluster_type} with k={n_clusters}'
                    acc_val_base = acc_val
                    
                    

                    
                
    if cluster_type == 'Hierarchical':
        for linkage in parameter_linkage:
        
            print("Linkage parameter =", linkage,'\n')
            for n_clusters in range_n_clusters:

                clusterer = agglomerative = AgglomerativeClustering(linkage = linkage, n_clusters=n_clusters)
                cluster_labels = clusterer.fit_predict(X)
                # The silhouette_score gives the average value for all the samples.
                # This gives a perspective into the density and separation of the formed
                # clusters


                silhouette_avg = silhouette_score(X, cluster_labels)
                dic_sil[n_clusters] = silhouette_avg

                print("For n_clusters =", n_clusters,
                        "The average silhouette_score is :", np.round(silhouette_avg,3))

                #Combine the true labels and cluster labels into a dataframe
                df_cluster_list = pd.DataFrame({'clusters' : clusterer.labels_, 'labels' : Y})

                #Group by clusters and for each cluster get the fraction of each label
                df_cluster_list_1 = pd.get_dummies(df_cluster_list['labels'])
                df_cluster_list_2 = pd.concat([df_cluster_list, df_cluster_list_1], axis=1)
                ddf_cluster_list_3 = df_cluster_list_2[['clusters','ALL','AML']].groupby('clusters').sum()

                le = preprocessing.LabelEncoder()
                label_encoder = le.fit(df_cluster_list['labels'])
                y = label_encoder.transform(df_cluster_list['labels'])
                y_pre = df_cluster_list['clusters']
                acc_val = accuracy_score(y_pre, y)
                
                
                le = preprocessing.LabelEncoder()
                label_encoder = le.fit(df_cluster_list['labels'])
                y = label_encoder.transform(df_cluster_list['labels'])
                y_pre = df_cluster_list['clusters']
                acc_val = accuracy_score(y_pre, y)
                
                
                print("\n class predict : ", np.round(acc_val, 2)*100, '%\n')
                print("\n class predict : \n", ddf_cluster_list_3, '\n')

            
                
                if silhouette_avg > sil_base:
                    df_bst_sil = df_cluster_list 
                    info_sil = f'{cluster_type} {linkage} with k={n_clusters}'
                    sil_base = silhouette_avg
                    
                if acc_val > acc_val_base:
                    df_bst_acc = df_cluster_list 
                    info_acc = f'{cluster_type} {linkage} with k={n_clusters}'
                    acc_val_base = acc_val
                    
            if silhouette_avg > sil_base:
                df_bst_sil = df_cluster_list 
                info_sil = f'{cluster_type} {linkage} with k={n_clusters}'
                sil_base = silhouette_avg
                    
            if acc_val > acc_val_base:
                df_bst_acc = df_cluster_list 
                info_acc = f'{cluster_type} {linkage} with k={n_clusters}'
                acc_val_base = acc_val



            if count == 0:
                info = f'{cluster_type}_{data}_{linkage}'
                df_sil_1 = pd.DataFrame.from_dict(dic_sil, orient='index') 
                df_sil_1[info] = df_sil_1[0]
                df_sil_1= df_sil_1.drop(0, axis=1)
                df_sil = df_sil_1
                count+=1
            else:
                info = f'{cluster_type}_{data}_{linkage}'
                df_sil_1 = pd.DataFrame.from_dict(dic_sil, orient='index') 
                df_sil_1[info] = df_sil_1[0]
                df_sil_1= df_sil_1.drop(0, axis=1)
                df_sil = pd.concat([df_sil,df_sil_1], axis=1)
            
    print("The best silloutte was achieved in ", info_sil,
                        " and the best accuracy was achieved in ", info_acc )


    return df_sil, df_bst_sil, df_bst_acc


