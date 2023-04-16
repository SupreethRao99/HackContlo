import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.cluster import AgglomerativeClustering, OPTICS, Birch

class SegmentClusterer:

    def __init__(self,
                 dataframe:pd.DataFrame,
                 dataframe_columns:list):
        
        self.dataframe = dataframe
        self.dataframe.dropna(inplace=True)
        
        self.dataframe.columns = [col.lower() for col in self.dataframe.columns]
        
        self.dataframe = self.dataframe[dataframe_columns]
        self.max_features = 3

        self.reduced_features = None

    def reduce_dimensions(self,
                          max_features:int=0):
        
        if max_features == 0:
            max_features = self.max_features

        #encode all categorical data, if any
        s = (self.dataframe.dtypes == 'object')
        categorical_object_cols = list(s[s].index)

        print("Categorical variables in the dataset:", categorical_object_cols)
        
        label_encoder = LabelEncoder()
        for categorical_label in categorical_object_cols:

            categorical_values = self.dataframe[categorical_label].tolist()
            label_encoded_values = label_encoder.fit_transform(categorical_values)
            
            self.dataframe[categorical_label] = label_encoded_values

        #scale data
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(self.dataframe)

        #reduce dimensions
        reducer = PCA(max_features)
        reduced_features = reducer.fit_transform(scaled_features)

        self.reduced_features = reduced_features

        return reduced_features
    
    def cluster_features(self, reduced_features):

        agg_clusterer = AgglomerativeClustering(n_clusters=4)
        agg_clusters = agg_clusterer.fit_predict(reduced_features)

        optics_clusterer = OPTICS(min_samples=10)
        optics_clusters = optics_clusterer.fit_predict(reduced_features)

        birch_clusterer = Birch()
        birch_clusters = birch_clusterer.fit_predict(reduced_features)

        return agg_clusters, optics_clusters, birch_clusters
    
    def get_output(self):

        reduced_features = self.reduce_dimensions()

        agg_output, optics_output, birch_output = self.cluster_features(reduced_features=reduced_features)

        return agg_output, optics_output, birch_output