# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 11:18:57 2021

@author: jmatt
"""

import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.realpath('..\jem_som'))
import SOM


cd = os.getcwd()
root_output_dir = os.path.join(cd,'Output')


fn = 'VTDEC_PFAS_Influent_8-5-2021_Sydney_Adams.csv'
data_df = pd.read_csv(fn)

label_header = 'Influent Type'
# label_header = 'Site'

  
labels = data_df[label_header]    
    
if label_header == 'Influent Type':
    headers_to_drop = [
        'Site',
        'Date']
    legend_text = {
        'Industrial Discharge':'ID',
        'Landfill Leachate':'LL',
        'Residential Only':'RO'}
else:
    headers_to_drop = [
        'Influent Type',
        'Date']
    
    label_set = set(data_df[label_header])
    
    legend_text = {}
    for label in label_set:
        legend_text[label] = label
    # legend_text = {
    #     'Industrial Discharge':'ID',
    #     'Landfill Leachate':'LL',
    #     'Residential Only':'RO'}

headers_to_keep = [header for header in data_df.keys() if header not in headers_to_drop]

data_df = data_df[headers_to_keep]


nonsingular_features = SOM.get_nonsingular_cols(data_df)
data_df = data_df[nonsingular_features]

data_labels = [key for key in data_df.keys() if key != label_header]

norm_data = True
if norm_data:
    X = SOM.min_max_norm(data_df,data_labels)
else:
    X = data_df






#Set the grid size
grid_size = [50,50]
# grid_size = [4,4]
#Set the starting learning rate and neighborhood size
alpha = 0.9
# neighborhood_size = 4
neighborhood_size = int(grid_size[0]/4)
# neighborhood_size = 1
#Set the number of training epochs
num_epochs = 500

load_trained = False
toroidal = True
distance = 'euclidean'

for i in range(3):
    #Init a SOM object
    SOM_model = SOM.SOM(grid_size,X,label_header,alpha,neighborhood_size,toroidal,distance)
    
    #Train the SOM
    
    output_dir = os.path.join(root_output_dir,f'run{i}')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if load_trained:
        weights_dir = r'SOM_output\run10_keep'
        weights_dir = weights_dir.split('\\')
        directory = cd
        for folder in weights_dir:
            directory = os.path.join(directory,folder)
            
        fn = 'weights.pkl'
        
        SOM_model.load_weights(directory,fn)
        
        
        
            
    else:
        SOM_model.train(num_epochs)
        SOM_model.save_weights(output_dir,'weights.pkl')
    
    SOM_model.plot_weight_hist(output_dir)
    
    #Plot the samples to the grid
    # SOM_model.plot_samples()
    
    #Calculate the U-matrix differences
    SOM_model.calc_u_matrix()
    #boolean flag to include D in the u-matrix calculations
    include_D = False
    #Plot the u-matrix
    sample_vis='labels'
    sample_vis='symbols'
    
    SOM_model.plot_u_matrix(include_D,output_dir=output_dir,labels=labels,sample_vis=sample_vis,legend_text=legend_text)
    plane_vis='u_matrix'
    plane_vis='weights'
    #Plot the feature planes
    SOM_model.plot_feature_planes(output_dir,labels=labels,sample_vis=sample_vis,plane_vis=plane_vis,legend_text=legend_text)
    
