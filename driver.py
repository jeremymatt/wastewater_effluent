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
output_dir = os.path.join(cd,'Output')
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)


fn = 'VTDEC_PFAS_Influent_8-5-2021_Sydney_Adams.csv'
data_df = pd.read_csv(fn)

label_header = 'Influent Type'

if label_header == 'Influent Type':
    headers_to_drop = [
        'Site',
        'Date']
else:
    headers_to_drop = [
        'Influent Type',
        'Date']
    
    

headers_to_keep = [header for header in data_df.keys() if header not in headers_to_drop]

data_df = data_df[headers_to_keep]


selected_features = SOM.get_nonsingular_cols(data_df)


# singular_features = [feat for feat in headers_to_keep if not feat in selected_features]
# print('\nWARNING: The following features have only one value:')
# if len(singular_features)>0:
#     for feat in singular_features:
#         print('    {}'.format(feat))
#     print('\n')
# else:
#     print('No features have')

label_set = set(data_df[label_header])

legend_text = {
    'Industrial Discharge':'ID',
    'Landfill Leachate':'LL',
    'Residential Only':'RO'}


