import numpy as np
import pandas as pd
import sys

# Decompose a circular feature like azimuth, LST, or orientation into its 1) sine and 2) cosine components.
# Drop original circular feature, unless optional drop argument is set explicitly to False.
def decompose_circ_feature(df: pd.DataFrame, *features, scale2rad=1, drop=True, debug=False):
    scale_to_convert_to_radians=scale2rad
    for feature in features:
        df[feature+"_sin"] = np.sin(df[feature] * scale_to_convert_to_radians)
        df[feature+"_cos"] = np.cos(df[feature] * scale_to_convert_to_radians)
        if drop:
            df = df.drop(columns=feature)
    return df


def scalar2vector(*arg):
    print("scalar2vector.scalar2vector() is depreciated. Use scalar2vector.uvmagnitude() instead.")
    sys.exit(1)

# Used to be named scalar2vector
# Given U and V components, return magnitude.
def uvmagnitude(df, drop=True, debug=False):
    possible_features = [f'SHR{z}{potential}{sfx}' for z in "136" for potential in ["","-potential"] for sfx in ["_min","_mean","_max"]]
    possible_features += [f'10{potential}{sfx}' for potential in ["","-potential"] for sfx in ["_min","_mean","_max"]]
    # Find these u/v features
    for possible_feature in possible_features:
        uc = "U"+possible_feature 
        vc = "V"+possible_feature 
        if uc in df.columns and vc in df.columns:
            df[possible_feature] = ( df[uc]**2 + df[vc]**2 )**(0.5)
            if drop:
                df = df.drop(columns=[uc,vc])
    return df

