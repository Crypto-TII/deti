# ---------------------------------------------------
# Prevent file locking errors
# ---------------------------------------------------
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

# ---------------------------------------------------
# Imports
# ---------------------------------------------------
import time
import numpy as np
import pandas as pd
import tensorflow as tf

def getFormattedTimestamp():
    from datetime import datetime
    # Format the timestamp
    formatted_timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
    return formatted_timestamp

def getSavePath(modelname, layerID, nExp, runID=None, mkdir=True):
    from pathlib import Path
    
    if not runID:
        runID = getFormattedTimestamp()
        
    pathName = f"results/model_{modelname}/layerID_{layerID}/nExp_{nExp}/runID_{runID}/"

    if mkdir:
        Path(pathName).mkdir(parents=True, exist_ok=True)

    return pathName

def parseArguments():

    # ---------------------------------------------------
    # Parse arguments from command line
    # ---------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description='Run the energy sign recovery.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # ---- add arguments to parser
    parser.add_argument('--model', type=str,
                        help='The path to a keras.model (https://www.tensorflow.org/tutorials/keras/save_and_load).')
    parser.add_argument('--layerID', type=int,
                        help='The ID of your target layer (as enumerated in model.layers).')
    parser.add_argument('--tgtNeurons', nargs='+', 
                        help="Specific target neuron IDs, e.g. '0 10 240'")
    parser.add_argument('--runID', type=str, 
                        help="A manual run ID (otherwise a time-tag run ID will be auto generated).")
    parser.add_argument('--dataset', type=str, 
                        help="If 'None' a random point will be chosen. If 'CIFAR10' a random CIFAR10 test image will be chosen as input.")
    parser.add_argument('--eps', type=int, 
                        help="(Optional) Precision parameter for lastLayer method (wiggle size is 10^-eps). Recommended 3 < eps < 8.")

    # ---- default values
    defaults = {'model': "./deti/modelweights/model_cifar10_256_256_256_256.keras",
                'layerID': 3,
                'tgtNeurons': None,
                'nExp': 200, 
                'runID': None, 
                'dataset': None, 
                'eps': 8, 
                }

    # ---- parse args
    parser.set_defaults(**defaults)
    args = parser.parse_args()

    return args

