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
from . import blackbox
from . import whitebox
from . import common

# ---------------------------------------------------
# Tensorflow settings
# ---------------------------------------------------
# Don't show TensorFlow warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Prevent Tensorflow from gobbling the whole GPU memory
devices = tf.config.list_physical_devices('GPU')
for device in devices:
    tf.config.experimental.set_memory_growth(device, True)

# Set to high precision
tf.keras.backend.set_floatx('float64')

# ---------------------------------------------------
# Set up logging
# ---------------------------------------------------
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig()

def recoverSignLastLayer(func, weights, biases, numberNeurons, inputShape, layerId, low=-1.0, high=1.0, dataset=None, eps=1e-8):
    
    numberLayers = len(weights)
    
    # Get output coefficients
    c = blackbox.getLastLayerOutputMatrixBlackbox(func, weights, biases, inputShape, layerId, dataset, eps=eps, tol=1e-3*eps)

    #nSamples = (numberNeurons+Y.shape[-1])//Y.shape[-1]
    
    # Sample random inputs
    X = np.random.uniform(low=low, high=high, size=(numberNeurons+model.output_shape[-1], inputShape[0]))
    
    # Outputs of the DNN
    Y = func(X)
    
    matrix = []
    linTerms = []
    B = []
    
    rankOK = False
    checkRank = True
    i = 0
    outID = 0
    
    while not rankOK:
        
        if outID >= model.layers[-1].weights[0].numpy().shape[-1]:
            print("PRECISION ERROR(2): Try rerunning or decreasing --eps")
            exit(-1)

        # ---------------------------------------------------
        # Collect Equations 
        # ---------------------------------------------------
        # COMPUTE EQUATION TERMS 
        # Value of the neurons before the ReLU
        yi = blackbox.getHiddenVector(weights, biases, numberLayers, X[i])
        mask = yi>0 # all positive neurons
        # compute outputs in the case of correct sign guess
        o1 = mask*np.abs(yi)
        # compute alternative outputs in the case of an incorrect sign guess
        o2 = (~mask)*np.abs(yi)
        # Add output coefficients 
        # has a 1.0 at the current output ID, otherwise zeros
        # bs = [0.0, 0.0, ..., 1.0, 0.0]
        bs = np.zeros(Y.shape[-1])
        bs[outID] = 1.0
        # collect all matrix coefficients
        coeff = np.hstack((c[:,outID]*(o1-o2), bs))
        # compute the linear terms 
        linTerm = np.array([np.sum(o2 * c[:,outID])])
        
        # COLLECT EQUATION TERMS
        matrix.append(coeff)
        B.append(Y[i,outID]-linTerm)
        
        # ---------------------------------------------------
        # Checks before next input
        # ---------------------------------------------------    
        i+=1
        
        if (i%len(X)==0): 
            outID += 1
            i = 0
            
        # # Once we have seen all X at one output ID, 
        # # start to check the rank
        # if i==numberNeurons: 
        #     checkRank=True
            
        # We check the rank every 10 datapoints
        if checkRank and (i%10==0): 
            currentRank = np.linalg.matrix_rank(matrix)
            # (for debugging) print('currentRank=', currentRank)
            rankOK = (currentRank>=(numberNeurons+Y.shape[-1]))
    
    # convert to arrays
    matrix = np.array(matrix)
    B = np.array(B)
    
    # ---------------------------------------------------
    # Solve system of equations
    # --------------------------------------------------- 
    coeff, _, _, _ = np.linalg.lstsq(matrix, B, rcond=None)
    
    return coeff

if __name__=='__main__':
    logger.info("""
    # ----------------------------------------------------------
    # This is DETI sign recovery using the last layer technique.
    # ----------------------------------------------------------
    """)
    
    args = common.parseArguments()
    logger.info(f"Parsed arguments for sign recovery: \n\t {args}.")

    model = tf.keras.models.load_model(args.model)
    logger.info(f"Model summary:")
    logger.info(model.summary())

    # ---------------------------------------------------
    # Recover signatures
    # ---------------------------------------------------
    # Update signs as they would be recovered as signatures
    logger.info("Recovering signatures...")
    weights, biases = whitebox.getSignatures(model, args.layerID)

    # ---------------------------------------------------
    # Inferred settings
    # ---------------------------------------------------
    inputShape = model.input_shape[1:]
    hiddenLayerIDs = [i for i in np.arange(1, len(model.layers)-1)]
    neuronsHiddenLayers = [model.layers[i].output_shape[-1] for i in hiddenLayerIDs]
    outputs = model.output_shape[-1]
    # check output activation function is linear
    if model.layers[-1].activation != tf.keras.activations.linear:
        model.layers[-1].activation = tf.keras.activations.linear
        logger.warning(f"The last layer has to have a linear activation function, instead found {model.layers[-1].activation}. We will replace this output function with a linear one automatically in your model.")
        model.layers[-1].activation = tf.keras.activations.linear
    logger.info(f"""
        Determined the following model parameters: 
            input shape: \t {inputShape}
            hiddenLayerIDs: \t {hiddenLayerIDs}
            neuronsHiddenLayers: \t {neuronsHiddenLayers}
            outputs: \t {outputs}
        """)
    
    # assert that the last hidden layer is to be recovered
    assert args.layerID==hiddenLayerIDs[-1], logger.error(f"This sign recovery should be run on the last hidden layer, i.e., layerID={hiddenLayerIDs[-1]}. ")
    
    # Number of neurons in target layer
    nNeurons = len(biases[-1])
    
    # ---------------------------------------------------
    # Filenames
    # ---------------------------------------------------
    modelname = args.model.split('/')[-1].replace('.keras', '')
    savePath = common.getSavePath(modelname, args.layerID, args.nExp, runID=args.runID, mkdir=True)
    filename_pkl = savePath + 'df.pkl'
    filename_md  = savePath + 'df.md'
    logger.info(f"Sign recovery results will be saved to \n\t {filename_md}.")
    
    
    # ---------------------------------------------------
    # Run sign recovery
    # ---------------------------------------------------

    logger.info("""
    
    # STEP 1 Determine the output coefficients 
    # (parallelizable on the neuron level)
    # ---------------------------------------------------""")
    starttimeCoeff = time.time()
    TOL_ZERO = 0.3
    TOL_ONE = 0.3

    func = lambda x : model.predict(x)

    coeff = recoverSignLastLayer(func, weights, biases, nNeurons, inputShape, args.layerID, dataset=args.dataset, eps=10**(-args.eps))
    # the last coefficients are the bias term of the output neurons
    coeff = coeff[:-outputs] 
    
    stoptimeCoeff = time.time()

    logger.info("""
    
    # STEP 2 Determine the signs
    # ---------------------------------------------------""")
    starttimeSigns = time.time()
    signsRecovered = []
    for x in coeff:
        if np.abs(x) < TOL_ZERO:
            signsRecovered.append(-1.)
        elif np.abs(1-x) < TOL_ONE:
            signsRecovered.append(1.)
        else:
            print("PRECISION ERROR(3): Try rerunning or decreasing --eps")
            exit(-1)
    signsRecovered = np.array(signsRecovered)

    stoptimeSigns = time.time()

    # ---------------------------------------------------
    # Save and analyze the results
    # ---------------------------------------------------
    # WHITEBOX: Get the real signs to be able to control our results:
    whiteSignsLayer = whitebox.getRealSigns(model, args.layerID)
    
    timeCoeffs = (stoptimeCoeff - starttimeCoeff)
    timeSigns = (stoptimeSigns - starttimeSigns)
    timeTotal = timeCoeffs + timeSigns
    timeParallelized = timeCoeffs/nNeurons + timeSigns
    
    df = pd.DataFrame()
    df['modelID'] = [modelname]*nNeurons
    df['layerID'] = [args.layerID]*nNeurons
    df['neuronID'] = np.arange(nNeurons)
    df['realSign'] = whiteSignsLayer
    df['recoveredSign'] = signsRecovered
    df['isCorrect'] = df['recoveredSign']==df['realSign']
    df['coeff'] = coeff
    df['coeffTimeSeconds'] = timeCoeffs /len(df) # time per neuron
    df['signsTimeSeconds'] = timeSigns /len(df) # time per neuron
    df['recoveryTimeSeconds'] = timeTotal /len(df) # time per neuron
    logger.info(df.to_markdown())
    
    logger.debug(f"Saving results to {filename_md} and {filename_pkl}...")
    df.to_pickle(filename_pkl)
    df.to_markdown(filename_md)
    
    logger.info("""
    # Sign Recovery Results
    # ---------------------------------------------------""")
    
    logger.info(f"Time spend in total:                                               \t {timeTotal:.2f} seconds.")
    logger.info(f"  Time spend for recovery of output coefficients (parallelizable): \t {timeCoeffs:.2f} seconds.")
    logger.info(f"  Time spend for recovery of signs (not parallelizable):           \t {timeSigns:.6f} seconds.")
    logger.info(f"==> Estimated parallelized execution time:                         \t {timeParallelized:.2f} seconds.")
    logger.info(f"Correctly recovered: {np.sum(df['isCorrect'])}/{len(df)}.")
    
    
