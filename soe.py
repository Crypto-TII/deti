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
from .blackbox import getLocalMatrixAndBias, findCorner
from . import common
from . import whitebox


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


def findSign_soe(func, shape, weights, biases, eps=1e-6, tol=None):
    """
    Finds the signs of the neurons corresponding to a given layer, assuming the weights/bias of this layer
    are known up to a global sign and the weights/bias of previous layers are known exactly.

    Parameters
    ----------
    func
        The DNN. func must handle vectorised inputs, i.e., on input
        [x_1, ..., x_n], it must return [DNN(x_1), ..., DNN(x_n)].
    shape : tuple
        The shape of a single input to func.
    weights : 
        A list of weights for the known layers (each of them a 2D array), the
        last of which is given up to a sign in each column
    biases :
        A list of biases for the known layers (each of them a 1D array), the
        last of which is given up to a sign in each column
    eps :
        Small size for the amplitude of random wiggles as a factor of the central value.
    tol :
        Small threshold at which a value is considered zero.

    Returns
    -------
    signs:
        A 1D int array corresponding to the sign of each neuron in the last layer.
    """
    
    print("""
    # STEP 1: Find point x
    # ---------------------------------------------------""")
    t0 = time.time()
    # Number of neurons in current layer
    n = weights[-1].shape[-1]

    # Number of neurons we can fix in the previous layer
    N = min( [weight.shape[0] for weight in weights[:2]] + [weight.shape[0]//2 for weight in weights[2:]] )

    # Find a point where n previous-layer neurons are positive
    if len(weights) == 1 and N >= n:
        # For the input layer, we just choose positive inputs
        x = np.ones(shape=shape)
    elif len(weights) == 2 and N >= n:
        # For the first hidden layer we just invert the matrix
        x = np.matmul(1-biases[0][range(N)], np.linalg.pinv(weights[0][:,range(N)])).reshape(shape)
    elif N >= n:
        # For deeper layers, if the network is contractive enough we just guess a good x
        while True:
            x = np.random.rand(*shape)
            M,b = getLocalMatrixAndBias(weights, biases, x.flatten())
            if np.linalg.matrix_rank(M) >= n:
                break
    else:
        # Otherwise we need a way to get a larger dimension
        print("ERROR: network is not contractive enough at hidden layer",len(weights))
        print("Expected dimension:", N)
        print("Required dimension:", n)
        exit(-1)
        # print("Attempting corner-finding variant (this might fail to terminate..)")
        # return findSign_soe2(func, shape, weights, biases, eps, tol)
    tx = time.time()-t0
    print(f"\t Execution time: \t {tx:.6f} seconds.")
        
    print("""
    # STEP 2: Get local matrix M
    # ---------------------------------------------------""")
    t0 = time.time()
    # Local matrix and outpout around this point
    M,b = getLocalMatrixAndBias(weights, biases, x.flatten())
    x = x.reshape((1,)+x.shape)
    y = func(x).flatten()
    tM = time.time()-t0
    print(f"\t Execution time: \t {tM:.2f} seconds.")

    print("""
    # STEP 3: Collect system of equations (parallelizable)
    # ---------------------------------------------------""")
    t0 = time.time()
    # Get system of equations
    X = []
    Y = []
    for i in range(n):
        wiggle = eps*np.random.uniform(low=-1, high=1, size=(1,)+shape)
        X.append(np.matmul(wiggle.flatten(), M))
        Y.append((func(x+wiggle).flatten() - y).flatten())
    tE = time.time()-t0
    print(f"\t Execution time: \t {tE:.2f}seconds \t {(tE/n):.6f} seconds parallelized.")

    print("""
    # STEP 4: Solve system of equations
    # ---------------------------------------------------""")
    t0 = time.time()
    # Solve system of equations
    X = np.array(X)
    Y = np.array(Y)
    a = np.matmul(np.linalg.pinv(X), Y)
    tS = time.time()-t0
    print(f"\t Execution time: \t {tS:.2f} seconds.")
    print(f"Total Execution Time {(tx+tM+tE+tS):.2f} seconds, respectively {(tx+tM+tE/n+tS):.2f} parallelized runtime.")

    if not tol:
        tol = 10**(np.mean(np.log10(np.abs(a))))
    
    # Assign signs according to the size of the coefficients
    signs = []
    for coefficient in a:
        if np.abs(coefficient) < tol:
            signs.append(-1)
        else:
            signs.append(1)
    
    # Flip signs if the predicted value at x was negative
    signs *= np.sign(np.matmul(x.flatten(), M) + b)
    
    return signs

def findSign_soe2(func, shape, weights, biases, eps=1e-6, tol=None):
    """
    Alternate version for findSign_soe that works for layers deeper
    than the second hidden layer that are not contractive enough.
    This may work if the current layer size is
    less than about 3/4 of all previous layer sizes.
    """
    # Number of neurons in current layer
    n = weights[-1].shape[-1]

    # Number of neurons we can fix in the previous layer
    N = min( [weight.shape[0] for weight in weights[:2]] + [weight.shape[0]//2 for weight in weights[2:]] )

    # Find a critical point for N previous-layer neurons    
    x = findCorner(weights[:-1], biases[:-1], shape, range(N), targetValue = 0, tol=1e-7)

    # Local matrix and outpout around this point
    M,b = getLocalMatrixAndBias(weights, biases, x.flatten())
    X0 = np.matmul(x.flatten(), M) + b
    x = x.reshape((1,)+x.shape)
    y = func(x).flatten()

    # Get system of equations
    X = []
    Y = []
    for i in range(n):
        wiggle = eps*np.random.uniform(low=-1, high=1, size=(1,)+shape)
        M,b = getLocalMatrixAndBias(weights, biases, (x+wiggle).flatten())
        X.append(np.matmul((x+wiggle).flatten(), M) + b - X0)
        Y.append((func(x+wiggle).flatten() - y).flatten())

    # Solve system of equations
    X = np.array(X)
    Y = np.array(Y)
    a = np.matmul(np.linalg.pinv(X), Y)

    if not tol:
        tol = 10**(np.mean(np.log10(np.abs(a))))
    
    # Assign signs according to the size of the coefficients
    signs = []
    for coefficient in a:
        if np.abs(coefficient) < tol:
            signs.append(-1)
        else:
            signs.append(1)
    
    #Flip signs if the predicted value at x was negative
    signs *= np.sign(X0)
    
    return signs

if __name__=='__main__':
    logger.info("""
    # ----------------------------------------------------------
    # This is DETI sign recovery using SOE
    # ----------------------------------------------------------
    """)
    
    args = common.parseArguments()
    if args.tgtNeurons:
        print("Warning: ignoring --tgtNeurons parameter (SOE must solve all neurons in a layer)")
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
    

    # Blackbox function
    if len(model.layers[-1].output.shape) > 1:
        func = lambda x:model.predict(x, verbose = 0)[:,0]
    else:
        func = lambda x:model.predict(x, verbose = 0)

    shape = model.layers[0].input.shape[1:]

    starttime = time.time()
    blackbox_signs = findSign_soe(func, shape, weights, biases)
    stoptime = time.time()


    # WHITEBOX: Get the real signs to be able to control our results:
    whitebox_signs = whitebox.getRealSigns(model, args.layerID)

    print("Layer "+str(args.layerID)+" correct signs: "+str(np.sum(whitebox_signs == blackbox_signs))+"/"+str(len(whitebox_signs)))
    failed = [i for i in range(len(whitebox_signs)) if whitebox_signs[i] != blackbox_signs[i]]
    if len(failed) > 0:
        print("Failed neurons:")
        print(failed)
    
    df = pd.DataFrame()
    df['modelID'] = [modelname]*nNeurons
    df['layerID'] = [args.layerID]*nNeurons
    df['neuronID'] = np.arange(nNeurons)
    df['realSign'] = whitebox_signs
    df['recoveredSign'] = blackbox_signs
    df['isCorrect'] = df['recoveredSign']==df['realSign']
    df['recoveryTimeSeconds'] = (stoptime - starttime)/len(df) # time per neuron
    logger.info(df.to_markdown())
    
    logger.debug(f"Saving results to {filename_md} and {filename_pkl}...")
    df.to_pickle(filename_pkl)
    df.to_markdown(filename_md)
    
    logger.info(f"Total runtime: {(stoptime - starttime):.2f} seconds.")
    logger.info(f"Correctly recovered: {np.sum(df['isCorrect'])}/{len(df)}.")
