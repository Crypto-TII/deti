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


def isLinear(func, x_low, x_upp, eps=1e-4, tol=1e-10, debug=False):
    """
    Determines if a function is linear in a given closed interval.

    Parameters
    ----------
    func
        The function. func takes as input a real number (float) and outputs a
        real number (float).
    x_low : float
        Lower bound of the interval.
    x_upp : float
        Upper bound of the interval.
    eps : float, optional
        "Infinitesimal" variation used when computing slopes and derivatives.
    tol : float, optional
        Tolerance to decide if two real numbers (floats) are "equal".
    debug : bool, optional
        Specifies whether debug information is printed to standard output.

    Returns
    -------
    bool
        True if the function is linear in the given interval, False otherwise.
    """
    
    # Let
    #   x_mid = (x_low + x_upp) / 2,
    #   L_low be the line passing through x_low,
    #   L_upp be the line passing through x_upp,
    #   L_mid_low be the line passing through x_mid on the left, and
    #   L_mid_upp be the line passing through x_mid on the right.
    #
    # To check for linearity, we check the following:
    #    (i) f(x_mid) = (f(x_low) + f(x_upp)) / 2
    #   (ii) L_low = L_upp and L_mid_low = L_low and L_mid_upp = L_upp
    
    if debug:
        print(f"Checking [{x_low}, {x_upp}]")
    
    # Check whether x_low and x_upp are too close
    if (np.abs(x_upp - x_low) < tol):
        if debug:
            print("  Points are too close to each other")
        return None
    
    x_mid = (x_low + x_upp) / 2
    
    # Value of func at x_low, x_upp and x_mid
    y_low = func(x_low)
    y_upp = func(x_upp)
    y_mid = func(x_mid)
    
    # Expected value of func at x_mid
    y_mid_exp = (y_low + y_upp) / 2
    
    if debug:
        print(f"  f({x_low})  = {y_low}")
        print(f"  f({x_upp})  = {y_upp}")
        print(f"  f({x_mid})  = {y_mid}")
        print(f"  fe({x_mid}) = {y_mid_exp}")
        print(f"  |f_mid - fe_mid| = {np.abs(y_mid - y_mid_exp)}")
    
    # Check (i) f(x_mid) = (f(x_low) + f(x_upp)) / 2
    if (np.abs(y_mid - y_mid_exp) > tol):
        return False
    
    # Slopes of L_low and L_upp
    m_low = (func(x_low + eps) - y_low) / eps
    m_upp = (y_upp - func(x_upp - eps)) / eps
    
    if debug:
        print(f"  m_low = {m_low}")
        print(f"  m_upp = {m_upp}")
        print(f"  |m_low - m_upp| = {np.abs(m_low - m_upp)}")
    
    # Check (ii) L_low = L_upp
    if (np.abs(m_low - m_upp) > tol):
        return False
    
    # Slopes of L_mid_low and L_mid_upp
    m_mid_low = (y_mid - func(x_mid - eps)) / eps
    m_mid_upp = (func(x_mid + eps) - y_mid) / eps
    
    if debug:
        print(f"  m_mid_low = {m_mid_low}")
        print(f"  m_mid_upp = {m_mid_upp}")
        print(f"  |m_mid_low - m_low| = {np.abs(m_mid_low - m_low)}")
        print(f"  |m_mid_upp - m_upp| = {np.abs(m_mid_upp - m_upp)}")
    
    # Check (ii) L_mid_low = L_low and L_mid_upp = L_upp
    if (np.abs(m_mid_low - m_low) > tol) or (np.abs(m_mid_upp - m_upp) > tol):
        return False
    
    return True

def getProjection(v, basis):
    '''
    Compute the projection of vector v onto the vector space generated by the
    given orthogonal basis. The vectors of the basis are given as row vectors.

    Parameters
    ----------
    v : array
        1-dimensional array representing the vector v.
    basis : array
        2-dimensional array with the row vectors of the orthogonal basis.

    Returns
    -------
    array
        1-dimensional array representing the projection of v.
    '''
    res = np.zeros_like(v)
    for bi in basis:
        res += (np.dot(v, bi) / np.dot(bi, bi)) * bi
    return res

def getWigglesProjection(weights, signaturesProj, diffs, diffsEps, lyrEps):
    '''
    signaturesProj contains the projections as row vectors
    '''
    # Get wiggles in the layer
    wigglesLyr = (lyrEps / np.linalg.norm(signaturesProj, axis=1))[:, np.newaxis] * signaturesProj
    # Get wiggles in the input
    coeff, _, _, _ = np.linalg.lstsq(diffs.T, wigglesLyr.T, rcond=None)
    return diffsEps * coeff.T

def recoverSign(model, weights, biases, layerId, neuronId,
                inputShape,
                nExp=200,
                dataset=None, 
                EPS_IN=1e-6,
                EPS_LYR=1e-8,
                EPS_ZERO=1e-12,
                LINEARITY_EPS=1e-4,
                LINEARITY_ZERO=1e-10,
                LINEARITY_DEBUG=False,
                # CHANGED SAMPLE DIFF ZERO
                SAMPLE_DIFF_ZERO=1e-13):
    """If dataset==None: random input point. If dataset='CIFAR10' use input point from CIFAR10 test data."""

    sampleL = []
    sampleR = []
    
    # record the time needed to find critical points
    tFindCrt = 0.0 
    # record the time needed for the sign recovery
    tSignRec = 0.0

    while True:

        # ==========
        #  Critical point
        # ==========
        #
        starttime = time.time()
        xi = blackbox.findCorner(weights, biases, inputShape, [neuronId], targetValue=0, dataset=dataset)
        # Get number of active neurons in each hidden layer
        yi = xi
        active = []
        for lyr in range(layerId - 1):
            yi = np.matmul(yi, weights[lyr]) + biases[lyr]
            yi *= (yi > 0)
            active.append(len(yi[yi > 0]))
        active = np.array(active)
        stoptime = time.time()
        tFindCrt += stoptime - starttime

        # MM, bb = deti.dnn.getLocalMatrixAndBias(weights[:-1], biases[:-1], xi)
        # print(np.min(active), np.linalg.matrix_rank(MM))

        # ==========
        #  Energy-maximising wiggle
        # ==========
        starttime = time.time()
        #
        # Get orthogonal basis for the input vector space for the target
        # layer and restrict its dimension to that of the minimum dimension
        # in previous layers
        B, diffs = blackbox.getOrthogonalBasisForInnerLayerSpace(xi, weights, biases, layerId - 1, EPS_IN)
        if layerId > 1:
            B = B[:np.min(active)]
        # Get projection of the neuron's signature onto the space above
        proj = getProjection(weights[-1][:, neuronId], B)
        signaturesProj = np.array([proj * (np.abs(proj) > EPS_ZERO)])
        # Get wiggle
        try:
            wigglesi = getWigglesProjection(weights[-1], signaturesProj, diffs, EPS_IN, EPS_LYR)
        except Exception:
            continue

        # ==========
        #  Check linearity
        # ==========
        def gamma(x):
            return model(np.array([xi + wigglesi[0] * x]), training=False).numpy()[0][0] #blackbox(xi + wigglesi[0] * x)[0]

        if not (isLinear(gamma,  0.0, 1.0, eps=LINEARITY_EPS, tol=LINEARITY_ZERO, debug=LINEARITY_DEBUG)) or \
           not (isLinear(gamma, -1.0, 0.0, eps=LINEARITY_EPS, tol=LINEARITY_ZERO, debug=LINEARITY_DEBUG)):
            continue

        # ==========
        # Evaluate DNN
        # ==========
        #
        fx = model.predict(np.array([xi - wigglesi[0], xi + wigglesi[0], xi]))

        # ==========
        # Samples
        # ==========
        #
        sL = np.linalg.norm(fx[0] - fx[2])
        sR = np.linalg.norm(fx[1] - fx[2])
        # Check that samples are "far" from each other
        if (np.abs(sL - sR) < SAMPLE_DIFF_ZERO):
            continue
        # Collect samples
        sampleL.append(sL)
        sampleR.append(sR)
        
        stoptime = time.time()
        tSignRec += stoptime - starttime

        if len(sampleL) == nExp:
            break

    sampleL = np.array(sampleL)
    sampleR = np.array(sampleR)

    # 4: Number of experiments that decided sign +
    m4 = np.sum((sampleL / sampleR) < 1.0)

    signm4 = (-2.0 * (m4 < nExp / 2) + 1.0) if m4 != (nExp / 2) else 0.0

    return signm4, nExp - m4, m4, sampleL, sampleR, tFindCrt, tSignRec



if __name__=='__main__':
    logger.info("""
    # ---------------------------------------------------
    # This is DETI sign recovery using energy deposition.
    # ---------------------------------------------------   
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

    # Number of neurons in target layer
    nNeurons = len(biases[-1])

    # Target all neurons if None is specified
    if args.tgtNeurons is None:
        args.tgtNeurons = np.array(range(nNeurons))
    else: 
        args.tgtNeurons = [int(value) for value in args.tgtNeurons]
    logger.info(f"Signs will be recovered for neuronIDs: \n\t {args.tgtNeurons}.")

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
    expNeg = []
    expPos = []
    rows = []

    # WHITEBOX: Get the real signs to be able to control our results:
    whiteSignsLayer = whitebox.getRealSigns(model, args.layerID)

    logger.info("""
    # NEURON-BY-NEURON SIGN RECOVERY (parallelizable)
    # ---------------------------------------------------""")
    for neuronId in args.tgtNeurons:

        # start timer
        starttime = time.time()

        # -------- run the actual sign recovery --------
        signm4, nExpNeg, nExpPos, sampleL, sampleR, tFindCrt, tSignRec = recoverSign(model, weights, biases,
                                                                                       args.layerID,
                                                                                       neuronId,
                                                                                       inputShape,
                                                                                       nExp = args.nExp, 
                                                                                       dataset=args.dataset)

        # stop timer
        stoptime = time.time()

        nExpMax = max(nExpNeg, nExpPos)
        expNeg.append(nExpNeg)
        expPos.append(nExpPos)

        # ---------------------------------------------------
        # Load whitebox information to check the sign recover
        # ---------------------------------------------------
        whiteIsCorrect = signm4 == whiteSignsLayer[neuronId]
        whiteResult = "OK" if whiteIsCorrect else "NO <====== Failure!"
        whiteRealSign = '+' if (whiteSignsLayer[neuronId] > 0) else '-'

        # ---------------------------------------------------
        # Log results
        # ---------------------------------------------------
        runtime = stoptime-starttime
        logger.info(f"NeuronID: {neuronId} \t -:{nExpNeg}, +:{nExpPos}, \t ratio ({nExpMax / args.nExp}) \t runtime:{runtime:.2f} seconds \t White-box evaluation: real sign {whiteRealSign} ==> sign recovery={whiteResult}")

        rows.append({'modelID': modelname,
                     'layerID': args.layerID,
                     'neuronID': neuronId,
                     'realSign': whiteRealSign,
                     'metric4Minus': nExpNeg,
                     'metric4Plus': nExpPos,
                     'percentage': nExpMax / args.nExp,
                     'isRecoveredCorrectly': whiteIsCorrect,
                     'tFindCrit': tFindCrt, 
                     'tSignRec': tSignRec,
                     'recoveryTimeSeconds': stoptime - starttime
                     })


        logger.debug(f"Saving results to {filename_md} and {filename_pkl}...")
        df = pd.DataFrame(rows)
        df.to_pickle(filename_pkl)
        df.to_markdown(filename_md)

        filename_np = savePath + f"neuronID_{neuronId}_samples.npz"
        logger.debug(f"Saving sign evaluations to {filename_np}...")
        # load using
        # ... data = np.load(..)
        # ... data["samplesL"]
        data = {"samplesL": sampleL,
                "samplesR": sampleR}
        np.savez(filename_np, **data)

    expNeg = np.array(expNeg)
    expPos = np.array(expPos)