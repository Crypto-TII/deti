_This repository contains the code for the manuscript "Polynomial Time Cryptanalytic Extraction of Neural Network Models"._

# Sign Recovery Attack on ReLU-based Deep Neural Networks

Implementation of an attack to recover the signs of a ReLU-based deep
neural network (DNN) in polynomial time. 

# Caveats 
### Infinite Numerical Precision
Note that in our manuscript we assume infinite numerical precision. In practice our attack works using 64-bit arithmetic. 
If you are trying to attack a TensorFlow model with lower precision, our attack might run into problems. To check the precision of your model, you can use `model.weights[0].dtype` which should return `'float64'`.

### Implementation Status
This is our very first implementation of the ideas presented in our manuscript. The code is parallelizable at one point in each of our sign recovery techniques SOE, Neuron Wiggle and Last Hidden Layer. 
These parallelizations are not implemented as of now. If you look into the source code, the places where we assume that parallelization is possible are clearly hightlighted, and it should be easy to see that the code can be parallelized at this point (for example in `for` loops). 

# Reproduce Attacks from our Manuscript

To reproduce the results of the attacks reported in our manuscript, please execute the following commands: 

## Attack on 784-128-1
Attack with SOE should return an (unparallelized) execution time of **(6.77+-0.04)s**.  
```
python -m deti.soe --model deti/models/unitary_784_128_1.keras --layerID 1 --runID 'soe'
```

Attack with Last Hidden Layer should return an (unparallelized) execution time of **(18.61+-0.05)s**.  
```
python -m deti.lastLayer --model deti/models/unitary_784_128_1.keras --layerID 1 --runID 'lastLayer'
```

## Attack on 100-200(x3)-10 
Attack with Neuron Wiggle should return a (parallelizable) runtime of about **(16.3+-0.4)s**, respectively **(18.8+-0.5)s** per neuron.
_(since our implementation is not parallelized we use only five randomly chosen neurons in this demo)_
```
python -m deti.neuronWiggle --model deti/models/unitary_100_200x3_10.keras --layerID 1 --runID 'neuronWiggle' --tgtNeurons 4 26 30 77 168 
python -m deti.neuronWiggle --model deti/models/unitary_100_200x3_10.keras --layerID 2 --runID 'neuronWiggle' --tgtNeurons 4 26 30 77 168 
```

Attack with Last Hidden Layer should return an (unparallelized) execution time of **(35.8+-0.2)s**.
```
python -m deti.lastLayer --model deti/models/unitary_100_200x3_10.keras --layerID 3 --runID 'lastLayer'
```

## Attack on 3072-256(x8)-10

Attack with SOE on layerID 1 (and 2) should return an (unparallelized) execution time of **(16+-1)s**.
```
python -m deti.soe --model deti/models/cifar10_rgb_8x256.keras --layerID 1 --runID 'soe'
```

Attack with Neuron Wiggle should return a (parallelizable) runtime of about **(182+-2)s** per neuron
(since our implementation is not parallelized we use only five randomly chosen neurons in this demo)
```
python -m deti.neuronWiggle --model deti/models/cifar10_rgb_8x256.keras --layerID 2 --runID 'neuronWiggle' --dataset 'CIFAR10' --tgtNeurons 4 26 30 77 168 
```

Attack with Last Hidden Layer should return an (unparallelized) execution time of **(189+-40)s**.
```
python -m deti.lastLayer --model deti/models/cifar10_rgb_8x256.keras --layerID 8 --runID 'lastLayer'
```

## Evaluate Accuracy on CIFAR10 

To evaluate the accuracy of our CIFAR10 network, please execute the following Python commands. You should obtain an accuracy of 0.5249 on the CIFAR10 test dataset.
```python
import tensorflow as tf
from keras.datasets import cifar10

model = tf.keras.models.load_model('deti/models/cifar10_rgb_8x256.keras')

def normalize_resize(image, label):
    image = tf.cast(image, tf.float32)
    image = tf.divide(image, 255)
    image = tf.image.resize(image, (32,32))
    return image, label

(trainX, trainy), (testX, testy) = cifar10.load_data()
testX, testy = normalize_resize(testX, testy)
testX = tf.keras.layers.Flatten()(testX)
model.evaluate(testX, testy)
# Expected result: 313/313 [==============================] - 2s 1ms/step - loss: 1.3801 - accuracy: 0.5249
```

# Create a new Unitary Balanced DNN
```python 
import deti
inputShape = (200,)
neuronsHiddenLayers = [200] * 8
outputs = 10

model = deti.unitarydnn.newRandomBalancedModel(inputShape, neuronsHiddenLayers, outputs, 1.0)
model.save('deti/models/randomdnn_8x200_10.keras')
```

# Dependencies

The code execution relies on standard Python modules such as NumPy, Pandas, and TensorFlow. If you start from an empty Python Anaconda environment, the following installation should be sufficient:

```
conda create -n tf-gpu tensorflow-gpu
conda activate tf-gpu
conda install -c nvidia cuda-nvcc
conda install pandas
conda install tabulate
conda install numpy
```

# General Usage

### SOE 
```
python -m deti.soe --model deti/models/cifar10_rgb_8x256.keras --layerID 2 --runID 'soe'
```

### Neuron Wiggle
```
python -m deti.neuronWiggle --model deti/models/cifar10_rgb_8x256.keras --layerID 2 --runID 'neuronWiggle'
```

### Last Hidden Layer
```
python -m deti.lastLayer --model deti/models/cifar10_rgb_8x256.keras --layerID 8 --runID 'lastLayer'
```
