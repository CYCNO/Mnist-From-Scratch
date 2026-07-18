<div align="center">

![](https://imgs.search.brave.com/kJxuALxSWYk1mnYxoMmeM_k0L6Rl40pFPWmOssedx9Y/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9jb254/LnJlYWR0aGVkb2Nz/LmlvL2VuL2xhdGVz/dC9faW1hZ2VzL01O/SVNUXzZfMC5wbmc)
# Mnist Prediction From Scratch

<a href="https://githubtocolab.com/CYCNO/Mnist-From-Scratch/blob/main/main.ipynb">
<img src="https://img.shields.io/badge/open%20in-googlecolab-red?logoColor=orange&logo=googlecolab&style=for-the-badge">
</a>
</div>

It is a Mnist prediction model written from scratch in python, it uses numpy for matrix multiplication, the reason for that is, python is very slow in doing fast multiplication of matrix, since it is not primary designed to do that.

## Dataset
Uses a subset of MNIST:
- Training: 10,000 images
- Testing: 5,000 images

you can download it from [here](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv) and put it in dataset/* which will make it look something like this
```
├── dataset
│  ├── mnist_test.csv
│  └── mnist_train.csv
```

## Architecture
It uses 2 hidden layers of 10 neurons each, and the architecture looks like 

```
Layer 0 (input layer): 28x28 = 784 neurons each for the pixels of the image
Layer 1 (First Hidden Layer):  10 neurons
Layer 2 (Second Hidden Layer): 10 neurons again.
Layer 3 (Output Layer): 10 outputs for values ranging from 0-9 
```

## Activation Functions
For Hidden Layers: it uses ReLU activation, I tried with sigmoid too but it was not giving any significant improvement 
```sh
ReLU       : {x, x > 0
              0, otherwise

d(ReLU)    : {1, x > 0
              0, otherwise
```

For Output Layer: It uses softmax to distribute the probability of all the outputs
```sh
softmax(S) : e^(x) / sum(e^(x))
```

## Backpropagation
To find the gradient of each parameter, it uses backprop, it find derivative(d_) for 3 different parameter, Z (the value of the neuron), W (weights) and B (biases). later this gradients are used to help in updating the neurons parameters 

The math behind it (spoiler: chain rule)
```sh
dZ[i]: W[i+1] @ dZ[i+1] * ReLU_derive(Z[i])   where @ = dot multiplication
dW[i] = (1 / m) * (dZ[i] @ A[i])                    * = element wise multiplication    
dB[i] = (1 / m) * sum(dZ[i])                        i = current layer
```
> since we going from forward to backward `i+1` means the next layer neurons parameters that has already been calculated

## Training
10,000 sample for 1000 epochs gives results as follow
|Hidden Layer Neurons|learning rate|accuracy (testing)|
|---------|-|-----|
| 10| 0.2| 91.0%|
| 16| 0.2| 92.2%|
| 32| 0.3| 93.7%|
| 64| 0.3| 94.3%|

> the accuracy is measured on data that model have never seen before

so as we can observer that we increase in amount of neurons in Hidden Layer, also increase the accuracy, there are also other ways to increase accuracy, like increasing learning rate or increasing its epochs.

## Ending Notes
you can also tinker little bit of model of mnist, since i have try to make it as customizable as possible, you can change its layers, its neurons, epochs or learning rate, as per your need

```python
mnist = Mnist([784, 10, 10, 10], X_train, Y_train)
mnist.train(iterations=2000, lr=0.1)
```

Hope you guys learned something new from this. Thank You

## Reference
- [Mnist in CSV](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)
- [Backprop Note](https://github.com/tsoding/ml-notes/blob/master/papers/grad.pdf)
