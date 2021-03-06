# Deep Learning on Computation Accelerators - Final Project on Neural Arithmetic Logic Units

This is a PyTorch implementation of our project, the code is heavily based on [Kevin Zakka](https://github.com/kevinzakka/NALU-pytorch) implementation to [Neural Arithmetic Logic Units](https://arxiv.org/abs/1808.00508) by *Andrew Trask, Felix Hill, Scott Reed, Jack Rae, Chris Dyer and Phil Blunsom* and on [MNIST Convnets](https://github.com/pytorch/examples/blob/master/mnist) pytorch example.

<p align="center">
 <img src="loss_nac_fc.PNG" alt="Drawing", width=60%>
</p>

## API
First, to produce the data run: mnistArthimetic.m 

Second, to train the network run:
```python
from models import *

python nac_project.py

```

## Experiments

To produce the graphs of our experiments,run:

```python
python readLogFile.py
```

This should generate the following plot:

<p align="center">
 <img src="lossComp.PNG" alt="Drawing", width=60%>
</p>

