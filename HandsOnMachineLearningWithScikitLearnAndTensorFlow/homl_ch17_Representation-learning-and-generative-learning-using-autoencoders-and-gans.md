# Chpater 17. Representation Learning and Generative Learning Using Autoencoders and GANs

Autoencoders (AE) are ANN capable of dense representations of the input data (*latent variables*) without supervision.
These codings are usually represented in lower dimensions, making AE powerful tools for dimensionality reduction for visualization and feature detectors for creating other models.

Generative Adversarial Networks (GAN) consist of two ANN: a *generator* that tries to create data that looks simillar to the training data, and a *discriminator* that tries to tell the real data from the fake data.
This architecture and training method are incredibly powerful and can create output that is incredibly realistic.

This chapter will explore both AE and GAN.


```python
import numpy as np
import pandas as pd 
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf 
import tensorflow.keras as keras

%matplotlib inline
np.random.seed(0)
sns.set_style('whitegrid')
```

## Efficient data representations


```python

```
