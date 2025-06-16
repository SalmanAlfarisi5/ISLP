## Single Layer Neural Network
We have one input layer, one hidden layer and one output layer. We choose parameters to minimize
$$\sum_{i=1}^n(y_i - f(x_i))^2$$
## Multilayer Neural Networks
There are more than one hidden layers.
### Convolutional Neural Network
The network first identifies low-level features in the input image, such as small edges, patches of color. These low-level features combined to form higher-level features, like the organs. Then the presence or absence of these high-level features contributes to the probability of any given output class.
### Convolution Layers
A convolution layer is made up of a large number of convolution filters, each of which is a template that determines whether a particular local feature is present in an image. A convolution filter relies on a simple operation, convolution which repeatedly multiplying matrix elements and then adding the results. If the submatrix of the original image resembles the convolution filter, then it will have a large value in the convolved image, otherwise it will have small value. 
- Color images have three channels represented by three dimensional feature map (array). Each channel is a two-dimensional feature map, one for red, one for green, and one for blue. A single convolution will also have three channels, one per color with same dimension but could be different weights. The results of the three convolutions are summed to form two-dimensional output feature map.
- If we use *K* different convolution filters at first hidden layer, we get *K* two-dimensional output feature maps which together are treated as a single three-dimensional feature map. Each of *K* output feature maps as separate channel of information, so now we have *K* channels in contrast to the three color channels of original feature map. 
- Typically apply ReLU activation function to the convolved image. This step is sometimes viewed as separate layer in the convolutional neural network, called detector layer.
### Pooling Layers
Pooling layer condense large image into smaller summary image. We can use max pooling or mean pooling
### Architecture of Convolutional Neural Network
![[Pasted image 20250615225836.png]] 
We do convolve and pool in sequence repeatedly then we flatten it at the end. 
- Each subsequent convolve layer takes three-dimensional feature map from previous layer and treats it like single multi-channel image.
- Since channel feature maps are reduced after each pool layer, we usually increase number of filters in the next convolution to compensate
- It's also possible to repeat several convolution layers before pool layer.
### Data Augmentation
Replicating each training image multiple times with each replicate randomly distorted in a natural way such that human recognition is unaffected. Typical distortions are zoom, shit, rotations and flips.
### Results Using Pretrained Classifier 
resnet50 -> https://arxiv.org/abs/1512.03385
## Document Classification
Need a way to featurize the words. Simplest way is bag-of-words model. Score each document for the presence or absence of each of the words. If the dictionary contains *M* words, meaning for each document, create a binary feature vector of length *M*. Bag-of-words model summarizes document by the word present but ignores the context. To take the context, we can:
- Bag-of-*n*-grams model. We can record the occurrences of more than one word, but pair of words to store the context.
- Treat the document as a sequence, taking account previous and subsequent words.
## Recurrent Neural Network
Many data sources are sequential in nature
- Text dataset
- Time series of temperature, rainfall, wind speed, air quality, and others. 
- Financial time series, here prediction is rather difficult.
- Audio dataset (speech, music and others). 
- Handwriting (Image data)

In a recurrent neural network (RNN), the input object *X* is a sequence. RNNs take advantage of sequential nature of input objects, like convolutional neural networks accommodate spatial structure of image inputs. The output *Y* can also be sequence but often is class. 
![[Pasted image 20250616164546.png]]
Suppose each vector $X_\ell$ has *p* components, $X_\ell^T = (X_{\ell1}, X_{\ell2}, \dots, X_{\ell p})$ and the hidden layer consists of *K* units  $A_\ell^T = (A_{\ell1}, A_{\ell2}, \dots, A_{\ell K})$, then the matrix:
- **W** is a $K \times (p+1)$
- **U** us a $K \times K$ matrix
- **B** is a $K + 1$ vector 

$$A_{\ell k} = g\left( w_{k0} + \sum_{j=1}^{p} w_{kj} X_{\ell j} + \sum_{s=1}^{K} u_{ks} A_{\ell - 1, s} \right),$$
$$
O_{\ell} = \beta_0 + \sum_{k=1}^{K} \beta_k A_{\ell k}
$$
Loss function for regression problems:
$$
(Y - O_L)^2
$$
With *n* inputs, the parameters are found by minimizing:
$$
\sum_{i=1}^{n} (y_i - o_{iL})^2 
= \sum_{i=1}^{n} \left( y_i - \left( \beta_0 + \sum_{k=1}^{K} \beta_k\, g\left( w_{k0} + \sum_{j=1}^{p} w_{kj} x_{iLj} + \sum_{s=1}^{K} u_{ks} a_{i, L-1, s} \right) \right) \right)^2.
$$
### Sequential Models for Document Classification
One-hot vs Embed
![[Pasted image 20250616172933.png]]
The embedding layer can be learnt trough neural network. We can also use pre-computed matrix in the embedding layer like word2vec or GloVe which preserve the semantic meaning in the embedding space.  
We could also use Long Term and Short Term Memory (LSTM). We consider hidden units both further back in time and closer in time. 
### Time Series Forecasting
Predicting stock prices is hard problem but predicting trading volume based on recent past history is more manageable. 
## When to Use Deep Learning
Typically we expect deep learning to be an attractive choice when the sample size of the training set is extremely large and when the interpretability of the model is not a high priority. Often, simple linear model tend to perform the same as complex deep learning, in this case we will use Occam's razor to just choose the simpler model for interpretability.
## Fitting Neural Network
### Back propagation
### Regularization and Stochastic Gradient Descent
### Dropout Learning
### Network Tuning
## Interpolation and Double Descent

I skipped some parts and this chapter is rather concise in the summary because I want to try the methods first.