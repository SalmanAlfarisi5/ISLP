Source: https://www.youtube.com/watch?v=alfdI7S6wCY
State of the art -> Most advanced development in particular field
Deep Learning -> Machine Learning that uses artificial neural network to learn from data. These neural networks are inspired by structure of human brain.
Machine Learning -> Ability to learn without explicitly being programmed
Artificial Intelligence -> Techniques that enables computers to mimic human behavior. 

Why use Deep Learning now?
- Big Data -> Larger datasets, easier collection and storage
- Hardware Advancement -> GPU, Massively parallelizable
- Software -> Open source libraries, new models 

Perceptron
![[Pasted image 20250616212309.png]]

Activation Functions to introduce non-linearities 
![[Pasted image 20250616212421.png]]

Simplified Perceptron
![[Pasted image 20250616213313.png]]

Multi Output Perceptron
![[Pasted image 20250616213350.png]]

Dense Layers -> All inputs are densely connected to all outputs

Single Layer Neural Network
![[Pasted image 20250616214218.png]]
![[Pasted image 20250616214343.png]]

Deep Neural Network
![[Pasted image 20250616214522.png]]

Quantifying Loss -> Difference between predicted and actual
Empirical Loss -> Total loss over entire dataset
Training Neural Network -> Finding network weights that achieve lowest loss
![[Pasted image 20250616215843.png]]
Loss is a function of network weights

Gradient Descent 
- Initialize weights randomly
- Loop until convergence:
	- Compute gradient -> Change of loss relative to the change of the weights
	- Update weights -> Subtract gradient times learning rate from the weights
- Return weights

Backpropagation 
![[Pasted image 20250616220745.png]]![[Pasted image 20250616220820.png]]

Setting Learning rate 
- Too small then converges slowly and gets stuck in false local minima
- Too large then overshoot, unstable and diverge
- Adaptive Learning Rate -> Can be made larger or smaller depending on the gradient and weights:
	- SGD
	- Adam
	- Adadelta
	- Adagrad
	- RMSProp

Gradient Descent -> Compute gradient over the entire dataset
Stochastic Gradient Descent -> Compute gradient with 1 data point only
Mini batch Gradient Descent -> Compute gradient with 1 batch of data points

Problems in Machine Learning
![[Pasted image 20250616221750.png]]

Regularization -> Technique that constraints optimization problem to discourage complex models to improve generalization of model on unseen data

Dropout -> During Training, randomly set some activation to 0 so that it won't rely on any node. 
Early Stopping -> Stop training before we have change to overfit 

