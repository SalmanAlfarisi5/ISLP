Sequences Data or Time Series
- Audio
- Word, Text, and Natural Languages 
- Biological Sequences
- Video
- Stock Market
- Weather

One to One -> Binary Classification
Many to One -> Sentiment Classification
One to Many -> Image Captioning, Text Generation
Many to Many -> Machine Translation, Translation & Forecasting Music Generation

Neurons with Recurrence 
![[Pasted image 20250617083437.png]]

Recurrent Neural Network (RNNs)
RNNs have a state $h_t$, that is updated at each time step as sequence is processed. 
Apply a recurrence relation at every time step to process a sequence (same function and set of parameters are used at every time step)
$$h_t = f_W(x_t, h_{t-1})$$
- $h_t$ is cell state
- $f_W$ is a function with weights **W**
- $x_t$ is input
- $h_{t-1}$ is old state

$$h_t = tanh(W_{hh}^Th_{t-1} + W_{xh}^Tx_t)$$
$$\hat{y_t} = W_{hy}^Th_t$$

RNNs: Computational Graph Across Time
![[Pasted image 20250617084557.png]]

RNNs: Backpropagation Through Time
![[Pasted image 20250617094226.png]]

Computing gradient with respect to $h_0$ involves many factors of $W_{hh}$ and repeated gradient computation.

Many gradients > 1 -> exploding gradients
Many gradients < 1 -> vanishing gradients

Gating Mechanism in Neurons
Idea: use gates to selectively add or remove information within each recurrent unit 
Example: Gated Recurrent Unit (GRU), Long Short Term Memory (LSTM)

Bottleneck -> a phenomenon by which the performance or capacity of an entire system is severely limited by a single component

Limitations of RNNs
- Encoding Bottleneck
- Slow, no parallelization
- Not long memory 

Desired Capabilities
- Continuous Stream
- Parallelization
- Long Memory

Intuition Behind Self-Attention -> Attending to most important parts of an input
- Identify which parts to attend to
- Extract the features with high attention 

Self-Attention with Neural Network -> Identify and attend most important features from input
- Encode position information
![[Pasted image 20250617101225.png]]
- Extract query, key, value for search
![[Pasted image 20250617101309.png]]
- Compute attention weighting, attention score: similarity between each query and key
![[Pasted image 20250617101520.png]]
![[Pasted image 20250617101531.png]]
- Extract features with high attention
![[Pasted image 20250617101654.png]]

Self-Attention framework
![[Pasted image 20250617101814.png]]
