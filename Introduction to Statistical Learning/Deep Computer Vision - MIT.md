Impact:
- Facial Detection & Recognition 
- Self-Driving Cars
- Medicine, Biology, Healthcare
- Accessibility

To computers, image are numbers
Tasks in Computer Vision:
- Regression -> output variable takes continuous value
- Classification -> output variable takes class label

Feature Detection -> identify features that distinguish classes. 
Learning Feature Representations -> Learn hierarchy of features directly from data instead of manual.

If we use fully connected neural network for image processing, we will loss spatial information due to flattening and we will have a lot of parameters. 

To use spatial structure, we connect patches of input layer to single neuron in subsequent layer. We can use sliding window here.

Feature Extraction with Convolution
- Apply a set of weights / filter to extract local features
- Use multiple filters to extract different features
- Spatially share parameters of each filter

Convolution -> Preserve spatial information that is present in the image by breaking image into smaller sub-images and find key information in the sub-images. 

Convolution Neural Network (CNN)
- Convolution -> apply filters to generate feature maps or convolution
- Non-Linearity -> activation function
- Pooling -> Downsampling operation on each feature map. 

For a neuron in hidden layer:
- Take inputs from patch
- Compute weighted sum
- Apply bias

CNNs: Spatial Arrangement of Output Volume
Layer Dimensions: $h \times w \times d$, where *h* and *w* are spatial dimensions and *d* (depth) is number of filters
Stride: filter step size
Receptive Field: locations in input image that a node is path connected to

Non-Linearity -> Apply after every convolution
Pooling -> Downscaling the features size

CNNs for Classification: Feature Learning
- Feature Learning
	- Learn features in input image through convolution
	- Use non-linearity through activation function
	- Reduce dimensionality and preserve spatial invariance with pooling
- Classification
	- CONV and POOL layers output high-level features of input
	- Fully connected layer use these features for classifying input image
	- Express output as probability of image belonging to a particular class