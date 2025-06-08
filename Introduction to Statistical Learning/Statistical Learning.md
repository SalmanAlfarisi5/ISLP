### What is Statistical Learning
Inputs -> predictors, independent variables, features, variables (**X**)
Outputs -> response, dependent variable (**Y**)
General formula is $Y = f(X) + \epsilon$    
$f$ is fixed but unknown function of $X$, $f$ represents systematic information that $X$ provides about $Y$
$\epsilon$ is random error term, which is independent of $X$ and has mean zero. 
Statistical learning refers to set of approaches for estimating $f$ 
### Why Estimate f
- Prediction
$\hat{Y} = \hat{f}(X)$. There are errors in predictions, *reducible error* and *irreducible error*. *Reducible Error* is the inaccuracy in predicting $f$ using $\hat{f}$ which can be reduced by choosing appropriate model. *Irreducible error* came from $\epsilon$ which can't be predicted using $X$. 
$$\begin{aligned}
\mathbb{E}(Y - \hat{Y})^2 
&= \mathbb{E}[f(X) + \varepsilon - \hat{f}(X)]^2 \\
&= \underbrace{[f(X) - \hat{f}(X)]^2}_{\text{Reducible}} + \underbrace{\mathrm{Var}(\varepsilon)}_{\text{Irreducible}}
\end{aligned}$$
- Inference
Inference is used to answer these questions:
	- Which predictors are associated with the response
	- What is the relationship between response and each predictor
	- Can the relationship between $Y$ and each predictor be adequately summarized using linear equation
### How we Estimate f
The goal is to estimate unknown function $f$ by using $\hat{f}$ such that $Y \approx \hat{f}(X)$ for any observation $(X,Y)$ 
- Parametric Methods or model-based approach, we estimate the parameters of the model.
- Non-Parametric Methods, we don't make assumptions about the functional form of $f$. 

### Trade-off between Prediction Accuracy and Model Interpretability
There is a trade-off between flexibility and interpretability. When inference is the goal, there are clear advantages to using simple and inflexible model. When we are interested only in prediction, flexible model might be better in this case. 
### Supervised and Unsupervised
Supervised has response and unsupervised doesn't.
Semi-supervised learning -> some observations have responses and some don't
### Regression and Classification
Quantitative -> takes on numerical values.
Qualitative -> takes on values in one of $K$ different classes or categories
Regression -> Response is Quantitative
Classification -> Response is Qualitative
### Assessing Model Accuracy
- Measuring Quality of Fit 
$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} \left(y_i - \hat{f}(x_i)\right)^2,$$
### Bias-Variance Trade-off
$$\mathbb{E}\left( y_0 - \hat{f}(x_0) \right)^2 = \mathrm{Var}(\hat{f}(x_0)) + \left[\mathrm{Bias}(\hat{f}(x_0))\right]^2 + \mathrm{Var}(\varepsilon).$$
To minimize expected test error, we need to get low variance and low bias for the model since we can't interfere with *Irreducible Error*. 
Variance -> amount by which $\hat{f}$ would change if we estimated it using different training data set. In general, more flexible statistical methods have higher variance.
Bias -> Error that is introduced by approximating real-life problem which may be extremely complicated using simpler model. Generally, more flexible methods result in less bias. 
Initially, as flexibility increase bias decrease faster than variance increases hence lowering Test MSE. Then, bias will decrease slower than variance increase as the model becomes to flexible hence increasing test MSE.
Finding sweet spot where both the bias and variance are low is the recurring themes of the book.
In real life, it's impossible to compute test MSE, bias or variance. 
### Classification Setting
$$\frac{1}{n} \sum_{i=1}^{n} \mathbb{I}(y_i \ne \hat{y}_i).$$
which calculates the fraction of incorrect classifications.
### Bayes Classifier 
For each observation, assign the most likely class, given its predictor values. Find the largest conditional probability for class j.
$$\Pr(Y = j \mid X = x_0)$$
Bayes decision boundary -> the probabilities between the adjacent class is the same.
Bayes classifier produces lowest possible test error rate, called *Bayes error rate*.
Overall, Bayes error rate:
$$1 - \mathbb{E} \left( \max_j \Pr(Y = j \mid X) \right),$$
The Bayes error rate is analogous to irreducible error
### K-Nearest Neighbors
If we don't know the conditional distribution of $Y$ given $X$, computing Bayes classifier is impossible. Many approaches attempt to estimate the conditional distribution of $Y$ given $X$ and classify given observation to class with highest probability. Identify K closest point to the test point and estimate based on majority rule
$$\Pr(Y = j \mid X = x_0) = \frac{1}{K} \sum_{i \in \mathcal{N}_0} \mathbb{I}(y_i = j).$$
Then choose test observation with largest probability.
If K is very low, decision boundary very flexible the patterns don't correspond to Bayes Decision Boundary. In other words, low bias and high variance.
If K is very high, KNN becomes less flexible and produces decision boundary close to linear. In other words, low variance and high bias classifier. 
If we choose right K, KNN can converge to Bayes Classifier. 
Small K -> Flexible model
Large K -> Inflexible model
