## Maximal Margin Classifier
### Hyperplane
In a *p*-dimensional space, a hyperplane is a flat affine subspace of dimension *p-1*. Equation of hyperplane $$\beta_0 + \beta_1 X_1 + \beta_2 X_2 + \cdots + \beta_p X_p = 0$$
defines a *p*-dimensional hyperplane. If a point $X = (X_1, X_2, \dots, X_p)^T$ in *p*-dimensional space satisfies the above equation, then *X* lies on the hyperplane.
### Classification Using a Separating Hyperplane
The properties are:
$$\beta_0 + \beta_1 X_1 + \beta_2 X_2 + \cdots + \beta_p X_p > 0 \quad \text{if} \quad y_i = 1$$
$$\beta_0 + \beta_1 X_1 + \beta_2 X_2 + \cdots + \beta_p X_p < 0 \quad \text{if} \quad y_i = -1$$
We ultimately get linear decision boundary
### Maximal Margin Classifier
Also known as Optimal Separating Hyperplane which is the separating hyperplane that is farthest from training observations. The minimal distance from observation to hyperplane is known as margin. The maximal margin hyperplane has highest margin or the hyperplane that has the farthest minimum distance to training observations. It can led to overfitting when *p* is large.
If $\beta_0, \beta_1, \dots, \beta_p$ are the coefficients of maximal margin hyperplane, then the sign of $$f(x^*) = \beta_0 + \beta_1 x_1^* + \beta_2 x_2^* + \cdots + \beta_p x_p^*$$ will be the class of the test observation. The observations that are closest to the maximal margin hyperplane or has the lowest margin are called support vectors. Maximal margin hyperplane depends directly on support vectors but not on the other observations. 
### Construction of Maximal Margin Classifier
The maximal margin hyperplane is the solution to the optimization problem:
$$
\begin{aligned}
&\max_{\beta_0, \beta_1, \ldots, \beta_p, M} \quad M \\
&\text{subject to} \quad \sum_{j=1}^{p} \beta_j^2 = 1, \\
&\quad\quad y_i\left(\beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + \cdots + \beta_p x_{ip}\right) \geq M \quad \forall\, i = 1, \ldots, n.
\end{aligned}
$$
The constraints ensure that each observation is on the correct side of the hyperplane and at least the distance to the hyperplane is *M*. Hence, *M* is the margin of our hyperplane and the optimization problem chooses the parameter to maximize the margin *M* which is definition of maximal margin hyperplane. 
### Non-Separable Case
In most case no separating hyperplane exists and so there is no maximal margin classifier. Hence the optimization problem has no solution. 
## Support Vector Classifier 
The fact that maximal margin hyperplane is extremely sensitive to a change in a single observations suggests that it may overfit training data. 
### Overview
Classifier based on hyperplane that does not perfectly separate the two classes in exchange for:
- Greater robustness to individual observation
- Better classification of most of the training observations

Support vector classifier or soft margin classifier could misclassify a few training observations in order to perform better for the remaining observations. 
### Details
We need to solve this optimization problem:
$$\begin{aligned}
&\max_{\beta_0, \beta_1, \ldots, \beta_p,\, \epsilon_1, \ldots, \epsilon_n,\, M} \quad M \\
&\text{subject to} \quad \sum_{j=1}^{p} \beta_j^2 = 1, \\
&\quad\quad y_i \left( \beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + \cdots + \beta_p x_{ip} \right) \geq M(1 - \epsilon_i), \\
&\quad\quad \epsilon_i \geq 0,\quad \sum_{i=1}^{n} \epsilon_i \leq C.
\end{aligned}$$
where *C* is non-negative tuning parameter. $\epsilon_1, \dots, \epsilon_n$ are slack variables. We simply plug in our text *x* to the obtained estimated formula from the optimization problem. If $\epsilon_i = 0$, then the *i*th observation is on the correct side of the margin, if $0 < \epsilon_i < 1$, then the *i*th observation on the wrong side of the margin. If $\epsilon_i > 1$, then it is on the wrong side of the hyperplane. 
## Support Vector Machines
### Classification with Non-Linear Decision Boundaries
We could use squares of the features, cubic of the features and any other non-linear transformation of the features to find the suitable boundary.
### Support Vector Machine
Support Vector Machine (SVM) is an extension of the support vector classifier that results from enlarging the feature space in a specific way, using kernels. Kernel is a function that quantifies the similarity of two observations. 
Linear kernel
$$K(x_i, x_{i'}) = \sum_{j=1}^{p} x_{ij} x_{i'j},$$
Polynomial Kernel
$$K(x_i, x_{i'}) = \left( 1 + \sum_{j=1}^{p} x_{ij} x_{i'j} \right)^d.$$
When support vector classifier is combined with non-linear kernel, the resulting classifier is known as support vector machine. 
$$f(x) = \beta_0 + \sum_{i \in \mathcal{S}} \alpha_i K(x, x_i).$$
radial kernel
$$K(x_i, x_{i'}) = \exp\left(-\gamma \sum_{j=1}^{p} (x_{ij} - x_{i'j})^2 \right).$$
## SVM More than Two Classes
### One-Versus-One Classification
Construct $C_2^K$ SVMs, find majority of class in which the new data point is assigned to.
### One-Versus-All Classification
Fit *K* SVMs each time comparing one of the *K* classes to the remaining *K-1* classes. 
## Relationship to Logistic Regression 



