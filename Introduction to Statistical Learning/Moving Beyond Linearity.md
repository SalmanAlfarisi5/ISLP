- Polynomial Regression -> extends linear model by adding extra predictors like squares or cubic. 
- Step Functions -> cut the ranges of variables into *K* distinct regions in order to produce qualitative variable
- Regression Splines -> dividing range of *X* into *K* distinct regions, then apply polynomial function to each region, the polynomial here will be constrained such that they join smoothly at region boundaries. 
- Smoothing Splines -> minimizing residual sum of squares criterion subject to a smoothness penalty. 
- Local Regression -> The regions are allowed to overlap, but in smooth way.
- Generalized additive models allow us to extend the methods above to deal with multiple predictors.

## Polynomial Regression
$$y_i = \beta_0 + \beta_1 x_i + \beta_2 x_i^2 + \beta_3 x_i^3 + \cdots + \beta_d x_i^d + \epsilon_i,$$
## Step Functions
Break range of *X* into bins and fit different constant in each bin.
$$\begin{aligned}
C_0(X)   &= I(X < c_1), \\
C_1(X)   &= I(c_1 \leq X < c_2), \\
C_2(X)   &= I(c_2 \leq X < c_3), \\
&\ \, \vdots \\
C_{K-1}(X) &= I(c_{K-1} \leq X < c_K), \\
C_K(X)   &= I(c_K \leq X),
\end{aligned}$$
$$y_i = \beta_0 + \beta_1 C_1(x_i) + \beta_2 C_2(x_i) + \cdots + \beta_K C_K(x_i) + \epsilon_i.$$
## Basis Functions
Polynomial and Step Functions are special case of basis function approach. 
$$y_i = \beta_0 + \beta_1 b_1(x_i) + \beta_2 b_2(x_i) + \beta_3 b_3(x_i) + \cdots + \beta_K b_K(x_i) + \epsilon_i.$$
It is actually standard linear model with predictors $b_1(x_i), b_2(x_i), \cdots b_K(x_i)$. Hence, least squares can be used here.
## Regression Splines
### Piecewise Polynomial 
Instead of fitting high degree polynomial over the entire range of *X*, we fit piecewise  polynomial regression involves fitting separate low-degree polynomials over different regions of *X*. The points where the coefficients change are called *knots*.
### Constraints and Splines
Define constraints such that the function is continuous, the first derivative is continuous and also the second derivative).
### Spline Basis Representation
$$y_i = \beta_0 + \beta_1 b_1(x_i) + \beta_2 b_2(x_i) + \cdots + \beta_{K+3} b_{K+3}(x_i) + \epsilon_i,$$
Regression splines can be represented using basis model. 
### Choosing Number and Locations of the Knots
We can use cross validation in deciding number of knots. Also, it's make sense to place more knots at the range where the data varies. Also, some people just distribute the knots uniformly. 
### Comparison to Polynomial Regression
Polynomial might overfit since it's too flexible
## Smoothing Splines
To ensure smoothness, we minimize:
$$\sum_{i=1}^{n} (y_i - g(x_i))^2 + \lambda \int \left( g''(t) \right)^2 dt$$
$\lambda$ is tuning parameter that controls the roughness of the smoothing spline, and hence the effective degrees of freedom
## Local Regression
- Gather the fraction $s = \frac{k}{n}$ of training points whose $x_i$  are closest to $x_0$.
- Assign weight $K_{i0} = K(x_i, x_0)$ to each point in this neighborhood, the farther the point the lower the weight
- Fit a weighted least squares regression using the weights by finding estimated parameter that will minimize $$\sum_{i=1}^{n} K_{i0} (y_i - \beta_0 - \beta_1 x_i)^2.$$
- Fitted values at $x_0$ is given by $\hat{f}(x_0) = \hat{\beta_0} + \hat{\beta_1{x_0}}$ 
## Generalized Additive Models 
Generalized Additive Models (GAMs) provide a general framework for extending standard linear model by allowing non-linear functions of each of the variables while maintaining additivity.  
