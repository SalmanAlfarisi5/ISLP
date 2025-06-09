## Why not Linear Regression?
Qualitative <-> Categorical
We can't use regression for classification since it will assume some kind of ordering in the response variable. We also will assume some different measures between the response variables this way. 
If the response is binary, then we can use linear regression with dummy variable $Y$. 
Regression method can't accommodate qualitative responses with more than two classes. 
Regression method will not provide meaningful estimates of the conditional probability since the value can be more than 1 or less than zero. 
## Logistic Regression
Use probability, values will be between 0 and 1.
### Logistic Model
Logistic Function:
$$p(X) = \frac{e^{\beta_0 + \beta_1 X}}{1 + e^{\beta_0 + \beta_1 X}}$$
The odds:
$$\frac{p(X)}{1-P(X)} = e^{\beta_0 + \beta_1 X}$$ Taking logarithm, we get
$$\log\left( \frac{p(X)}{1 - p(X)} \right) = \beta_0 + \beta_1 X$$
### Estimating Regression Coefficients
We use likelihood function such that plugging the estimates of the parameter into model for $P(X)$ yields a number close to one for defaulted individuals and close to zero for others.
$$\ell(\beta_0, \beta_1) = \prod_{i : y_i = 1} p(x_i) \prod_{i' : y_{i'} = 0} \left(1 - p(x_{i'})\right)$$We need to maximize the above likelihood function.
### Making Prediction
Simply plug-in the value of X to the formula. 
### Multiple Logistic Regression
$$\log\left( \frac{p(X)}{1 - p(X)} \right) = \beta_0 + \beta_1 X_1 + \cdots + \beta_p X_p$$
$$p(X) = \frac{e^{\beta_0 + \beta_1 X_1 + \cdots + \beta_p X_p}}{1 + e^{\beta_0 + \beta_1 X_1 + \cdots + \beta_p X_p}}$$
### Multinomial Logistic Regression
Softmax  

