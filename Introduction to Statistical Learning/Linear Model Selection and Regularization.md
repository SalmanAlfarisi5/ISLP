 - Prediction Accuracy
If *n* >> *p*, number of observations is much larger than features, least squares will have low variance and low bias. However, if *n* is not much larger than *p*, there can be lot of variability in least square fit, resulting in overfitting. If *p* > *n*, then there is no lo unique least squares coefficient estimate, means infinitely many solutions, which results in high variance. By constraining or shrinking estimated coefficients, we can reduce variance with negligible increase in bias which improves the accuracy.
 - Model Interpretability
There could be useless features which add unnecessary complexity to the resulting model. We could simply remove those variables. 

There are many methods:
- Subset Selection
Find subset of the features that are believed to be related to the response, then fit the model to these subset of features.
- Shrinkage
Estimated coefficients are reduced to almost zero or even zero depending on the type of the *shrinkage* or *regularization*, hence can perform variable selection. 
- Dimension Reduction
Project *p* predictors into *M* dimensional subspace where *M* < *p*. This is achieved by computing *M* different linear combinations or projections of the variables. Then, *M* projections are used as predictors to fit a linear regression model by least squares. 
## Subset Selection
### Best Subset Selection
We fit the model $2^p$ times for each combination of the subset. Then, find best model from all the resulting model. Computationally expensive.
### Stepwise Selection
- Forward Stepwise Selection
Start from no predictor, find among *p* predictors, which one give smallest *RSS* or highest $R^2$, continue to 2 predictors all the way until all predictors. We can use threshold to decide whether we will add one more predictor or not. *n < p* is not a problem in forward stepwise selection. 
- Backward Stepwise Selection
Start from using all predictors, iteratively remove least useful predictor, one-at-a-time. We can also use threshold to decide whether we want to remove the least useful predictor at step *k*. *n* should be larger than *p* so that full model can be fit. 
- Hybrid Approaches
Simply combine forward stepwise and backward stepwise. After adding each new variable, we can remove any variables that is not relevant.

We can also use validation set or CV to predict test error and choose which model to use. 
### Choosing Optimal Model
The problem is model containing all f the predictors will always have smallest RSS and largest $R^2$, since these quantities are related to training error. Instead, we want a model with low test error. We can use validation set or cross-validation approach. We can also indirectly estimate test error by making adjustment to training error to account for the bias due to overfitting. We can also compute $C_p$, AIC, BIC, and Adjusted $R^2$.
#### $C_p$, AIC, BIC, and Adjusted $R^2$ 
$$C_p = \frac{1}{n} \left( \mathrm{RSS} + 2d\hat{\sigma}^2 \right),$$
$$AIC = \frac{1}{n} \left( \mathrm{RSS} + 2d\hat{\sigma}^2 \right),$$
$$BIC = \frac{1}{n} \left( \mathrm{RSS} + log(n)d\hat{\sigma}^2 \right),$$
$$\text{Adjusted } R^2 = 1 - \frac{\text{RSS}/(n - d - 1)}{\text{TSS}/(n - 1)}.$$
#### Validation and Cross-Validation
Simply do validation set or cross validation for each of the k predictor model.
## Shrinkage Methods
We can simply add regularization instead of having to choose the *d* features manually.
### Ridge Regression
We minimize:
$$\sum_{i=1}^{n} \left( y_i - \beta_0 - \sum_{j=1}^{p} \beta_j x_{ij} \right)^2 + \lambda \sum_{j=1}^{p} \beta_j^2 = \text{RSS} + \lambda \sum_{j=1}^{p} \beta_j^2,$$
Instead of:
$$\text{RSS} = \sum_{i=1}^{n} \left( y_i - \beta_0 - \sum_{j=1}^{p} \beta_j x_{ij} \right)^2.$$
$\lambda$  is tuning parameter. 
$\lambda \sum_{j} \beta_j^2$ is penalty and small when $\beta$ close to zero or coefficients are small. 
As $\lambda$ increases, flexibility of ridge regression fit decreases since it punishes model with high coefficients. As $\lambda$ increase, variance will decrease since model become simple and bias will slightly increase. $\lambda$ = 0 is simply least square. 
Ridge regression works best in situations where least squares estimates have high variance.
Ridge regression will include all *p* predictors in final model and most of the coefficients won't be zero unless the tuning parameter is so high. It won't cause a problem for prediction accuracy, but it does for interpretability. 
Another formulation for Ridge Regression
$$\min_{\boldsymbol{\beta}} \left\{ \sum_{i=1}^{n} \left( y_i - \beta_0 - \sum_{j=1}^{p} \beta_j x_{ij} \right)^2 \right\}
\quad \text{subject to} \quad \sum_{j=1}^{p} \beta_j^2 \leq s,$$

### Lasso Regression
$$\sum_{i=1}^{n} \left( y_i - \beta_0 - \sum_{j=1}^{p} \beta_j x_{ij} \right)^2 + \lambda \sum_{j=1}^{p} |\beta_j| = \text{RSS} + \lambda \sum_{j=1}^{p} |\beta_j|.$$
Lasso uses $\ell_1$ penalty or $\ell_1$ norm which is simply the sum of absolute value. 
Ridge uses $\ell_2$ penalty or $\ell_2$ norm which is the sum of squares of the coefficients. 
Lasso shrinks coefficient estimates towards zero event to exactly zero when tuning parameter is sufficiently large which result in model which is easier to interpret compared to those produced from Ridge Regression. Lasso hence yields sparse model which only involve subset of variables. 
Another formulation for Lasso Regression
$$\min_{\boldsymbol{\beta}} \left\{ \sum_{i=1}^{n} \left( y_i - \beta_0 - \sum_{j=1}^{p} \beta_j x_{ij} \right)^2 \right\}
\quad \text{subject to} \quad \sum_{j=1}^{p} |\beta_j| \leq s,$$
### Selecting Tuning Parameter
We can use validation set or CV to select the optimal tuning parameter. 
## Dimension Reduction Methods
We transform the predictors and fit a least squares model using transformed variables. We reduce the dimension from *p+1* to *M+1* coefficients. If *p* is large relative to *n*, selecting a value of *M* << *p* can significantly reduce the variance of fitted coefficients. 
### Principal Components Regression
#### PCA
The first principal component direction of the dataset is that along which the observations vary the most. Another interpretation of PCA: First principal component vector defines line that is as close as possible to the data or in another word minimizes sum of squared perpendicular distances between each point and the line. Second Principal component is perpendicular to the first principal component direction. 
#### PCR
Construct the first *M* principal components then use these components as predictors in linear regression model that is fit using least squares. Often small number of principal components suffice to explain most of the variability in the data and relationship with the response. 
Usually we need to standardize each predictor before generating the principal components to ensure all variables on the same scale. In the absence of standardization, high-variance variables will tend to play larger role in principal components obtained. 
### Partial Least Squares
The PCR way is unsupervised way of identifying principal components. On the other hand, Partial Least Squares (PLS) use supervised method. PLS consider both the responses and the predictors. 
## Considerations in High Dimensions
### High-Dimensional Data
Datasets containing more features than observations. 
### What Goes Wrong in High Dimensions?
Data will be fit exactly by the model which can overfit on unseen dataset 
### Regression in High Dimensions
We need to fit less flexible model than least squares, like forward stepwise selection, ridge regression, lasso regression and PCR.
Key principle analysis of high-dimensional data:
- Regularization or Shrinkage plays a key role in high-dimensional problems
- Appropriate tuning parameters selection is crucial for good predictive performance. 
- Test error tends to increase as dimensionality of the problem increases unless the additional features are truly associated with the responses. 
### Interpreting Results in High Dimensions 
