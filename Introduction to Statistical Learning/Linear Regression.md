## Simple Linear Regression
Single predictor variable $Y$, assuming  linear relationship between $X$ and $Y$ 
$$Y \approx \beta_0 + \beta_1X$$Sometimes called regressing $Y$ on $X$ or $Y$ onto $X$. Once we have obtained $\hat{\beta_0}$ and $\hat{\beta_1}$ from our training data for the model coefficients, we can use the model for prediction:
$$\hat{y} = \hat{\beta_0} + \hat{\beta_1}x$$

### Estimating the Coefficients
We try to minimize Residual sum of squares (RSS) which led us to this formula:
$$\hat{\beta}_1 = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{n} (x_i - \bar{x})^2}, \\
\hat{\beta}_0 = \bar{y} - \hat{\beta}_1 \bar{x},$$
### Assessing Accuracy of Coefficient Estimates

I decided not to continue this for linear regression since I have taken a module (ST3131) in NUS which covers this part in detail in one whole semester. Sorry :v
