## Simple Linear Regression
Single predictor variable $Y$, assuming  linear relationship between $X$ and $Y$ 
$$Y \approx \beta_0 + \beta_1X$$Sometimes called regressing $Y$ on $X$ or $Y$ onto $X$. Once we have obtained $\hat{\beta_0}$ and $\hat{\beta_1}$ from our training data for the model coefficients, we can use the model for prediction:
$$\hat{y} = \hat{\beta_0} + \hat{\beta_1}x$$

### Estimating the Coefficients
We try to minimize Residual sum of squares (RSS) which led us to this formula:
$$\hat{\beta}_1 = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{n} (x_i - \bar{x})^2}, \\
\hat{\beta}_0 = \bar{y} - \hat{\beta}_1 \bar{x},$$
### Assessing Accuracy of Coefficient Estimates
## Multiple Linear Regression
## Other Considerations
### Qualitative Predictors
### Extensions of Linear Model
Interaction effect -> change in one variable might alter the association between other variable and response. 
### Non-linear Relationships
*Polynomial Regression* 
### Potential Problems
- Non-linearity of the response-predictor relationships
Plot the residual against fitted values or $\hat{y}$. If there is a pattern in the plot, then it's might be the true function is non-linear. Try use quadratic terms, if the pattern disappears, then quadratic term is better fit than the linear, we could also try logarithm and square root.
- Correlation of error terms
Try to plot residual against time, if the errors are uncorrelated, then there shouldn't be any pattern in the plot.
- Non-constant variance of error terms
*heteroscedasticity*. We can transform the $Y$, for example by taking the logarithm or taking the square root. 
- Outliers
Residual plot can be used to identify outliers, we ca also plot the *studentized residuals*, obtained from dividing each residual by its estimated standard error. We might just remove the outlier, but the outlier might also indicate our model is not good enough.
- High-leverage points
High leverage observations tend to have sizable impact on estimated regression line. It is harder to note in multiple regression since the high-leverage point might be on the normal range if we consider each feature separately, but when we plot all the features, then it lies outside the normal area range. We can calculate *leverage statistics* which large values indicated observation with high leverage.
- Collinearity
Try to look at correlation matrix of the predictors. We also need to be cautious of *multicollinearity*. Instead of inspecting correlation matrix, we can compute *variance inflation factor (VIF)*. VIF is ratio of variance of $\hat{\beta_j}$ when fitting full model divided by the variance of $\hat{\beta_j}$ if fit on its own. VIF exceeds 5 or 10 indicates problematic amount of collinearity. We can simply drop the problematic variables from the regression. We can also combine the collinear variables together into a single predictor. 
## Linear Regression vs KNN
KNN suffer from *curse of dimensionality*, that is K observations that are nearest to a test observation might be very far from the test set when the features are many which lead to poor prediction in KNN.
In general, parametric methods will tend to outperform non-parametric approaches when there is a small number of observations per predictor / feature. Also, linear regression has more interpretability than KNN. 

I decided not to continue this for linear regression since I have taken a module (ST3131) in NUS which covers this part in detail in one whole semester. Since I create this summary solely to help me study not to create a good summary of the book, I decided to omit most parts in this chapter with the assumption that I have understand it. Sorry :v
