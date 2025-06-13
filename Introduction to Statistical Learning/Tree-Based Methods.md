 Stratifying or segmenting predictor space into number of simple regions. To make the prediction, we use mod or mean of training observations in the region which it belongs. 
## Basic of Decision Trees
### Regression Trees
Building Regression Trees:
- Divide predictor space into non-overlapping regions
- For every observation, the predicted value is the mean or median of the region.

The goal is find the regions division in which the RSS is minimized:
$$\sum_{j=1}^{J} \sum_{i \in R_j} (y_i - \hat{y}_{R_j})^2,$$
It is computationally infeasible to consider every possible partition of the feature space. For this reason, we take top-down (start top of the tree), greedy (only consider at particular step) approach which known as recursive binary splitting. 
#### Tree Pruning
We can limit the tree from growing but it might be sub-optimal since we might exclude a good split. For that reason, we grow a very large tree and prune it back to get a subtree. We can use cost complexity pruning rather than considering every possible subtree, we minimize:
$$\sum_{m=1}^{|T|} \sum_{i : x_i \in R_m} (y_i - \hat{y}_{R_m})^2 + \alpha |T|$$
$|T|$ indicates number of terminal node. Tuning parameter $\alpha$ controls trade-off between subtree's complexity and the fit. We can then select the tuning parameter by using validation set or cross-validation.
Algorithm:
- Use recursive binary splitting to grow a large tree on training data, stop when each terminal node has fewer item than threshold
- Apply cost complexity pruning to obtain sequence of best subtree as a function of $\alpha$ 
- Use K-fold cross-validation to choose $\alpha$. Divide training observations into *K* folds, for each fold:
	- repeat step 1 and 2 on the other than *k*-th fold.  
	- Evaluate MSE as a function of $\alpha$
	- Average results for each value of each value of $\alpha$ and choose $\alpha$ that minimize the average error
- Return subtree from step 2 that corresponds to chosen value of $\alpha$
### Classification Trees
In classification tree, we use mode instead of mean or median. We use classification error rate which is simply the fraction of training observations in that region that do not belong to the mode of the region. However, it turns out classification error is not sufficiently sensitive for tree-growing.
Gini index can be used in this case:
$$G = \sum_{k=1}^{K} \hat{p}_{mk}(1 - \hat{p}_{mk}),$$
where $\hat{p}_{mk}$ represents proportion of training observations in the *m*th region that are from the *k*th class. Gini index takes on small value if the proportion close to zero or one, for this reason Gini index is referred as measure of node purity. 
We can also use entropy:
$$D = -\sum_{k=1}^{K} \hat{p}_{mk} \log \hat{p}_{mk}.$$
Gini index and entropy are quite similar numerically. When building classification tree, either Gini index or entropy are used to evaluate quality of a split. However, classification error rate is preferable if prediction accuracy of final pruned tree is the goal.
### Tree vs Linear Model
If the relationship between features and response is linear, then linear model will work well. If the relationship is highly non-linear and complex, decision tree might be used instead. 
### Pros and Cons
Pros:
- Easy to explain
- Mirror human decision-making
- Can be displayed graphically
- Can handle qualitative predictors without need to create dummy variables

Cons:
- Have lower predictive accuracy than other models
- Tree can be very non-robust, a small change in data can cause large change in final estimated tree.
## Bagging, Random Forests, Boosting, Bayesian Additive Regression Trees
Ensemble method is an approach combining many simple model to obtain a single powerful model. The simple model are usually referred as weak learners. 
### Bagging
Bagging -> Bootstrap aggregation is used to reduce variance of statistical learning method. 
Averaging number of observations reduces the variance.
$$\hat{f}_{\text{avg}}(x) = \frac{1}{B} \sum_{b=1}^{B} \hat{f}^{b}(x).$$
Instead of getting new data, we bootstrap it from single training set.
$$\hat{f}_{\text{bag}}(x) = \frac{1}{B} \sum_{b=1}^{B} \hat{f}^{*b}(x).$$
We can apply bagging to regression trees where each tree are grown deep and are not pruned. Hence, each individual tree has high variance but low bias. Applying bagging will reduce the variance. Bagging also applicable to classification tree by simply output the mode. 
#### Out-of-Bag Error Estimation
We use out-of-bag samples, the remaining of the observations not used to fit a given bagged tree. 
#### Variable Importance Measures
Bagging improves prediction accuracy at the expense of interpretability. We can obtain overall summary of the importance of each predictor using RSS (for regression) and Gini index (for classification). 
### Random Forests
When building each tree, random sample of *m* predictors is chosen as split candidates from full set of *p* predictors. The split is allowed to use only one of those *m* predictors. This to avoid each tree being highly correlated in presence of the strong predictor. 
While bagging in each step consider each of the feature, random forest only consider subset of the features in each step and each tree.
### Boosting
Each tree is grown using information from previously grown trees. Each tree is fit on a modified version of the original data set. Given current model, we fit decision tree to residuals from the model. 
Algorithm:
- Set $\hat{f}(x) = 0$ and $r_i = y_i$ for all *i* in training set
- For $b = 1, 2, \dots, B$; repeat:
	- Fit a tree $\hat{f}^b$ with d splits (d + 1) terminal nods to the training data $(X,r)$
	- Update $\hat{f}$ by adding shrunken version of the new tree: $$\hat{f}(x) \leftarrow \hat{f}(x) + \lambda \hat{f}^b(x)$$
	- Update residuals, $$r_i \leftarrow r_i - \lambda \hat{f}^b(x_i)$$
- Output the boosted model, $$\hat{f}(x) = \sum_{b=1}^{B} \lambda \hat{f}^{b}(x).$$
Tuning parameters:
- Number of trees *B*, if *B* is too large, boosting can overfit. We use CV to select *B*
- Shrinkage parameter $\lambda$ 
- Number of *d* splits in each tree, $d=1$ often works well, each tree is a stump.
### Bayesian Additive Regression Trees (BART)
Notations:
- *K* denote number of regression trees
- *B* denote number of iterations for which the BART algorithm will run
- $\hat{f}_k^b(x)$ represents prediction at *x* for the *k*th regression tree used in the *b*th iteration

Algorithm:
- Let $$\hat{f}_1^1(x) = \hat{f}_2^1(x) = \cdots = \hat{f}_K^1(x) = \frac{1}{nK} \sum_{i=1}^{n} y_i.$$
- Compute $$\hat{f}^1(x) = \sum_{k=1}^{K} \hat{f}_k^1(x) = \frac{1}{n} \sum_{i=1}^{n} y_i.$$
- For $b = 2, \dots, B$:
	- For $k = 1,2,\dots, K$:
		- For $i = 1,2, \dots, n$, compute current partial residual $$r_i = y_i - \sum_{k'<k} \hat{f}_{k'}^{b}(x_i) - \sum_{k'>k} \hat{f}_{k'}^{b-1}(x_i).$$
		- Fit a new tree, $\hat{f}_k^b(x)$ to $r_i$ by randomly perturbing the *k*th tree from previous iteration, $\hat{f}_k^{b-1}(x)$. Perturbations that improve the fit are favored. 
	- Compute $$\hat{f}^{b}(x) = \sum_{k=1}^{K} \hat{f}_k^{b}(x).$$
- Compute the mean after *L* burn-in samples, $$\hat{f}(x) = \frac{1}{B - L} \sum_{b = L+1}^{B} \hat{f}^{b}(x).$$
### Summary of Tree Ensemble Methods
- In bagging, trees are grown independently on random samples of observations. The trees then to be quite similar each other.
- In random forests, trees are grown independently on random samples of observations. However, in each split we only consider random subset of the features to reduces correlation between trees.
- In boosting, we only use original data. Tree are grown successively using slow learning approach, each new tree is fit to the signal that is left over from earlier trees.
- In BART, we grow tree successively. Each tree is perturbed in order to avoid local minima and achieve a more thorough exploration of model space. 