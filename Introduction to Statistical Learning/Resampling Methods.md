Model Assessment -> Evaluating model's performance
Model Selection -> Selecting proper level of flexibility
## Cross Validation
### Validation Set Approach 
Randomly divide available set into two parts, training set and validation set or hold-out set. 
Drawbacks:
- Which observations are included in training set affects the performance
- Model would not be as good since we have lesser training data
### Leave One Out Cross Validation (LOOCV)
Only use a single random observation for the validation set and the remaining *n-1* observations will be the training set. We use this *n* times until all observation becomes validation set. We then simply average the MSE to get estimate for the test MSE
Advantages:
- Less bias since trained on *n-1* observations. It won't overestimate test error rate as much as validation set approach does. 
- Performing LOOCV multiple times will always yield the same results

Disadvantages:
- Time and memory consuming since need to train *n* times.

### k-Fold Cross Validation
Randomly divide set of observations into *k* groups of folds with approximately equal size. Each of the k group will be the validation set, and the remaining k-1 groups will be the training set, we will do the training and evaluating for k times. We simply average the MSE to get approximate of test MSE
Advantages:
- Not as expensive as LOOCV which requires training n times
### Bias-Variance Trade-Off for k-Fold Cross-Validation
K-Fold CV tend to gives more accurate estimates of test error rate compared to LOOCV. Validation set approach can overestimates test error rate since it only uses half of the entire dataset to train. LOOCV will give unbiased estimates of test error since training data is *n-1* which almost equals to *n*. On the other hand k-fold CV will lead to an intermediate level of bias. 
Test error estimate resulting from LOOCV tends to have higher variance than test error estimate from k-fold CV. So, usually we will do k-fold cross validation with k being 5 or 10. 
### Cross Validation on Classification Problems
Instead of MSE, we just count the mismatch between prediction and actual dataset. 
## Bootstrap 
We repeatedly sampling observations from original data set. Sampling is performed with replacement, means any observation can occur more than once in the bootstrap data set. 

