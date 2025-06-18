# DataScience2025 Project
## classifying thyroid cancer whether it is Malignant / Benign

### Goal
by using random forest classifier, find the method to increase the predictibility of the model

### defining preprocessor 
we used XGBoost classifier to find the feature importances.
after that we except the features that have low importances.
we recognized that even the feature importances is low, that feature may be essential to classify in the leaf level,
and we knew that ideal way is to try various feature combination.
but we have very strict time limitation so we thought reducing features based on feature importances is the most efficient way in our situation

### K-fold stratified cross validation
to estimate the performance differneces of the model between model that adds derived features and base model we used 10-fold stratified cross validation

#### problem
we found a critical problem in the beginning of the project
the problem was data imbalances in the target classes
we try to solve this imbalances by using smote technique which is artificially make data of the small side by interpolation

### performance comparison: base model vs model with derived features
after the cross validation we found that the average performance of the model with derived features is 1.9%p higher than base model

### paired t-test
to validate whether the result is statistically meaningful or not, we did paried t-test and we got statistically meaningful result that the p-value is under 0.05

### hyper parameter tuning
to find the best parameters of our model, we did hyper parameter tuning by GridSearchCV

### performance comparison: model with derived features vs model with derived features + tuning
we compared the performance between model with features and the model with features plus tuning
to compare the performance, we also did 10-fold stratified cross validation
we found that the performance of the accuracy 0.6%p increased after the hyper parameter tuning

### performance comparisoin: base model vs model with derived features + tuning
finally we compared the performance between base model and model with derived features plus tuning with the whole dataset
we found that the performance increased 2.33%p after feature engineering and hyper parameter tuning

### chi square hypothesis validation
we did hypothesis validation with the derived features
actually important thing in data analysis is that all we can do is to suggest the "possibility" not a "casuality"
by chi square hypothesis validation we got the chi square statistics and the p-value of the hypothesis

### reminiscence
there were some problems while proceeding the project
- first: data imbalances
  - i already mentioned this problem on the above side, we tried to solved this problem by using smote technique. but smote only solve the data imbalance problem in training set level not a test set level
  - so the left assignment is to think about the imbalance problem in the test level
- second: lack of knowledge in the medical domain
   - in fact, nobody in our team had a knowledge in cancer because all the teammates major in computer science
   - so in the interpretation stage, all we can do is just suggest the possibility which features affect the classifying the malignant cancer
