[<img src="https://img.shields.io/badge/ScikitLearn-ML-important.svg?logo=scikitlearn">](<LINK>)
[<img src="https://img.shields.io/badge/HuggingFace-BERT-important.svg?logo=huggingface">](<LINK>)

<h1 align="center" style="font-size:60px;">Sentimental</h1>

It is a comparative analysis of classification performance between BERT and traditional ML models for sentiment analysis on clothing reviews. In addition to that, a clustering of the various types of reviews are also performed.

# Technologies
<img src="https://scikit-learn.org/stable/_static/scikit-learn-logo-small.png" width="50">
<img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" width="50">

# Prerequisites
* Download the [clothing reviews](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews) .csv file
* Download the .ipynb file from the repository
* Open [Google Colab](https://colab.research.google.com/) , upload the .ipynb file and change runtime type to GPU
* Upload the data file to the 'Files' section at the left side of opened notebook

# Aims
The notebook already has the outputs. You can also rerun each cell to check the outputs for yourself but the Inferences described below are based on the current notebook outputs. The notebook covers the following objectives:

#### 1. Clustering
* [K-means clustering](https://en.wikipedia.org/wiki/K-means_clustering) with K=5 is applied on the [TFIDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) feature extracted from hotel reviews
* Example reviews as well as top 5 tokens from each cluster are displayed to see the distribution of positive, neutral and negative reviews. A confusion matrix is also visualized.

**Inference:**
Based on the sampled documents, cluster 0 reviews seem to be predominantly negative, clusters 1 and 2 are comprised mostly of positive reviews, whereas clusters 3 and 4 are a mix of positive and negative reviews, hence neutral in nature. From the top 5 tokens, it can be inferred that clusters 0 and 3 are about clothing sizes, cluster 1 talks more about dress colors, whereas clusters 2 and 4 are about reviews related to purchases.

From the confusion matrix, it is evident that in each cluster, the no. of Positive labels is the highest followed by the no. of Neutral and Negative labels. Although none of the clusters are able to pick up a single label, overall all the clusters are predominantly Positive.

#### 2. ML Classification (Scikit-Learn)
* [Dummy classifier](https://www.geeksforgeeks.org/ml-dummy-classifiers-using-sklearn/), [Logistic Regression](https://www.geeksforgeeks.org/understanding-logistic-regression/) and [SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) are used for text classification.
* The [F1 score](https://en.wikipedia.org/wiki/F-score) for each class in case of the best performing classifier is visualized.
* The performance of [AdaBoost](https://en.wikipedia.org/wiki/AdaBoost) is then compared with the baseline classifiers.
* Hyperparameter tuning is applied for Logistic Regression with TFIDF.

**Inference:**
Logistic Regression using tfidf vectors is the best-performing classifier according to F1 score. Although Logistic Regression with TFIDF vectors does obtain a higher accuracy on the validation set, the dataset is imbalanced with the no. of positive labels significantly higher than the negative and neutral labels, which makes accuracy a bad metric for judging model fit on the validation data. In this scenario, F1 score is a much more reliable metric. The model gives a medium F1 score of 51.953% which is due to the fact that the model has a higher precision score than a recall score, indicating that the model on average is capable of classifying true positives correctly than finding the true positives.

On the other hand, the rest of the classifiers have a fairly low F1 score due to low precision and recall values indicating a poor model fit and hence not desirable for the chosen dataset and the classification task. If the dummy classifiers are chosen as the baselines, the remaining 3 classifiers perform similar or even better than the baselines (by F1 scores). The ‘most frequent’ baseline strategy has a tendency to return the most frequent class label, which is a terrible strategy for fitting an unbalanced dataset in this case. The ‘stratified’ strategy does improve upon the precision of the ‘most frequent’ strategy but at the expense of the classifier accuracy. For the other three classifiers, it is observed that TFIDF input reports better performance than One-hot encoded vectors as it is able to capture both the frequency and relevance information of words in every document.

The usage of an ensemble classifier such as AdaBoost will reduce the variance of a single model and improve classification accuracy by iteratively focusing more on the misclassified instances in the previous step. This seems to hold true as AdaBoost improves upon the models that use one-hot vectors and the dummy classifiers. It, however, is unable to surpass Logistic regression with TFIDF except in the case of recall which is essential for the task at hand i.e. sentiment analysis, where we would want our model to find a large proportion of true positives effectively for better product analysis (in this case, clothes) rather than correctly classify the true positives.

#### 3. BERT Classification (HuggingFace)
* Logistic Regression is applied on the [context vectors](https://link.springer.com/chapter/10.1007/10719871_14) obtained from [BERT](https://huggingface.co/blog/bert-101)
* An [end-to-end classifier](https://www.analyticsvidhya.com/blog/2021/12/googles-bert/) is also utilized for text classification.
* Different hyperparameter settings are also tried out for the end-to-end classifier.
* The fine-tuned end-to-end HuggingFace neural network is evaluated on the test data and a confusion matrix is visualized since this model is the best performing classifier out of all the classifiers.

**Inference:**
The end-to-end neural classifier performed much better than the context vectors on Logistic Regression because a neural network is able to learn useful non-linear feature patterns in the data over time compared to a traditional ML model like Logistic Regression which needs to be provided with a specific feature representation (in this case, context vectors) and it is able to learn that representation only. This ability of neural networks to capture non-linear patterns in a task like sentiment analysis from text makes them superior than traditional ML models.

From the confusion matrix, it is clear that the majority of labels is in the Positive class indicating a class imbalance in the dataset. It is also evident that the classifier is able to classify the majority of labels into their correct classes. If we consider the Positive label as ‘positive’ and the Neutral and Negative labels as ‘negative’, it can be observed from the matrix that the classifier is making more false positives than false negatives. This can be due to the subjective nature when it comes to product ratings. The customer might rate a product neutrally but give a fairly positive review about it, which is evident from the large number of Neutral reviews (70) predicted as Positive. False negatives, on the other hand, might be due to the customer actually liking the product but confusing a rating of 1 as an indicator of a great product, like for instance, 74 and 13 Positive reviews have been misclassified as Neutral and Negative respectively.

In terms of accuracy on the test data, the final performance is appreciable for sentiment analysis on clothing products since the bulk of the reviews have been classified correctly. However, since the F1 score is moderately high, false positives and false negatives will have an effect when this classifier is deployed. For example, the 70 Neutral and 18 Negative products misclassified as Positive, can be suggested to customers as high-rated products when in reality they are mediocre. Similarly, the 74 and 13 Positive products misclassified as Neutral and Negative will never be recommended to the customers leading to a bad user experience as the customers miss out on these supposed great products.
