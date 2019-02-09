# Data-income-classificationproblem
Binary classification

Predicting if income will be >=50k using census data
go to binaryclassification.pdf
###############################################################################################
The prediction task is to determine whether a person makes over $50K a year.

I  used five different classification algorithms:

Decision Tree Classifier

Random Forest Classifier

Logistic classifier

SVM classifier

K Neighbors Classifier

I evaluated my predictions using different metrics:

Accuracy 

Precision 

Recall 

F1 

Area under precision recall 

###################### The data ###################################################################
This data was extracted from the 1994 Census bureau database .

Attributes: >50K, <=50K
age: 
work class: 
education: 
education-num: 
marital-status: 
occupation: 
relationship
race: 
sex: 
capital-gain: continuous
capital-loss: continuous
hours-per-week: continuous
native-country: 

################# Results ######################################################################
Accuracy Fraction of predictions our model got right 
The best was the classification tree.
Precision Proportion of those predicted positive, how many of them are actual positive. 
The best was the classification tree.
Recall Proportion of the actual positive that were predicted correctly
The best was the Kneighbors.
F1 Conveys the balance between the precision and the recall 
The best was the Kneighbors
Area under precision recall The precision-recall curve shows the tradeoff between precision and recall for different threshold. 
The best was the classification tree.

