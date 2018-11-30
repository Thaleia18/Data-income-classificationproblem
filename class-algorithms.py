

####First decision tree
###
#####Here I will start with a decision tree classification, first I use K-fold cross validation to find the optimum depth:

######
######DECISION TREE ##################################################################################################
#####
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
adult_features = ['age','sex','education_num','hours_per_week','single','married','separated','divorced','widowed','highdegree','capital_gain','capital_loss','income']
X = adult_income_data[adult_features]

cv = KFold(n_splits=10)            # Number of Cross Validation folds
accuracies = list()
max_features = len(list(X))
depth_range = range(1, max_features)

# Testing max_depths from 1 to max features
for depth in depth_range:
    fold_accuracy = []
    tree_model = DecisionTreeClassifier(max_depth = depth)

    for train_fold, valid_fold in cv.split(X):
        f_train = X.loc[train_fold] # Extract train data with cv indices
        f_valid = X.loc[valid_fold] # Extract valid data with cv indices
        model = tree_model.fit(X = f_train.drop(['income'], axis=1), 
                               y = f_train["income"]) # We fit the model with the fold train data
        valid_acc = model.score(X = f_valid.drop(['income'], axis=1), 
                                y = f_valid["income"])# We calculate accuracy with the fold validation data
        fold_accuracy.append(valid_acc)

    avg = sum(fold_accuracy)/len(fold_accuracy)
    accuracies.append(avg)
    #print("Accuracy per fold: ", fold_accuracy, "\n")
    #print("Average accuracy: ", avg)
    #print("\n")
    
# To show the accuracy foe each depth
df = pd.DataFrame({"Max Depth": depth_range, "Average Accuracy": accuracies})
df = df[["Max Depth", "Average Accuracy"]]
print(df.to_string(index=False))


###
#####The best classification tree We will use depth=7 to create the best classification tree
### VISUALIZATION

from IPython.display import Image as PImage
from subprocess import check_call
from PIL import Image, ImageDraw, ImageFont
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, average_precision_score, precision_score, recall_score

adult_features = ['age','sex','education_num','hours_per_week','single','married','separated','divorced','widowed','highdegree','capital_gain','capital_loss','income']
X = adult_income_data[adult_features]
y = X['income']
X2 = X.drop(['income'], axis=1).values 
train_X, val_X, train_y, val_y = train_test_split(X2, y, random_state = 0)

# Create Decision Tree with max_depth = 9
decision_tree = DecisionTreeClassifier(max_depth = 7)
decision_tree.fit(train_X, train_y)

# Predicting results for validation dataset
#score=model.score( val_X, val_y)
#print(score)

# Export our trained model as a .dot file
with open("tree1.dot", 'w') as f:
     #f = Source(
         f=tree.export_graphviz(decision_tree,
                              out_file=f,
                              max_depth = 9,
                              impurity = True,
                              feature_names = ['age','sex','education_num','hours_per_week','single','married','separated','divorced','widowed','highdegree','capital_gain','capital_loss'],
                              class_names = ['<=50K', '>50K'],
                              rounded = True,
                              filled= True )#)
        
#Convert .dot to .png to allow display in web notebook
check_call(['dot','-Tpng','tree1.dot','-o','tree1.png'])

# Annotating chart with PIL
img = Image.open("tree1.png")
draw = ImageDraw.Draw(img)
font = ImageFont.truetype('/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf', 26)
draw.text((10, 0), # Drawing offset (position)
          '"Title <= Income', # Text to draw
          (0,0,255), # RGB desired color
          font=font) # ImageFont object with desired font
img.save('tree_income.png')
PImage("tree_income.png")

#######
###METRICS DECISION TREE
y_pred=decision_tree.predict(val_X)
print("Accuracy:", decision_tree.score(val_X, val_y))
print("Precision:", precision_score(val_y, y_pred))
print("Recall:", recall_score(val_y, y_pred))
print("F1:", f1_score(val_y, y_pred))
print("Area under precision Recall:", average_precision_score(val_y, y_pred))


##############3
#####3
#######3 RANDOM FOREST##################################################################################################

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import statsmodels.api as sm

adult_features = ['age','sex','education_num','hours_per_week','single','married','separated','divorced','widowed','highdegree','capital_gain','capital_loss','income']
X = adult_income_data[adult_features]
y = X['income']
X2 = X.drop(['income'], axis=1).values 
train_X, val_X, train_y, val_y = train_test_split(X2, y, random_state = 0)

forest = RandomForestClassifier(random_state=1)
forest.fit(train_X, train_y)
y_pred = forest.predict(val_X)
print("Accuracy:", forest.score(val_X, val_y))
print("Precision:", precision_score(val_y, y_pred))
print("Recall:", recall_score(val_y, y_pred))
print("F1:", f1_score(val_y, y_pred))
print("Area under precision Recall:", average_precision_score(val_y, y_pred))


##########33
#### LOGISTIC REGRESSION  ##################################################################################################
#########

With logistic regression

In [14]:
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

adult_features = ['age','sex','education_num','hours_per_week','single','married','separated','divorced','widowed','highdegree','capital_gain','capital_loss','income']
X = adult_income_data[adult_features]
y = X['income']
X2 = X.drop(['income'], axis=1).values 
train_X, val_X, train_y, val_y = train_test_split(X2, y, random_state = 0)

logreg = LogisticRegression()
logreg.fit(train_X, train_y)
y_pred = logreg.predict(val_X)
print("Accuracy:", logreg.score(val_X, val_y))
print("Precision:", precision_score(val_y, y_pred))
print("Recall:", recall_score(val_y, y_pred))
print("F1:", f1_score(val_y, y_pred))
print("Area under precision Recall:", average_precision_score(val_y, y_pred))


############3
#######3SVM classifier ##################################################################################################
###

from sklearn.svm import SVC 
from sklearn import metrics
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

adult_features = ['age','sex','education_num','hours_per_week','single','married','separated','divorced','widowed','highdegree','capital_gain','capital_loss','income']
X = adult_income_data[adult_features]
y = X['income']
X2 = X.drop(['income'], axis=1).values 
train_X, val_X, train_y, val_y = train_test_split(X2, y, random_state = 0)

 
svclassifier = SVC()  
svclassifier.fit(train_X, train_y)
y_pred = svclassifier.predict(val_X)
print("Accuracy:", svclassifier.score(val_X, val_y))
print("Precision:", precision_score(val_y, y_pred))
print("Recall:", recall_score(val_y, y_pred))
print("F1:", f1_score(val_y, y_pred))
print("Area under precision Recall:", average_precision_score(val_y, y_pred))

##################
########   KNEIGHBORS ##################################################################################################
############33
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

adult_features = ['age','sex','education_num','hours_per_week','single','married','separated','divorced','widowed','highdegree','capital_gain','capital_loss','income']
X = adult_income_data[adult_features]
y = X['income']
X2 = X.drop(['income'], axis=1).values 
train_X, val_X, train_y, val_y = train_test_split(X2, y, random_state = 0)

neiclassifier = KNeighborsClassifier(n_neighbors=5)  
neiclassifier.fit(train_X, train_y)
y_pred = neiclassifier.predict(val_X)
print("Accuracy:", neiclassifier.score(val_X, val_y))
print("Precision:", precision_score(val_y, y_pred))
print("Recall:", recall_score(val_y, y_pred))
print("F1:", f1_score(val_y, y_pred))
print("Area under precision Recall:", average_precision_score(val_y, y_pred))
