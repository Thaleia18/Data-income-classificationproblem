import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#### DATA DESCRIPTION
###
#Data income, classsification problem The original data has attributes:
#income (>50K, <=50K), age( continuous), workclass:( Private, Self-emp-not-inc...),
#fnlwgt: continuous, education (Bachelors,...), education-num (continuous), marital-status (Married-civ-spouse...),
#occupation (Tech-support,..), relationship (Wife,...), race (White,...), sex (Female, Male),
#capital-gain (continuous) capital-loss (continuous), hours-per-week (continuous) and native-country.

#I removed attributes that contain just minor categories. I kept attributes that have larrge categories,
#for example for race white and black are large categories and for native-country United States is the main caegory.

#So my final attributes are: income, age, education-num, marital-status, sex, capital-gain, capital-loss, 
#hours per week, native country. Here I cleaned the data set everything so it has just numerical variables.

#####DATA FROM
#####
#######https://www.kaggle.com/uciml/adult-census-income

adult_file_path = '../input/adult.csv'
adult_income_data = pd.read_csv(adult_file_path)
adult_income_data.columns =adult_income_data.columns.str.strip().str.lower().str.replace('.', '_')
adult_income_data.describe()

######SOME EXPLORATORY GRAPHS
####
####
sns.stripplot(x='sex', y='hours_per_week', data=adult_income_data,hue='income',marker='X')
sns.boxplot(x='hours_per_week',y='marital_status',data=adult_income_data,palette='rainbow',hue='income')
sns.boxplot(x='hours_per_week',y='education',data=adult_income_data,palette='rainbow',hue='income')
sns.boxplot(x='hours_per_week',y='race',data=adult_income_data,palette='rainbow',hue='income')

sns.stripplot(x='capital_gain', y='sex', data=adult_income_data,hue='income')
#plt.gca().set_xscale('log')
plt.xlim(0, 6000)

sns.stripplot(x='age', y='sex', data=adult_income_data,hue='income')


#####
###TRANSOFRMING TO DUMMY VARIABLES
######
money = {'<=50K': 0,">50K": 1} 
ale = {'Female': 0,"Male": 1} 
adult_income_data.income = [money[item] for item in adult_income_data.income] 
adult_income_data.sex = [ale[item] for item in adult_income_data.sex] 
white=[]
black=[]
native_american=[]
single=[]
married=[]
separated=[]
divorced=[]
widowed=[]
highdegree=[]
for i in range(len(adult_income_data.race)):
    white.append(1) if adult_income_data.race[i]=="White" else white.append(0)
    black.append(1) if adult_income_data.race[i]=="Black" else black.append(0)
    native_american.append(1) if adult_income_data.native_country[i]=="United-States" else native_american.append(0)
    single.append(1) if  adult_income_data.marital_status[i]=='Never-married' else single.append(0)
    married.append(1) if  adult_income_data.marital_status[i]=='Married-civ-spouse' else married.append(0)
    separated.append(1) if  adult_income_data.marital_status[i]=='Separated' else separated.append(0)
    divorced.append(1) if  adult_income_data.marital_status[i]=='Divorced' else divorced.append(0)
    widowed.append(1) if  adult_income_data.marital_status[i]=='Widowed' else widowed.append(0)
    highdegree.append(1) if adult_income_data.education[i]=='Masters' else (highdegree.append(1) if adult_income_data.education[i]=='Doctorate' else highdegree.append(0))
adult_income_data['white'] = white
adult_income_data['black'] = black
adult_income_data['born_usa'] = native_american
adult_income_data['single'] =single
adult_income_data['married'] =married
adult_income_data['separated'] =separated
adult_income_data['divorced'] =divorced
adult_income_data['widowed'] =widowed
adult_income_data['highdegree'] =highdegree
adult_features = ['age','sex','education_num','hours_per_week','born_usa','white','black','single','married','separated','divorced','widowed','highdegree','capital_gain','capital_loss','income']
#y=adult_income_data.income
#print(adult_income_data.capital-gain)
data2 = adult_income_data[adult_features]
data2.head()

####CORRELATIONS
###
######Here I explored the correlation between features and income.

colormap = plt.cm.magma
plt.figure(figsize=(16,16))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(data2.corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)


######The race and country of birth had the smallest correlation with the income so I remove them form my data.

