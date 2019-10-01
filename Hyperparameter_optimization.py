
'''
The below program demonstrates data preprocessing, one hot encoding, label encoding and 
one hot encoding of top n frequent categories 
and hyperparameter optimization for training a Multilayer Perceptron using keras wrapper for sckit learn

'''
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, Flatten, LeakyReLU, BatchNormalization, Dropout
from keras.activations import relu, sigmoid


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13]
y = dataset.iloc[:, 13]


# one hot encoding
geography =pd.get_dummies(X["Geography"] ,drop_first=True)
gender =pd.get_dummies(X['Gender'] ,drop_first=True)
# KDD Orange Cap compettion- top n categories based of frequencies 
#Checking frequency of unique category for geography variable
#lets make a list of only most frequent 3 categories for dummy variables.
# This step is helpful when categorical variable has many categories
top_3 = [x for x in dataset.Geography.value_counts().sort_values(ascending = False).head().index]
# Making binary variables 
for label in top_3:
    dataset[label] = np.where(dataset['Geography']== label , 1, 0)

# concatenating the newly created variable and original geography variable in a new dataframe
dataset[['Geography']+ top_3].head(40)



labelencoder_X_1 = LabelEncoder()
X['Geography'] = labelencoder_X_1.fit_transform(X['Geography'])
labelencoder_X_2 = LabelEncoder()
X.iloc[:,2] = labelencoder_X_2.fit_transform(X.iloc[:, 2])

# Keeping cateorical features in a df
df_cat_feat= X[['Gender', 'Geography']]
# Making a list of column headers 
cat_feat_list = list(cat_feat.columns)

''' 
Creating a dictionary with keys as list of unique values of columns of df
Assigning names to list from iterator to be used in dataframe function when converting one
hot encoded numpy array to DF for concatenating with original dataframe
'''
dict_of_list = {}
for var in df_cat_feat:
    dict_of_list["list_" + str(var)]  = X[var].unique().tolist()
 # Creating individual list from dict keys 
for key, value in dict_of_list.items():
    exec(key + '= value')
# List has to be sorted as label encoding is done in alphabatical order
list_Geography.sort()
one_hot_enc_var_list = list_Gender + list_Geography


# Encoding 1 categorical feature at a time using index
# indicating the 2nd column
onehotencoder = OneHotEncoder(categorical_features = [1])
       
# Encoding all categorical features
onehotencoder = OneHotEncoder()
# Converting the encoded variables to numpy ndarray
M = onehotencoder.fit_transform(df_cat_feat).toarray()
# Converting it into dataframe
df = pd.DataFrame(M, columns = one_hot_enc_var_list )
# dropping one category from each new dummy variables 
df = df.drop(['Female', 'France'], axis =1)

## Concatenate the Data Frames 
# 1st way by using results of pd.get_dummies
X= pd.concat([X,geography,gender],axis=1)
# Results of one hot encoding
X = pd.concat([X, df], axis = 1)

## Drop Unnecessary columns
X= X .drop(['Geography','Gender'],axis=1)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Hyperparameter Optimization

def create_model(layers, activation):
    model = Sequential()
    for i, nodes in enumerate(layers):
        if i == 0:
            model.add(Dense(nodes, input_dim=X_train.shape[1]))
            model.add(Activation(activation))
            model.add(Dropout(0.3))
        else:
            model.add(Dense(nodes))
            model.add(Activation(activation))
            model.add(Dropout(0.3))
    # Final layer 
    model.add(Dense(units=1, kernel_initializer='glorot_uniform',
                    activation='sigmoid'))  # Note: no activation beyond this point

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


model = KerasClassifier(build_fn=create_model, verbose=0)

layers = [[20], [40, 20], [45, 30, 15]]
activations = ['sigmoid', 'relu']
param_grid = dict(layers=layers, activation=activations, batch_size=[128, 256], epochs=[30])
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)

grid_result = grid.fit(X_train, y_train)

[grid_result.best_score_, grid_result.best_params_]


pred_y = grid.predict(X_test)
y_pred = (pred_y > 0.5)

# Performance check 
# Confusion matrix, Accuracy, precision, recall and f1
cm = confusion_matrix(y_test, y_pred)
print(cm)

score=accuracy_score(y_test,y_pred)
print(score)

class_rep = classification_report(y_test, y_pred)
print(class_rep)







