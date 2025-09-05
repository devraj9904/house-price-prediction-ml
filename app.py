import pandas as pd         #used to read data (like CSV/Excel files), clean it, and analyze it easily.
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression   #Imports the Linear Regression model from scikit-learn
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
                                                    



data = pd.read_csv("train.csv")                     #load the dataset in the variable

print(data.head())                                  #display the first 5 rows of the dataset

print(data.info())                                  #shows information about the columns like name, type

print(data.describe())                              #shows basic statistics like min, avg

data.fillna(data.median(numeric_only=True), inplace=True) 
'''
this method calculates the median of all numerical cols. Along with that it fills the missing values with 
the median.
'''

for col in data.select_dtypes(include='object').columns:
    data[col].fillna(data[col].mode()[0], inplace=True)
'''
for col in data.select_dtypes(include='object').columns--- this selects all the text columns.
 data[col].fillna(data[col].mode()[0], inplace=True)--- this replaces the missing values with the value
 that has the most occurence in that column(mode).
'''


data = pd.get_dummies(data, drop_first=True)
'''
pd.get_dummies()--- method converts the categorical col names from text to 0/1 
drop_first--- 
'''

y = data["SalePrice"]          # Target we want to predict
X = data.drop("SalePrice", axis=1)  # All other columns are input features


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
'''
train_test_split--- function from scikit-learn
X, y → features and target we prepared
test_size=0.2 → 20% of data goes to testing, 80% for training
random_state=42 → ensures the split is the same every time you run the code
Output:
X_train → features for training
X_test → features for testing
y_train → target for training
y_test → target for testing
'''


model = LinearRegression()      #create an instance of the model

model.fit(X_train, y_train)    
#Trains the model using the training data. 
#The model learns patterns between features (X_train) and target (y_train)

y_pred = model.predict(X_test)
'''
predict() → uses the trained model to predict prices for the test set
y_pred → predicted house prices
'''


mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error:", mae)

'''
mean_absolute_error → average difference between predicted and actual prices
smaller the MAE, the better
'''


################## RandomForest Model #################

rf_model= RandomForestRegressor(
    n_estimators=500,     # Step 1: more trees
    max_depth=20,         # Step 2: limit depth
    min_samples_leaf=2,   # Step 3: avoid overfitting
    max_features="sqrt",  # Step 4: feature selection
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train,y_train)  # RandomForest model training on training data

rf_pred=rf_model.predict((X_test))  # Prediction on test data

rf_mae = mean_absolute_error(y_test, rf_pred)
print("Random Forest MAE:", rf_mae)     # Calculate the error


################ RandomForest with Grid Search ################

rf=RandomForestRegressor(random_state=42)


# Parameter grid
param_grid= {
    "n_estimators": [100,200,300],      # number of trees
    "max_depth": [None,10,20],          # max depth of tree
    "min_samples_split": [2,5,10],      # min samples needed to split a node
    "min_samples_leaf": [1,2,4]         # min samples per leaf
}


# Setup + GridSearch
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    scoring="neg_mean_absolute_error",  # we want to minimize MAE
    cv=3,                               # 3-fold cross-validation
    n_jobs=-1,                          # use all CPU cores for speed
    verbose=2                           # show progress
)


grid_search.fit(X_train,y_train)        # Run grid search on training data
best_rf = grid_search.best_estimator_   # 5. Get best model

print("Best Parameters:", grid_search.best_params_)


# 6. Evaluate on test set
y_pred = best_rf.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print("Best Random Forest MAE:", mae)


################ Gradient Boosting Regressor Model ###############

gbr=GradientBoostingRegressor(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.1,
    random_state=42
)

gbr.fit(X_train,y_train)        # train the model

y_pred=gbr.predict(X_test)         # Predict on test


mae = mean_absolute_error(y_test, y_pred)
print("Gradient Boosting MAE:", mae)



################ Gradient Boosting Regressor Model with Grid Search ###############

gbr=GradientBoostingRegressor(random_state=42)

param_grid={
    "n_estimators": [100,200,300,400,500],
    "learning_rate": [0.05, 0.1, 0.2],
    "max_depth": [3, 4, 5],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

grid_search=GridSearchCV(
    estimator=gbr,
    param_grid=param_grid,
    scoring="neg_mean_absolute_error",
    cv=3,
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train,y_train)        # train the model


best_gbr = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)     # Get the best model and MAE


y_pred= best_gbr.predict(X_test)
mae= mean_absolute_error(y_test,y_pred)
print("Best Gradient Boosting MAE:", mae)



