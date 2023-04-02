import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import FunctionTransformer

# Load the data
df = pd.read_csv('customer_data.csv')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df.drop(columns=['product_preference']), 
    df['product_preference'], 
    test_size=0.2, 
    random_state=42)

# Define the columns to be transformed
num_cols = ['age', 'income']
cat_cols = ['gender', 'education']

# Define the transformers for numeric and categorical data
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Define the feature engineering function
def feature_engineering(X):
    X['age_squared'] = X['age'] ** 2
    X['income_squared'] = X['income'] ** 2
    return X

# Define the column transformer to apply the transformers to each column
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols),
        ('feat', FunctionTransformer(feature_engineering), num_cols)
    ])

# Define the models to be evaluated
models = [
    {
        'name': 'RandomForestClassifier',
        'estimator': RandomForestClassifier(),
        'hyperparameters': {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5, 10]
        }
    },
    {
        'name': 'SVC',
        'estimator': SVC(),
        'hyperparameters': {
            'kernel': ['linear', 'rbf', 'poly'],
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto']
        }
    }
]

# Define the pipeline to combine the preprocessor and the models
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', None)
])

# Perform grid search over each model and print the best parameters and score
for model in models:
    print(f'Tuning hyperparameters for {model["name"]}')
    grid_search = GridSearchCV(
        model['estimator'], 
        param_grid=model['hyperparameters'], 
        cv=5, 
        n_jobs=-1, 
        verbose=1
    )
    pipeline.set_params(classifier=grid_search)
    pipeline.fit(X_train, y_train)
    print(f'Best parameters: {pipeline["classifier"].best_params_}')
    print(f'Test accuracy: {pipeline.score(X_test, y_test)}')
