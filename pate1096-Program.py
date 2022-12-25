# %%
# Importing the pre-requisites for the program

import pandas as pd
import sklearn.datasets as skd
from sklearn.datasets import __all__ as sklearn_datasets_list
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.model_selection import KFold, cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import ipywidgets as widgets
from IPython.display import display, clear_output
import importlib
import sys

# Setting options for the pandas library
pd.options.display.precision = 4
pd.options.display.max_columns = 9
pd.options.display.width = None

# Declaring Linear Regression Model in a variable
linear_regression = LinearRegression()

# Declaring a class to halt the process without killing the kernel
class ExitException(Exception): pass

# Setting options for the seaborn library
sns.set(font_scale = 2)
sns.set_style('whitegrid')

# Initializing the 1st step for the program globally
current_step = 1

# %%
# Exit and Continue buttons with exit program and increment step functions
def halt_program(arg = None):
    try:
    # Raising exception when user clicks the Exit button without killing the jupytor ipython kernel
        raise ExitException("You have exited the program successfully!")

    except ExitException as exception:
        # Provide the user with restart button to restart the program
        print(exception)
        
        restart_btn = widgets.Button(description = 'Restart?')
        restart_btn.on_click(restart_program)
        exit_btn = widgets.Button(description = 'Exit.')
        exit_btn.on_click(exit_program)

        display(restart_btn)
        display(exit_btn)

# Function to restart the program alongwith resetting the current step to 1st step (dataset selection) and clearing the output
def restart_program(arg = None):
    global current_step
    current_step = 1
    clear_output(wait = True)
    print("You have restarted the program!")
    user_interaction_module()
    
# Function to exit the program, which will result in completely unbind of the variables from the system and killing the kernel process
def exit_program(arg = None):
    global current_step
    current_step = 1
    clear_output(wait = True)
    sys.exit()

# Function to increment the step counter by 1 for code to execute the next step
def increment_step(arg):
    global current_step
    current_step = current_step + 1
    clear_output(wait = True)
    user_interaction_module()

# Function to display the user action button to execute the action of choice
def ask_user_buttons():
    halt_btn = widgets.Button(description = 'Exit?')
    halt_btn.on_click(halt_program)

    continue_btn = widgets.Button(description = 'Continue!')
    continue_btn.on_click(increment_step)
    
    display(halt_btn)
    display(continue_btn)


# Setting the active_option globally for the code, so that the functions to be executed knows which dataset is selected
def option_california_housing(arg = None):
    global active_option
    print('You chose the California Housing Dataset')
    active_option = 'option_california_housing'
    load_dataset()

def option_diabetes(arg):
    global active_option
    print('You chose the Diabetes Dataset')
    active_option = 'option_diabetes'
    load_dataset()

# Optional input from the sklearn.datasets dropdown
def option_other_dataset(arg = None):
    print(f'You chose the {arg["new"]} Dataset from the Dropdown')
    global active_option
    active_option = arg['new']
    load_dataset()

# Visualizing the dataset options using IPython widgets
def dataset_selection():
    print("Please select a dataset to run the program with.")

    select_california_btn = widgets.Button(description = 'Select California Housing Dataset') 
    select_california_btn.on_click(option_california_housing)

    select_diabetes_btn = widgets.Button(description = 'Select Diabetes Dataset') 
    select_diabetes_btn.on_click(option_diabetes)

    another_dataset_dropdown = widgets.Dropdown(description = 'Choose One', options = sklearn_datasets_list)
    another_dataset_dropdown.observe(option_other_dataset, names = 'value')
    
    display(select_california_btn)
    display(select_diabetes_btn)
    print("Select from the dropdown if you want to select another dataset. ::: IN PROGRESS (DO NOT USE)")
    display(another_dataset_dropdown)


# Loading the dataset based on the option selected, and additional else statement for dynamic import based on sklearn.datasets dropdown
def load_dataset():
    global dataset

    if active_option == 'option_california_housing':
        dataset = skd.fetch_california_housing()
    elif active_option == 'option_diabetes':
        dataset = skd.load_diabetes()
    else:
        dataset =  importlib.import_module(skd, active_option)

    print(dataset.DESCR)

    # Asking the user for input decision
    ask_user_buttons()


# Setting median values for the dataframe, and additional code to set MedianVal for other sklearn.datasets
def data_exploration():
    global dataset_df

    dataset_df = pd.DataFrame(dataset.data, columns = dataset.feature_names)

    if active_option == 'option_california_housing':
        dataset_df['MedHouseValue'] = pd.Series(dataset.target)
    elif active_option == 'option_diabetes':
        dataset_df['DiseaseProg'] = pd.Series(dataset.target)
    else:
        dataset_df['MedianVal'] = pd.Series(dataset.target)

    print('dataframe.head ::: ', dataset_df.head())
    print('dataframe.describe ::: ', dataset_df.describe())

    sample_df = dataset_df.sample(frac = 0.1, random_state = 17)

    if active_option == 'option_california_housing':
        for feature in dataset.feature_names:
            plt.figure(figsize = (16, 9))
            sns.scatterplot(data = sample_df, x = feature, y = 'MedHouseValue', hue = 'MedHouseValue', palette = 'cool', legend = False)
    elif active_option == 'option_diabetes': 
        for feature in dataset.feature_names:
            plt.figure(figsize = (16, 9))
            sns.scatterplot(data = sample_df, x = feature, y = 'DiseaseProg', hue = 'DiseaseProg', palette = 'hot', legend = False)
    else:
        for feature in dataset.feature_names:
            plt.figure(figsize = (16, 9))
            sns.scatterplot(data = sample_df, x = feature, y = 'MedianVal', hue = 'MedianVal', palette = 'hot', legend = False)
            
    # Asking the user for input decision
    ask_user_buttons()


# Splitting the datasets into training and testing datasets, also declaring global variables for the code
def data_splitting():
    global X_train, X_test, y_train, y_test 
    X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, random_state = 11)
    X_train.shape
    X_test.shape

    # Asking the user for input decision
    ask_user_buttons()


# Training the model
def training_model():
    linear_regression.fit(X = X_train, y = y_train)
    LinearRegression(copy_X = True, fit_intercept = True, n_jobs = None, normalize = False)

    for i, name in enumerate(dataset.feature_names):
        print(f'{name:>10}: {linear_regression.coef_[i]}')

    print('LinearRegression Intercept ::: ', linear_regression.intercept_)

    # Asking the user for input decision
    ask_user_buttons()


# Testing the model
def testing_model():
    global predicted, expected
    predicted = linear_regression.predict(X_test)
    expected = y_test
    expected[:5]

    # Asking the user for input decision
    ask_user_buttons()


# Visualizing the Expected vs Predicted values from the selected dataset
def expected_vs_predicted_viz():
    df = pd.DataFrame()
    df['Expected'] = pd.Series(expected)
    df['Predicted'] = pd.Series(predicted)

    figure = plt.figure(figsize = (9, 9))
    axes = sns.scatterplot(data = df, x = 'Expected', y = 'Predicted', hue = 'Predicted', palette = 'cool', legend = False)

    start = min(expected.min(), predicted.min())
    end = max(expected.max(), predicted.max())
    axes.set_xlim(start, end)
    axes.set_ylim(start, end)

    line = plt.plot([start, end], [start, end], 'k--')

    # Asking the user for input decision
    ask_user_buttons()


# Model Metrics/Statistics based on the tests conducted
def regression_model_metrics():
    metrics.r2_score(expected, predicted)
    metrics.mean_squared_error(expected, predicted)

    # Asking the user for input decision
    ask_user_buttons()


# Comparing metrics of various models to check best model performance
def model_comparisions():
    estimators = {
    'LinearRegression': linear_regression,
    'ElasticNet': ElasticNet(),
    'Lasso': Lasso(),
    'Ridge': Ridge()
    }

    for estimator_name, estimator_object in estimators.items():
        kfold = KFold(n_splits = 10, random_state = 11, shuffle = True)
        scores = cross_val_score(estimator = estimator_object,
            X = dataset.data, y = dataset.target, cv=kfold,
            scoring = 'r2')
        print(f'{estimator_name:>16}: ' +
            f'mean of r2 scores = {scores.mean():.3f}')
    
    # Asking the user for input decision
    halt_program()


# Case based conditional module to execute following function stack based on the current_step's value
def user_interaction_module():
    if current_step == 1:
        print("Step 1: Load the dataset")
        dataset_selection()
    elif current_step == 2:
        print("Step 2: Explore the data")
        data_exploration()
    elif current_step == 3:
        print("Step 3: Split the data for training and testing")
        data_splitting()
    elif current_step == 4:
        print("Step 4: Train the data model")
        training_model()
    elif current_step == 5:
        print("Step 5: Test the data model")
        testing_model()
    elif current_step == 6:
        print("Step 6: Visualize the expected vs. predicted")
        expected_vs_predicted_viz()
    elif current_step == 7:
        print("Step 7: Create the regression model metrics")
        regression_model_metrics()
    elif current_step == 8:
        print("Step 8: Comparing mean of r2 scores of various models")
        model_comparisions()

# %%
# Initialize the Program
user_interaction_module()


