import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# source nba/bin/activate

# Calculates the accuracy of true Y values vs predicted Y values
def accuracy(Y, Yhat): 
    return np.sum(Y==Yhat)/len(Y)

# Preprocesses the data 
def preprocess(data):
    # Uncomment the line below for the sensitivity analysis (dropping important feature)
    #data = data.drop(columns = ['WS', 'WS/48'])
    data['Rank'] = np.where(data['Rank'] > 1, 0, data['Rank'])
    data.dropna(inplace = True)
    player_names = data['Player']
    data_new = data.drop(columns=["Player"])
    Xdata = data_new.drop(columns='Rank')
    Ydata = data_new['Rank']
    Xtest = Xdata[:64]
    Ytest = Ydata[:64]
    Xtrain = Xdata[64:]
    Ytrain = Ydata[64:]
    Xtrain, Xval, Ytrain, Yval = train_test_split(Xtrain, Ytrain, test_size = 0.3, random_state = 5)
    return Xtrain, Ytrain, Xval, Yval, Xtest, Ytest, player_names

# Splits the test data by year from 2019 - 2023
def preprocess_test(Xtest, Ytest):
    Xtest_2023 = Xtest[:13]
    Ytest_2023 = Ytest[:13]
    Xtest_2022 = Xtest[13:25]
    Ytest_2022 = Ytest[13:25]
    Xtest_2021 = Xtest[25:40]
    Ytest_2021 = Ytest[25:40]
    Xtest_2020 = Xtest[40:52]
    Ytest_2020 = Ytest[40:52]
    Xtest_2019 = Xtest[52:]
    Ytest_2019 = Ytest[52:]
    return Xtest_2023, Ytest_2023, Xtest_2022, Ytest_2022, Xtest_2021, Ytest_2021, Xtest_2020, Ytest_2020, Xtest_2019, Ytest_2019

# Returns the player names corresponding to the years in the test set
def preprocess_names(names):
    players_2023 = names[:13]
    players_2022 = names[13:25]
    players_2022 = players_2022.reset_index(drop = True)
    players_2021 = names[25:40]
    players_2021 = players_2021.reset_index(drop = True)
    players_2020 = names[40:52]
    players_2020 = players_2020.reset_index(drop = True)
    players_2019 = names[52:64]
    players_2019 = players_2019.reset_index(drop = True)
    return players_2019, players_2020, players_2021, players_2022, players_2023

# Gets the index of the maximum value in the 2d array format that sklearn returns probabilites in
def find_max_idx(arr):
    max_idx = 0
    curr_max = float("-inf")
    for i in range(len(arr)):
        if arr[i][1] > curr_max:
            curr_max = arr[i][1]
            max_idx = i
    return max_idx

# Validation testing with different 'm' values for the Random Forest
def random_forest_graphs_1(Xtrain, Ytrain, Xval, Yval):
    random_forest_1 = RandomForestClassifier(n_estimators = 2, criterion = "entropy", random_state = 30)
    random_forest_2 = RandomForestClassifier(n_estimators = 10, criterion = "entropy", random_state = 30)
    random_forest_3 = RandomForestClassifier(n_estimators = 50, criterion = "entropy", random_state = 30)
    random_forest_4 = RandomForestClassifier(n_estimators = 250, criterion = "entropy", random_state = 30)
    random_forest_1.fit(Xtrain, Ytrain)
    random_forest_2.fit(Xtrain, Ytrain)
    random_forest_3.fit(Xtrain, Ytrain)
    random_forest_4.fit(Xtrain, Ytrain)
    random_forest_1_accuracy = accuracy(Yval, random_forest_1.predict(Xval))
    random_forest_2_accuracy = accuracy(Yval, random_forest_2.predict(Xval))
    random_forest_3_accuracy = accuracy(Yval, random_forest_3.predict(Xval))
    random_forest_4_accuracy = accuracy(Yval, random_forest_4.predict(Xval))
    random_forest_1_f1 = f1_score(Yval, random_forest_1.predict(Xval))
    random_forest_2_f1 = f1_score(Yval, random_forest_2.predict(Xval))
    random_forest_3_f1 = f1_score(Yval, random_forest_3.predict(Xval))
    random_forest_4_f1 = f1_score(Yval, random_forest_4.predict(Xval))
    m_scores = ('2','5','50','250')
    random_forest_results = {
        'accuracy' : (random_forest_1_accuracy, random_forest_2_accuracy, random_forest_3_accuracy, random_forest_4_accuracy),
        'f1' : (random_forest_1_f1, random_forest_2_f1, random_forest_3_f1, random_forest_4_f1)
    }
    x = np.arange(len(m_scores))
    width = 0.25  
    multiplier = 0
    fig, ax = plt.subplots(layout='constrained')
    for attribute, measurement in random_forest_results.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    ax.set_title('Accuracy and F1 score of Different m values')
    ax.set_xticks(x + width, m_scores)
    ax.legend(loc='center', ncols=4)
    ax.set_ylim(0, 1)
    plt.show()

# Validation testing with different 'k' values for Random Forest
def random_forest_graphs_2(Xtrain, Ytrain, Xval, Yval):
    random_forest_1 = RandomForestClassifier(n_estimators = 250, criterion = "entropy", random_state = 5, max_features = "sqrt")
    random_forest_2 = RandomForestClassifier(n_estimators = 250, criterion = "entropy", random_state = 5, max_features = "log2")
    random_forest_1.fit(Xtrain, Ytrain)
    random_forest_2.fit(Xtrain, Ytrain)
    random_forest_1_accuracy = accuracy(Yval, random_forest_1.predict(Xval))
    random_forest_2_accuracy = accuracy(Yval, random_forest_2.predict(Xval))
    random_forest_1_f1 = f1_score(Yval, random_forest_1.predict(Xval))
    random_forest_2_f1 = f1_score(Yval, random_forest_2.predict(Xval))
    k_values = ('sqrt(d)','log2(d)')
    random_forest_results = {
        'accuracy' : (random_forest_1_accuracy, random_forest_2_accuracy),
        'f1' : (random_forest_1_f1, random_forest_2_f1)
    }
    x = np.arange(len(k_values))
    width = 0.25  
    multiplier = 0
    fig, ax = plt.subplots(layout='constrained')
    for attribute, measurement in random_forest_results.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    ax.set_title('Accuracy and F1 score of Different k values')
    ax.set_xticks(x + width, k_values)
    ax.legend(loc='center', ncols=4)
    ax.set_ylim(0, 1)
    plt.show()

# Validation testing for different neural network model architectures
def nn_graph_1(Xtrain, Ytrain, Xval, Yval):
    mlp1 = MLPClassifier(hidden_layer_sizes = (128, 256), max_iter = 500, random_state = 5, batch_size = 8, alpha = 0.0001, solver = 'sgd')
    mlp2 = MLPClassifier(hidden_layer_sizes = (4,8,16,32,64,), max_iter = 500, random_state = 5, batch_size = 8, alpha = 0.0001, solver = 'sgd')
    mlp3 = MLPClassifier(hidden_layer_sizes = (16,32,64,128,), max_iter = 500, random_state = 5, batch_size = 8, alpha = 0.0001, solver = 'sgd')
    mlp1.fit(Xtrain, Ytrain)
    mlp2.fit(Xtrain, Ytrain)
    mlp3.fit(Xtrain, Ytrain)
    mlp1_accuracy = accuracy(Yval, mlp1.predict(Xval))
    mlp2_accuracy = accuracy(Yval, mlp2.predict(Xval))
    mlp3_accuracy = accuracy(Yval, mlp3.predict(Xval))
    mlp1_f1 = f1_score(Yval, mlp1.predict(Xval))
    mlp2_f1 = f1_score(Yval, mlp2.predict(Xval))
    mlp3_f1 = f1_score(Yval, mlp3.predict(Xval))
    architectures = ('shallow/wide','deep/skinny', 'middle')
    random_forest_results = {
        'accuracy' : (mlp1_accuracy, mlp2_accuracy, mlp3_accuracy),
        'f1' : (mlp1_f1, mlp2_f1, mlp3_f1)
    }
    x = np.arange(len(architectures))
    width = 0.25  
    multiplier = 0
    fig, ax = plt.subplots(layout='constrained')
    for attribute, measurement in random_forest_results.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    ax.set_title('Accuracy and F1 score of Different Network Architectures')
    ax.set_xticks(x + width, architectures)
    ax.legend(loc='center', ncols=4)
    ax.set_ylim(0, 1)
    plt.show()

# Validation testing for the L2 Regularization strength
def nn_graph_2(Xtrain, Ytrain, Xval, Yval):
    mlp1 = MLPClassifier(hidden_layer_sizes = (4,8,16,32,64,), max_iter = 500, random_state = 5, batch_size = 8, alpha = 0, solver = 'sgd')
    mlp2 = MLPClassifier(hidden_layer_sizes = (4,8,16,32,64,), max_iter = 500, random_state = 5, batch_size = 8, alpha = 0.0001, solver = 'sgd')
    mlp3 = MLPClassifier(hidden_layer_sizes = (4,8,16,32,64,), max_iter = 500, random_state = 5, batch_size = 8, alpha = 0.001, solver = 'sgd')
    mlp1.fit(Xtrain, Ytrain)
    mlp2.fit(Xtrain, Ytrain)
    mlp3.fit(Xtrain, Ytrain)
    mlp1_accuracy = accuracy(Yval, mlp1.predict(Xval))
    mlp2_accuracy = accuracy(Yval, mlp2.predict(Xval))
    mlp3_accuracy = accuracy(Yval, mlp3.predict(Xval))
    mlp1_f1 = f1_score(Yval, mlp1.predict(Xval))
    mlp2_f1 = f1_score(Yval, mlp2.predict(Xval))
    mlp3_f1 = f1_score(Yval, mlp3.predict(Xval))
    lambda_values = ('0','0.0001', '0.001')
    random_forest_results = {
        'accuracy' : (mlp1_accuracy, mlp2_accuracy, mlp3_accuracy),
        'f1' : (mlp1_f1, mlp2_f1, mlp3_f1)
    }
    x = np.arange(len(lambda_values))
    width = 0.25  
    multiplier = 0
    fig, ax = plt.subplots(layout='constrained')
    for attribute, measurement in random_forest_results.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    ax.set_title('Accuracy and F1 score of Different L2 Regularization Strengths')
    ax.set_xticks(x + width, lambda_values)
    ax.legend(loc='center', ncols=4)
    ax.set_ylim(0, 1)
    plt.show()

def main():
    data = pd.read_csv("mvp.csv")
    Xtrain, Ytrain, Xval, Yval, Xtest, Ytest, player_names = preprocess(data)
    Xtest_2023, Ytest_2023, Xtest_2022, Ytest_2022, Xtest_2021, Ytest_2021, Xtest_2020, Ytest_2020, Xtest_2019, Ytest_2019 = preprocess_test(Xtest, Ytest)
    players_2019, players_2020, players_2021, players_2022, players_2023 = preprocess_names(player_names)
    # Uncomment out the lines below to get the data distributions
    #sns.catplot(data = Xtrain)
    #sns.catplot(data = Xtest)
    #plt.show()

    # Logistic Regression model training/validation/testing
    print("Logistic Regression Results:")
    log_regression = LogisticRegression(max_iter = 500, random_state = 5)
    log_regression.fit(Xtrain, Ytrain)
    print("Accuracy: " + str(accuracy(Yval, log_regression.predict(Xval))))
    print("F1 Score: " + str(f1_score(Yval, log_regression.predict(Xval))))
    print("2023 - " + players_2023[find_max_idx(log_regression.predict_proba(Xtest_2023))])
    print("2022 - " + players_2022[find_max_idx(log_regression.predict_proba(Xtest_2022))])
    print("2021 - " + players_2021[find_max_idx(log_regression.predict_proba(Xtest_2021))])
    print("2020 - " + players_2020[find_max_idx(log_regression.predict_proba(Xtest_2020))])
    print("2019 - " + players_2019[find_max_idx(log_regression.predict_proba(Xtest_2019))])

    # Random Forest model training/validation/testing
    print("\nRandom Forest Results:")
    # Uncomment out the lines below to get the validation graphs
    #random_forest_graphs_1(Xtrain, Ytrain, Xval, Yval)
    #random_forest_graphs_2(Xtrain, Ytrain, Xval, Yval)
    random_forest = RandomForestClassifier(n_estimators = 250, criterion = "entropy", random_state = 5)
    random_forest.fit(Xtrain, Ytrain)
    print("Accuracy: " + str(accuracy(Yval, random_forest.predict(Xval))))
    print("F1 Score: " + str(f1_score(Yval, random_forest.predict(Xval))))
    print("2023 - " + players_2023[find_max_idx(random_forest.predict_proba(Xtest_2023))])
    print("2022 - " + players_2022[find_max_idx(random_forest.predict_proba(Xtest_2022))])
    print("2021 - " + players_2021[find_max_idx(random_forest.predict_proba(Xtest_2021))])
    print("2020 - " + players_2020[find_max_idx(random_forest.predict_proba(Xtest_2020))])
    print("2019 - " + players_2019[find_max_idx(random_forest.predict_proba(Xtest_2019))])

    # Neural Network model training/validation/testing
    print("\nNeural Network Results:")
    # Uncomment out the lines below to get the validation graphs
    #nn_graph_1(Xtrain, Ytrain, Xval, Yval)
    #nn_graph_2(Xtrain, Ytrain, Xval, Yval)
    mlp = MLPClassifier(hidden_layer_sizes = (4,8,16,32,64,), max_iter = 500, random_state = 5, batch_size = 8, alpha = 0.001, solver = 'sgd', verbose = False) # Set verbose to true to get loss per epoch
    mlp.fit(Xtrain, Ytrain)
    # Uncomment out the lines below to get the training accuracy and f1 score
    #print(str(accuracy(Ytrain, mlp.predict(Xtrain))))
    #print(str(f1_score(Ytrain, mlp.predict(Xtrain))))
    print(str(f1_score(Ytrain, mlp.predict(Xtrain))))
    print("Accuracy: " + str(accuracy(Yval, mlp.predict(Xval))))
    print("F1 Score: " + str(f1_score(Yval, mlp.predict(Xval))))
    print("2023 - " + players_2023[find_max_idx(mlp.predict_proba(Xtest_2023))])
    print("2022 - " + players_2022[find_max_idx(mlp.predict_proba(Xtest_2022))])
    print("2021 - " + players_2021[find_max_idx(mlp.predict_proba(Xtest_2021))])
    print("2020 - " + players_2020[find_max_idx(mlp.predict_proba(Xtest_2020))])
    print("2019 - " + players_2019[find_max_idx(mlp.predict_proba(Xtest_2019))])

    # Baseline (Preseason Odds): 
    # 2019 - Lebron James
    # 2020 - Giannis
    # 2021 - Giannis
    # 2022 - Luka Doncic 
    # 2023 - Luka Doncic 

    

if __name__ == "__main__":
    main()