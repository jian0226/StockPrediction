import numpy as np
import pandas as pd
from technical_indicators import *
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor


def assemble_predictors(O, H, L, C, V):

    # Compute 10 technical indicators:
    window = 14
    SMA = get_simple_moving_average(C, window)
    EWMA = get_exponential_weighted_moving_average(C)
    UB, LB = get_acceleration_bands(SMA, H[window - 1:], L[window - 1:])
    PROC = get_price_rate_of_change(C)
    OBV = get_on_balance_volume(C, V)
    MACD = get_MACD(C)
    WR = get_Williams(H, L, C)
    SOSCI = get_stochastic_oscillator(H, L, C)
    RSI = get_RSI(C)

    Std_Dev = get_standard_deviation(C, window)
    MFI = get_money_flow_index(H, L, C, V, window)

    # Process
    remove_pts = 2 * window - 3
    num_points = len(H) - remove_pts

    #
    num_predictors = 17
    Predictors = np.zeros((num_points, num_predictors), dtype=float)

    # Here we define the predictors in order of importance. Notice
    # this is the same order that is listed in the paper. This means
    # We can use Predictors[:, :1] to use only the best variable, or
    # we can use Predictors[:, :5] to use only the best 5 variables, etc...

    Predictors[:, 0] = OBV[-num_points:]
    Predictors[:, 1] = MACD[-num_points:]
    Predictors[:, 2] = Std_Dev[-num_points:]
    Predictors[:, 3] = SMA[-num_points:]
    Predictors[:, 4] = EWMA[-num_points:]
    Predictors[:, 5] = RSI[-num_points:]
    Predictors[:, 6] = UB[-num_points:]
    Predictors[:, 7] = LB[-num_points:]
    Predictors[:, 8] = V[-num_points:]
    Predictors[:, 9] = C[-num_points:]
    Predictors[:, 10] = L[-num_points:]
    Predictors[:, 11] = H[-num_points:]
    Predictors[:, 12] = PROC[-num_points:]
    Predictors[:, 13] = O[-num_points:]
    Predictors[:, 14] = WR[-num_points:]
    Predictors[:, 15] = SOSCI[-num_points:]
    Predictors[:, 16] = MFI[-num_points:]

    return Predictors


# Note, here we change the number of predictors to use:
num_predictors_to_keep = 10


ticker_list = ['HD', 'GS', 'IBM', 'INTC', 'AXP', 'AAPL', 'CAT', 'KO', 'DIS', 'CVX', 'CSCO', 'BA', 'GE', 'JPM',
               'MCD', 'JNJ', 'MRK', 'NKE', 'PFE', 'VZ', 'V', 'UNH', 'UTX', 'PG']
accuracy_nn = np.zeros(len(ticker_list))
accuracy_rf = np.zeros(len(ticker_list))

importance_of_each_predictor = np.zeros(num_predictors_to_keep, dtype=float)

path_to_data = '../01_Official_Data/'

#
for ticker_idx in range(len(ticker_list)):

    df = pd.read_csv(path_to_data + ticker_list[ticker_idx] + '.csv')

    O = np.array(df['Open'])
    H = np.array(df['High'])
    L = np.array(df['Low'])
    C = np.array(df['Close'])
    V = np.array(df['Volume'])

    # Compute Predictors:
    X = assemble_predictors(O, H, L, C, V)

    # Normalize data, so that all columns have the same norm. Otherwise, Neural Network will fail.
    for col in range(17):
        X[:, col] = X[:, col] / np.linalg.norm(X[:, col])

    # Compute Output Log Return:
    return_window = 5  # We use information at time "t" to predict the return at time "t + return_window"
    LogReturn = get_log_return(C, return_window)
    y = LogReturn[-len(X):]

    # Apply time shift (to make sure we are actually predicting future)
    X = X[:-return_window, :]   # keep everything except the last 'return_window' rows (and all cols)
    y = y[return_window:]       # keep everything except the first 'return_window' rows.

    # In this line, we select only the predictors we want to use.
    # If we do nothing, we are using 17 predictors.
    X = X[:, :num_predictors_to_keep]

    # Train-Test data split:
    num_train = 1700
    num_test = len(X) - num_train

    X_train = X[:num_train]
    y_train = y[:num_train]

    X_test = X[num_train:]
    y_test = y[num_train:]

    # NEURAL NETWORK BEGINS HERE:
    # We use 25 neurons and 50 hidden layers.
    # 'lbfgs' is a method to solve the backpropagation (Quasi-Newton)
    our_neural_network = MLPRegressor(solver='lbfgs', activation='tanh', alpha=1e-7, hidden_layer_sizes=(25, 50), random_state=1)
    our_random_forest = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=0)

    # This line trains the model:
    our_neural_network.fit(X_train, y_train)
    our_random_forest.fit(X_train, y_train)

    importance_of_each_predictor += our_random_forest.feature_importances_

    # This is the prediction (on test data, not train)
    # The Neural Network hasn't "seen" X_test.
    y_pred_nn = our_neural_network.predict(X_test)
    y_pred_rf = our_random_forest.predict(X_test)


    # Check if the prediction is good or not:
    total_test_nn = 0
    total_test_rf = 0
    for i in range(len(y_test)):
        total_test_nn += 0.5 * (np.sign(y_test[i]*y_pred_nn[i]) + 1)
        total_test_rf += 0.5 * (np.sign(y_test[i]*y_pred_rf[i]) + 1)

    accuracy_nn[ticker_idx] = 100 * int(total_test_nn) / num_test  # this is a percentage.
    accuracy_rf[ticker_idx] = 100 * int(total_test_rf) / num_test  # this is a percentage.

print("Our average guess rate (NN) is ", np.mean(accuracy_nn))
print("Our average guess rate (RF) is ", np.mean(accuracy_rf))

#print("Variable Importance: ", importance_of_each_predictor)
