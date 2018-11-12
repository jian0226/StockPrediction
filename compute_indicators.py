import numpy as np
import pandas as pd
from scipy import stats
# from matplotlib import pyplot as plt


def get_simple_moving_average(C, window=14):

    num_points = len(C)
    SMA = np.zeros(num_points - window + 1, dtype=float)

    # Compute Simple Moving Average:
    for i in range(num_points - window + 1):
        SMA[i] = sum( C[i:i+window] ) / window

    return SMA


def get_acceleration_bands(SMA, H, L):

    upper_band = SMA * (H * (1 + 4 * (H - L) / (H + L) ))
    lower_band = SMA * (H * (1 - 4 * (H - L) / (H + L) ))

    return upper_band, lower_band


def get_exponential_weighted_moving_average(C, window=14):
    C = C.squeeze()
    prev_ema = C[:window].mean()
    ema = [prev_ema]
    m = len(C)
    multiplier = 2 / float(window + 1)
    for i in range(window, m):
        cur_ema = (C[i] - prev_ema) * multiplier + prev_ema
        prev_ema = cur_ema
        ema.append(cur_ema)
    return np.array(ema)


def get_RSI(C):
    # Relative strength index:

    C = C.squeeze()
    n = len(C)
    x0 = C[:n - 1]
    x1 = C[1:]
    change = x1 - x0
    avgGain = []
    avgLoss = []
    loss = 0
    gain = 0
    for i in range(14):
        if change[i] > 0:
            gain += change[i]
        elif change[i] < 0:
            loss += abs(change[i])
    averageGain = gain / 14.0
    averageLoss = loss / 14.0
    avgGain.append(averageGain)
    avgLoss.append(averageLoss)
    for i in range(14, n - 1):
        if change[i] >= 0:
            avgGain.append((avgGain[-1] * 13 + change[i]) / 14.0)
            avgLoss.append((avgLoss[-1] * 13) / 14.0)
        else:
            avgGain.append((avgGain[-1] * 13) / 14.0)
            avgLoss.append((avgLoss[-1] * 13 + abs(change[i])) / 14.0)
    avgGain = np.array(avgGain)
    avgLoss = np.array(avgLoss)
    RS = avgGain / avgLoss
    RSI = 100 - (100 / (1 + RS))

    return RSI


def get_stochastic_oscillator(H,L,C):

    n = len(H)
    highestHigh = []
    lowestLow = []
    for i in range(n - 13):
        highestHigh.append(H[i:i + 14].max())
        lowestLow.append(L[i:i + 14].min())
    highestHigh = np.array(highestHigh)
    lowestLow = np.array(lowestLow)
    k = 100 * ((C[13:] - lowestLow) / (highestHigh - lowestLow))

    return k


def get_Williams(H, L, C):

    n = len(H)
    highestHigh = []
    lowestLow = []
    for i in range(n - 13):
        highestHigh.append(H[i:i + 14].max())
        lowestLow.append(L[i:i + 14].min())
    highestHigh = np.array(highestHigh)
    lowestLow = np.array(lowestLow)
    w = -100 * ((highestHigh - C[13:]) / (highestHigh - lowestLow))

    return w


def get_MACD(close):

    ma1 = get_exponential_weighted_moving_average(close.squeeze(), 12)
    ma2 = get_exponential_weighted_moving_average(close.squeeze(), 26)
    macd = ma1[14:] - ma2

    return macd


def get_price_rate_of_change(C, n_days=1):

    C = C.squeeze()
    n = len(C)

    x0 = C[:n - n_days]
    x1 = C[n_days:]
    PriceRateOfChange = (x1 - x0) / x0

    return PriceRateOfChange


def get_on_balance_volume(C, V):
    # getOnBalanceVolume takes C, V, where
    # C stands for closing price.
    # V stands for trading volume.
    # The output is the variable OBV,
    # length is number of points - 2:

    C = C[1:]
    n = len(C)
    x0 = C[:n - 1]
    x1 = C[1:]
    change = x1 - x0
    OBV = []
    prev_OBV = 0

    for i in range(n - 1):
        if change[i] > 0:
            current_OBV = prev_OBV + V[i]
        elif change[i] < 0:
            current_OBV = prev_OBV - V[i]
        else:
            current_OBV = prev_OBV
        OBV.append(current_OBV)
        prev_OBV = current_OBV
    OBV = np.array(OBV)

    return OBV


def ease_of_movement(H, L):

    # Allocate memory for EMV:
    EMV = np.zeros(len(H), dtype=float)

    for i in range(len(H) - 1):
        distance_moved = ((H[i+1] + L[i+1]) / 2 - (H[i] + L[i]) / 2)
        box_ratio = ( (V[i+1]/100000000) / (H[i+1] - L[i+1]) )
        if box_ratio == 0:
            print("We have found zero box ratio")

        EMV[i+1] = distance_moved / box_ratio


    return EMV


def volume_rate_of_change(V):

    VRC = np.zeros(len(H), dtype=float)
    for i in range(len(V) - 1):
        if V[i] == 0:
            print("We have found zero volume!")

        VRC[i+1] = (V[i+1] / V[i]) - 1

    return VRC


def get_standard_deviation(C, return_window):

    Std_Dev = np.zeros(len(C) - return_window, dtype=float)
    for i in range(len(Std_Dev)):
        Std_Dev[i] = np.std(C[i:i+return_window])

    return Std_Dev


def get_money_flow_index(H, L, C, V, return_window):

    MFI = np.zeros(len(C)-return_window, dtype=float)

    for i in range(len(MFI)):

        Pivot = (H[i:i+return_window] + L[i:i+return_window] + C[i:i+return_window]) / 3

        Money_Flow = Pivot * np.sum(V[i:i+return_window])

        Sum_Positive_Money_Flow = np.sum([x for x in Money_Flow if x > 0])
        Sum_Negative_Money_Flow = np.sum([x for x in Money_Flow if x < 0])

        if Sum_Negative_Money_Flow == 0:
            Sum_Negative_Money_Flow = 1.0
        Money_Ratio = Sum_Positive_Money_Flow / Sum_Negative_Money_Flow

        MFI[i] = 100 - (100 / (1 + Money_Ratio))

    return MFI


def get_log_return(C, return_window):
    LogReturn = np.zeros(len(C)-1, dtype=float)
    for i in range(len(C) - return_window):
        LogReturn[i] = np.log10(C[i+return_window] / C[i])

    return LogReturn

# ADT, ALTR, DTV, UA,
ticker_list = ['ABT', 'ABBV', 'ACN', 'ADBE', 'AAP', 'AES', 'AET', 'AFL', 'AMG', 'A', 'APD', 'AKAM', 'AA', 'AGN',
               'ALXN', 'ALLE', 'ADS', 'ALL', 'MO', 'AMZN', 'AAL', 'AEP', 'AXP', 'AMT', 'AMP', 'AME', 'AMGN','APH',
               'APC', 'ADI', 'AON', 'APA', 'AIV', 'AMAT', 'ADM', 'AIZ', 'T', 'ADSK','ADP','AN', 'AZO', 'AVGO', 'AVB',
               'AVY', 'AEE', 'AIG', 'ABC', 'BLL', 'BAC', 'BK', 'BAX', 'BBT','BDX', 'BBBY', 'BRK-B', 'BBY','BLX',
               'HRB', 'BA', 'BWA', 'BXP', 'BMY', 'BF-B', 'CHRW', 'CA', 'COG', 'CPB', 'COF', 'CAH', 'HSIC','KMX', 'CCL',
               'CAT', 'CBS','CELG', 'CNP', 'CTL', 'CERN', 'CF', 'SCHW', 'CHK', 'CVX', 'CMG', 'CB', 'CI', 'XEC', 'CINF',
               'CTAS', 'CSCO', 'C', 'CTXS', 'CLX', 'CME', 'CMS', 'KO', 'CCE', 'CTSH', 'CL', 'CMCSA', 'CMA', 'COP',
               'CNX', 'ED', 'CMI', 'CVS', 'DHI', 'DHR', 'DRI', 'DVA', 'DE', 'DLPH', 'DAL', 'XRAY', 'DVN',
               'DO', 'DFS', 'DISCA', 'DISCK', 'DG', 'DLTR', 'D', 'DOV', 'DTE', 'DUK', 'DNB', 'ETFC', 'EMN',
               'ETN', 'EBAY', 'ECL', 'EIX', 'EW', 'EA', 'EMR', 'ENDP', 'ESV', 'ETR', 'EOG', 'EQT', 'EFX', 'EQIX',
               'EQR', 'ESS', 'EL', 'ES', 'EXC', 'EXPE', 'EXPD', 'ESRX', 'XOM', 'FFIV', 'FB', 'FAST', 'FDX', 'FIS',
               'FITB', 'FSLR', 'FE', 'FLIR', 'FLS', 'FLR', 'FMC', 'FTI', 'F', 'FOSL', 'BEN', 'FCX', 'FTR', 'GME', 'GPS',
               'GRMN', 'GD', 'GE', 'GGP', 'GIS', 'GM', 'GPC', 'GNW', 'GILD', 'GS', 'GT', 'GOOGL', 'GOOG', 'GWW', 'HOG',
               'HRS', 'HIG', 'HAS', 'HCA', 'HCP', 'HP', 'HES', 'HPQ', 'HD', 'HON', 'HRL', 'HST', 'HUM', 'HBAN', 'ITW',
               'IR', 'INTC', 'ICE', 'IBM', 'IP', 'IPG', 'IFF', 'INTU', 'ISRG', 'IVZ', 'IRM', 'JEC', 'JBHT', 'JNJ', 'JCI',
               'JPM', 'JNPR', 'KSU', 'K', 'KEY', 'KMB', 'KIM', 'KMI', 'KLAC', 'KSS', 'KR', 'LB', 'LLL', 'LH', 'LRCX',
               'LM', 'LEG', 'LEN', 'LLY', 'LNC', 'LMT', 'L', 'LOW', 'LYB', 'MTB', 'MAC', 'M', 'MNK', 'MRO', 'MPC', 'MAR',
               'MMC', 'MLM', 'MAS', 'MA', 'MAT', 'MKC', 'MCD', 'MCK', 'MMV', 'MDT', 'MRK', 'MET', 'KORS', 'MCHP', 'MU',
               'MSFT', 'MHK', 'TAP', 'MDLZ', 'MNST', 'MCO', 'MS', 'MOS', 'MSI', 'MUR', 'MYL', 'NDAQ', 'NOV', 'NAVI',
               'NFLX', 'NWL', 'NEM', 'NWSA', 'NEE', 'NLSN', 'NKE', 'NI', 'NE', 'NBL', 'JWN', 'NSC', 'NTRS', 'NOC', 'NRG',
               'NUE', 'NVDA', 'ORLY', 'OXY', 'OMC', 'OKE', 'ORCL', 'OI', 'PCAR', 'PH', 'PDCO', 'PAYX', 'PNR', 'PBCT',
               'PEP', 'PKI', 'PRGO', 'PFE', 'PCG', 'PM', 'PSX', 'PNW', 'PXD', 'PBI', 'PNC', 'RL', 'PPG', 'PPL', 'PX',
               'PFG', 'PG', 'PGR', 'PLD', 'PRU', 'PEG', 'PSA', 'PHM', 'PVH', 'QRVO', 'PWR', 'QCOM', 'DGX', 'RRC', 'RTN',
               'O', 'RHT', 'REGN', 'RF', 'RSG', 'RHI', 'ROK', 'COL', 'ROP', 'ROST', 'R', 'CRM', 'SCG', 'SLB', 'STX',
               'SEE', 'SRE', 'SHW', 'SPG', 'SWKS', 'SLG', 'SJM', 'SNA', 'SO', 'LUV', 'SWN', 'SE', 'SWK', 'SBUX', 'STT',
               'SRCL', 'SYK', 'STI', 'SYMC', 'SYY', 'TROW', 'TGT', 'TEL', 'THC', 'TDC', 'TXN', 'TXT', 'HSY',
               'TMO', 'TIF', 'TMK', 'TSS', 'TSCO', 'RIG', 'TRIP', 'FOXA', 'TSN', 'UNP', 'UNH', 'UPS', 'URI', 'UTX',
               'UHS', 'UNM', 'URBN', 'VFC', 'VLO', 'VAR', 'VTR', 'VRSN', 'VZ', 'VRTX', 'VIAB', 'V', 'VNO', 'VMC', 'WMT',
               'WBA', 'DIS', 'WM', 'WAT', 'ANTM', 'WFC', 'WDC', 'WU', 'WY', 'WHR', 'WYNN', 'XEL', 'XRX', 'XLNX', 'XL',
               'XYL', 'YUM', 'ZBH', 'ZION', 'ZTS', 'AAPL']

# Main code begins here:

path_to_data = '../01_Official_Data/'
feature_names = ['SMA', 'EWMA', 'UB', 'LB', 'PROC', 'OBV', 'MACD', 'WR', 'SOSCI', 'RSI', 'O', 'H', 'L', 'C', 'V', 'Std_Dev', 'MFI']
Correlation_Total_For_All_Stocks = np.zeros(len(feature_names), dtype=float)


for ticker in ticker_list:     # We run this 400 times

    print("Working on ", ticker)

    df = pd.read_csv(path_to_data + ticker + '.csv')

    O = np.array(df['Open'])
    H = np.array(df['High'])
    L = np.array(df['Low'])
    C = np.array(df['Close'])
    V = np.array(df['Volume'])

    window = 14

    # Compute 10 technical indicators:
    SMA = get_simple_moving_average(C, window)
    EWMA = get_exponential_weighted_moving_average(C)
    UB, LB = get_acceleration_bands(SMA, H[window-1:], L[window-1:])
    PROC = get_price_rate_of_change(C)
    OBV = get_on_balance_volume(C, V)
    MACD = get_MACD(C)
    WR = get_Williams(H, L, C)
    SOSCI = get_stochastic_oscillator(H,L,C)
    RSI = get_RSI(C)
    Std_Dev = get_standard_deviation(C, window)
    MFI = get_money_flow_index(H, L, C, V, window)

    # Process
    remove_pts = 2*window - 3
    num_points = len(H) - remove_pts

    # Compute Log Return, keep last
    return_window = 5
    LogReturn = get_log_return(C, return_window)
    LogReturn = LogReturn[-num_points:]

    #
    num_predictors = 17
    Predictors = np.zeros((num_points, num_predictors), dtype=float)

    # Keep the last 'num_points', fill-out one column at a time:
    Predictors[:, 0] = SMA[-num_points:]
    Predictors[:, 1] = EWMA[-num_points:]
    Predictors[:, 2] = UB[-num_points:]
    Predictors[:, 3] = LB[-num_points:]
    Predictors[:, 4] = PROC[-num_points:]
    Predictors[:, 5] = OBV[-num_points:]
    Predictors[:, 6] = MACD[-num_points:]
    Predictors[:, 7] = WR[-num_points:]
    Predictors[:, 8] = SOSCI[-num_points:]
    Predictors[:, 9] = RSI[-num_points:]
    Predictors[:,10] = O[-num_points:]
    Predictors[:,11] = H[-num_points:]
    Predictors[:,12] = L[-num_points:]
    Predictors[:,13] = C[-num_points:]
    Predictors[:,14] = V[-num_points:]
    Predictors[:,15] = Std_Dev[-num_points:]
    Predictors[:,16] = MFI[-num_points:]

    # Compute correlation of the 10 indicators to the logReturn.
    # This calculation is for a single stock.
    Correlation_For_One_Stock = np.zeros((num_predictors), dtype=float)
    for pred_index in range(len(Correlation_For_One_Stock)):   # We run this 10 times

        # We compute a shifted correlation.
        # This function outputs two numbers, we keep first.
        tmp = stats.pearsonr(Predictors[:-return_window, pred_index], LogReturn[return_window:])
        Correlation_For_One_Stock[pred_index] = tmp[0]

    # Add result: We use 'abs' to keep positive & negative
    Correlation_Total_For_All_Stocks = Correlation_Total_For_All_Stocks + abs(Correlation_For_One_Stock)

# Print Features:
feature_idx = np.argsort(-Correlation_Total_For_All_Stocks)
for idx in feature_idx:
    print(feature_names[idx], ' = ', Correlation_Total_For_All_Stocks[idx]/len(ticker_list))

