import numpy as np

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


def get_log_return(C, return_window):
    LogReturn = np.zeros(len(C)-1, dtype=float)
    for i in range(len(C) - return_window):
        LogReturn[i] = np.log10(C[i+return_window] / C[i])

    return LogReturn

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