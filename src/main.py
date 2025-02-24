import pandas as pd
import pandas_ta as ta
from backtesting import Backtest
from tqdm import tqdm
import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.backtest import MyStrat

tqdm.pandas()


def read_csv_to_dataframe(file_path):
    df = pd.read_csv(file_path)
    df["Gmt time"] = df["Gmt time"].str.replace(".000", "")
    df['Gmt time'] = pd.to_datetime(df['Gmt time'], format='%d.%m.%Y %H:%M:%S')
    df = df[df.High != df.Low]
    df.set_index("Gmt time", inplace=True)
    return df


def read_data_folder(folder_path="./data"):
    dataframes = []
    file_names = []
    for file_name in tqdm(os.listdir(folder_path)):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            df = read_csv_to_dataframe(file_path)
            dataframes.append(df)
            file_names.append(file_name)
    return dataframes, file_names


def total_signal(df, current_candle):
    current_pos = df.index.get_loc(current_candle)

    # Buy condition
    c1 = df['High'].iloc[current_pos] > df['High'].iloc[current_pos - 1]
    c2 = df['High'].iloc[current_pos - 1] > df['Low'].iloc[current_pos]
    c3 = df['Low'].iloc[current_pos] > df['High'].iloc[current_pos - 2]
    c4 = df['High'].iloc[current_pos - 2] > df['Low'].iloc[current_pos - 1]
    c5 = df['Low'].iloc[current_pos - 1] > df['High'].iloc[current_pos - 3]
    c6 = df['High'].iloc[current_pos - 3] > df['Low'].iloc[current_pos - 2]
    c7 = df['Low'].iloc[current_pos - 2] > df['Low'].iloc[current_pos - 3]

    if c1 and c2 and c3 and c4 and c5 and c6 and c7:
        return 2

    # Symmetrical conditions for short (sell condition)
    c1 = df['Low'].iloc[current_pos] < df['Low'].iloc[current_pos - 1]
    c2 = df['Low'].iloc[current_pos - 1] < df['High'].iloc[current_pos]
    c3 = df['High'].iloc[current_pos] < df['Low'].iloc[current_pos - 2]
    c4 = df['Low'].iloc[current_pos - 2] < df['High'].iloc[current_pos - 1]
    c5 = df['High'].iloc[current_pos - 1] < df['Low'].iloc[current_pos - 3]
    c6 = df['Low'].iloc[current_pos - 3] < df['High'].iloc[current_pos - 2]
    c7 = df['High'].iloc[current_pos - 2] < df['High'].iloc[current_pos - 3]

    if c1 and c2 and c3 and c4 and c5 and c6 and c7:
        return 1

    return 0


def add_total_signal(df):
    df['TotalSignal'] = df.progress_apply(lambda row: total_signal(df, row.name), axis=1)
    return df

def add_atr(df, length=10):
    df['atr'] = ta.atr(df['High'], df['Low'], df['Close'], length)
    return df

def add_pointpos_column(df, signal_column):
    """
    Adds a 'pointpos' column to the DataFrame to indicate the position of support and resistance points.

    Parameters:
    df (DataFrame): DataFrame containing the stock data with the specified SR column, 'Low', and 'High' columns.
    sr_column (str): The name of the column to consider for the SR (support/resistance) points.

    Returns:
    DataFrame: The original DataFrame with an additional 'pointpos' column.
    """

    def pointpos(row):
        if row[signal_column] == 2:
            return row['Low'] - 1e-4
        elif row[signal_column] == 1:
            return row['High'] + 1e-4
        else:
            return np.nan

    df['pointpos'] = df.apply(lambda row: pointpos(row), axis=1)
    return df


def plot_candlestick_with_signals(df, start_index, num_rows):
    """
    Plots a candlestick chart with signal points.

    Parameters:
    df (DataFrame): DataFrame containing the stock data with 'Open', 'High', 'Low', 'Close', and 'pointpos' columns.
    start_index (int): The starting index for the subset of data to plot.
    num_rows (int): The number of rows of data to plot.

    Returns:
    None
    """
    df_subset = df[start_index:start_index + num_rows]

    fig = make_subplots(rows=1, cols=1)

    fig.add_trace(go.Candlestick(x=df_subset.index,
                                 open=df_subset['Open'],
                                 high=df_subset['High'],
                                 low=df_subset['Low'],
                                 close=df_subset['Close'],
                                 name='Candlesticks'),
                  row=1, col=1)

    fig.add_trace(go.Scatter(x=df_subset.index, y=df_subset['pointpos'], mode="markers",
                             marker=dict(size=10, color="MediumPurple", symbol='circle'),
                             name="Entry Points"),
                  row=1, col=1)

    fig.update_layout(
        width=1200,
        height=800,
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white'),
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False),
        showlegend=True,
        legend=dict(
            x=0.01,
            y=0.99,
            traceorder="normal",
            font=dict(
                family="sans-serif",
                size=12,
                color="white"
            ),
            bgcolor="black",
            bordercolor="gray",
            borderwidth=2
        )
    )

    fig.show()

def display_results(results, file_names):
    agg_returns = sum([r["Return [%]"] for r in results])
    num_trades = sum([r["# Trades"] for r in results])
    max_drawdown = min([r["Max. Drawdown [%]"] for r in results])
    avg_drawdown = sum([r["Avg. Drawdown [%]"] for r in results]) / len(results)

    win_rate = sum([r["Win Rate [%]"] for r in results]) / len(results)
    best_trade = max([r["Best Trade [%]"] for r in results])
    worst_trade = min([r["Worst Trade [%]"] for r in results])
    avg_trade = sum([r["Avg. Trade [%]"] for r in results]) / len(results)
    max_trade_duration = max([r["Max. Trade Duration"] for r in results])
    # avg_trade_duration = sum([r["Avg. Trade Duration"] for r in results]) / len(results)

    print(f"Aggregated Returns: {agg_returns:.2f}%")
    print(f"Number of Trades: {num_trades}")
    print(f"Maximum Drawdown: {max_drawdown:.2f}%")
    print(f"Average Drawdown: {avg_drawdown:.2f}%")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Best Trade: {best_trade:.2f}%")
    print(f"Worst Trade: {worst_trade:.2f}%")
    print(f"Average Trade: {avg_trade:.2f}%")
    print(f"Maximum Trade Duration: {max_trade_duration} days")
    # print(f"Average Trade Duration: {avg_trade_duration:.2f} days")
    equity_curve(results, file_names)

def equity_curve(results, file_names):
    equity_curves = [stats['_equity_curve']['Equity'] for stats in results]
    max_length = max(len(equity) for equity in equity_curves)

    # Pad each equity curve with the first value to match the maximum length
    padded_equity_curves = []
    for equity in equity_curves:
        first_value = equity.iloc[0]
        padding = [first_value] * (max_length - len(equity))
        padded_equity = padding + equity.tolist()
        padded_equity_curves.append(padded_equity)

    equity_df = pd.DataFrame(padded_equity_curves).T

    import matplotlib.pyplot as plt

    equity_df.plot(kind='line', figsize=(10, 6), legend=True).set_facecolor('black')
    plt.gca().spines['bottom'].set_color('black')
    plt.gca().spines['left'].set_color('black')
    plt.gca().tick_params(axis='x', colors='black')
    plt.gca().tick_params(axis='y', colors='black')
    plt.gca().set_facecolor('black')
    plt.legend(file_names)


def main():
    folder_path = "./data/data_forex"
    dataframes, file_names = read_data_folder(folder_path)

    for i, df in enumerate(dataframes):
        print("working on dataframe ", i, "...")
        df = add_atr(df)
        df = add_total_signal(df)
        df = add_pointpos_column(df, "TotalSignal")
        dataframes[i] = df  # Update the dataframe in the list

    print(sum([frame["TotalSignal"].value_counts() for frame in dataframes], start=0))
    # plot_candlestick_with_signals(dataframes[0], start_index=300, num_rows=355)

    results = []
    heatmaps = []

    for df in dataframes:
        bt = Backtest(df, MyStrat, cash=5000, margin=1 / 5, commission=0.0002)
        stats, heatmap = bt.optimize(
                                    # slperc=[i / 100 for i in range(1, 8)],
                                    # tpperc=[i / 100 for i in range(1, 8)],
                                    sl_atr_ratio=[i/10 for i in range(1, 100)],
                                    maximize='Return [%]', max_tries=3000,
                                    random_state=0,
                                    return_heatmap=True)
        # print(bt.run())
        # bt.plot(plot_return=True, plot_drawdown=True, smooth_equity=True)

        results.append(stats)
        heatmaps.append(heatmap)

    display_results(results, file_names)

    import seaborn as sns
    import matplotlib.pyplot as plt

    # Convert multiindex series to dataframe
    # heatmap_df = heatmaps[0].unstack()
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmaps, annot=True, cmap='viridis', fmt='.0f')
    plt.show()

if __name__ == '__main__':
    main()