import bs4 as bs
import datetime as dt
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
import pandas_datareader as pdr
from alpha_vantage.timeseries import TimeSeries
import pickle
import requests
import re

style.use('ggplot')

# GET ALL INTRADAY DATA FOR PAST YEAR OR TWO instead of daily data of last 18 years

def get_yahoo_crumb_cookie():
    """Get Yahoo crumb cookie value."""
    res = requests.get('https://finance.yahoo.com/quote/SPY/history')
    yahoo_cookie = res.cookies['B']
    yahoo_crumb = None
    pattern = re.compile('.*"CrumbStore":\{"crumb":"(?P<crumb>[^"]+)"\}')
    for line in res.text.splitlines():
        m = pattern.match(line)
        if m is not None:
            yahoo_crumb = m.groupdict()['crumb']
    return yahoo_cookie, yahoo_crumb

def save_sp500_tickers():
    resp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text,'lxml')
    table = soup.find('table',{'class':'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        if ticker =='BF.B':
            ticker = 'BF-B'
        elif ticker =='BRK.B':
            ticker = 'BRK-B'
        tickers.append(ticker)
    with open("sp500tickers.pickle","wb") as f:
        pickle.dump(tickers,f)
    print(tickers)
    return tickers

# save_sp500_tickers()



def get_data_from_alpha_vantage(reload_sp500=False):
    cookie,crumb=get_yahoo_crumb_cookie()
    if reload_sp500:
        tickers = save_sp500_tickers()
    else:
        with open("sp500tickers.pickle", "rb") as f:
            tickers = pickle.load(f)

    # going to store this locally
    if not os.path.exists('stock_dfs'):
        os.makedirs('stock_dfs')

    starttime = dt.datetime(2000,1,1)
    endtime = dt.datetime(2016,12,31)

    for ticker in tickers:
        print(ticker)
        if not os.path.exists('stock_dfs/{}.csv'.format(ticker)):
            ts = TimeSeries(key='(ENTER ALPHA VANTAGE KEY HERE',output_format='pandas')
            # df = web.DataReader(ticker,'yahoo',start,end)
            # df = pdr.get_data_yahoo(ticker,start=starttime,end=endtime)
            data,meta_Data = ts.get_daily_adjusted(symbol=ticker,outputsize='full')
            data.to_csv('stock_dfs/{}.csv'.format(ticker))
        else:
            print('Already have {}'.format(ticker))

# get_data_from_alpha_vantage()
# We only got 47 Files

def compile_data():
    with open("sp500tickers.pickle","rb") as f:
        tickers = pickle.load(f)

    main_df = pd.DataFrame()

    for count,ticker in enumerate(tickers[:47]):
        df = pd.read_csv('stock_dfs/{}.csv'.format(ticker))
        df.set_index('date',inplace=True)

        df.rename(columns = {'5. adjusted close':ticker}, inplace=True)
        df.drop(['1. open','2. high','3. low','4. close', '6. volume', '7. dividend amount', '8. split coefficient'],1,inplace=True)

        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df,how='outer')

        if count % 10 ==0:
            print(count)
    print(main_df.head)
    main_df.to_csv('sp500_joined_closes.csv')

# compile_data()

def visualize_data():
    df = pd.read_csv('sp500_joined_closes.csv')
    # df['MMM'].plot()
    # plt.show()
    df_corr = df.corr()

    print(df_corr.head())

    data = df_corr.values
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    heatmap = ax.pcolor(data,cmap='RdYlGn',vmin=-1,vmax=1)
    fig.colorbar(heatmap)
    ax.set_xticks(np.arange(data.shape[0])+0.5,minor=False)
    ax.set_yticks(np.arange(data.shape[1])+0.5,minor=False)
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    column_labels = df_corr.columns
    row_labels = df_corr.index

    ax.set_xticklabels(column_labels)
    ax.set_yticklabels(row_labels)
    plt.xticks(rotation=90)
    # heatmap.setclim(-1,1)
#     would not use the above line for covariance
    plt.tight_layout()
    plt.show()



visualize_data()

