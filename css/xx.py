_COLAB = True

from ipywidgets.widgets.widget_description import DescriptionStyle
import ipywidgets as widgets
from IPython.display import display
from IPython.display import clear_output
from ipywidgets import Layout

import json
import datetime
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly

from requests_oauthlib import OAuth1
from pandas.tseries.holiday import USFederalHolidayCalendar
from plotly.subplots import make_subplots
from dateutil.relativedelta import relativedelta
from matplotlib.ticker import PercentFormatter

if _COLAB:
    from google.colab import data_table
    
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

class InvestAPI:
    def __init__(self, credential):
        self.ally_oauth_secret  = credential[0]
        self.ally_oauth_token   = credential[1]
        self.ally_client_key    = credential[2]
        self.ally_client_secret = credential[3]
        self.ally_url    = 'https://devapi.invest.ally.com/v1/'
        self.ally_valid_auth_dt = datetime.timedelta(seconds=10)
        self.ally_auth_time     = None
        self.ally_auth          = None

        self._dat_ = None #pd.DataFrame(symbols, columns=['symbol'])
        self.dat_options = None
        self.dat_quotes  = None
        
        self.alphavantage_url   = 'https://www.alphavantage.co/query?'
        self.alphavantage_key   = credential[4]
        
    def _get_ally_data(self, url):
        now = datetime.datetime.now()
        if self.ally_auth == None or self.ally_auth_time + self.ally_valid_auth_dt < now:
            self.ally_auth_time = now
            self.ally_auth = OAuth1(self.ally_client_key, self.ally_client_secret, self.ally_oauth_token,
                                    self.ally_oauth_secret, signature_type='auth_header')    
        return requests.get(url, auth=self.ally_auth)

    def _get_ally_quote(self, symbols):
        if not isinstance(symbols, str): # list
            symbols = ",".join(symbols)
        url = self.ally_url+"market/ext/quotes.json?symbols={symbols}".format(symbols=symbols)
        res = self._get_ally_data(url)
        res = res.json()['response']['quotes']['quote']
        if isinstance(res,dict):
            df = pd.DataFrame(res, index=[0])
        else:
            df = pd.DataFrame(res)
            
        numeric_cols  = ['adp_100', 'adp_200', 'adp_50', 'adv_21', 'adv_30', 'adv_90', 'ask',
                         'asksz', 'basis', 'beta', 'bid', 'bidsz',
                         'chg', 'cl', 'contract_size', 'cusip',
                         'days_to_expiration', 'div', 'divexdate', 
                         'dollar_value', 'eps', 'hi', 'iad',
                         'idelta', 'igamma', 'imp_volatility', 'incr_vl', 'irho', 'issue_desc',
                         'itheta', 'ivega', 'last', 'lo', 'op_delivery', 'op_flag',
                         'op_style', 'op_subclass', 'openinterest', 'opn', 'opt_val', 'pchg',
                         'pcls', 'pe', 'phi', 'plo', 'popn', 'pr_adp_100',
                         'pr_adp_200', 'pr_adp_50', 'pr_openinterest', 'prbook',
                         'prchg', 'prem_mult', 'put_call', 'pvol', 'qcond',
                         'secclass', 'sesn', 'sho', 'strikeprice', 
                         'tr_num', 'tradetick', 'trend', 'under_cusip',
                         'vl', 'volatility12', 'vwap', 'wk52hi', 
                         'wk52lo', 'xdate', 'xday', 'xmonth', 'xyear', 'yield']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        datetime_cols = ['ask_time','bid_time','date','datetime','divpaydt','pr_date','wk52hidate','wk52lodate']
        df[datetime_cols] = df[datetime_cols].apply(pd.to_datetime, errors='coerce')            
            
        return df
    
    def _get_ally_timesales(self, symbol, query):
        url = self.ally_url+"market/timesales.json?symbols={symbol}&{query}"\
                  .format(symbol=symbol,query=query)
        res = self._get_ally_data(url)
        #res = res.json()['response']['quotes']['quote']
        return res
    
    def _get_ally_option(self, symbol, query):
        url = self.ally_url+"market/options/search.json?symbol={symbol}&query={query}"\
                .format(symbol=symbol,query=query)
        res = self._get_ally_data(url)
        res = res.json()['response']['quotes']['quote']
        df = pd.DataFrame(res)

        df['xdate'] = pd.to_datetime(df['xdate'])
        numeric_cols = ['strikeprice','bid','ask','idelta','igamma','imp_Volatility','itheta','irho']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

        return df
    
    def _get_ally_strikes(self, symbol):        
        url = self.ally_url+"market/options/strikes.json?symbol={symbol}".format(symbol=symbol)
        res = self._get_ally_data(url)
        strikes = np.array(res.json()['response']['prices']['price']).astype(float)
        return strikes

    def get_raw_options(self, symbol):
        today = datetime.datetime.today()
        df = self._get_ally_option(symbol, 'xdate-gte:'+today.strftime('%Y%m%d'))
        columns = ['xdate','strikeprice','bid','ask','idelta','imp_Volatility','itheta','igamma','rootsymbol','put_call']
        df = df.loc[:, columns]
        df.rename(columns={'strikeprice':'strike', 'rootsymbol':'symbol','xdate':'date',
                           'imp_Volatility':'iv','idelta':'delta','igamma':'gamma','itheta':'theta'}, inplace=True)
        df['date'] = df.date.dt.date
        df['day'] = (df.date-pd.to_datetime(today.strftime('%Y-%m-%d')).date()).dt.days+1
        df['yr%'] = round((df['bid']-0.65/100)/df.strike/df.day*365*100,2)
        return df

    def get_raw_quotes(self, symbols):
        df = self._get_ally_quote(symbols)
        df.set_index('date', inplace=True)
        df.index.name = None
        df['time'] = df.datetime.dt.strftime('%I:%M %p')
        columns = ['symbol','bid','ask','time','cl','lo','opn','hi','chg','wk52hi','wk52lo']
        df = df.loc[:, columns].sort_values(by=['symbol'])        
        return df

    # s[strike price lower/upper bnd]
    # d[expiration date range]
    def get_option_chain(self, symbol, s=[None, None], d=[None,None]):
        df = self.get_raw_options(symbol)
        index  = ['date','strike','symbol','day']
        values = ['bid','ask','yr%','delta','gamma','theta','iv']

        df = pd.pivot_table(df, values=values, index=index, columns=['put_call'])
        df.columns = df.columns.swaplevel(0, 1)
        df.sort_index(axis=1, level=0, inplace=True)
        df = df.reindex(values, axis=1, level=1)
        df.reset_index(level=index, inplace=True)
        df.columns = df.columns.to_flat_index()
        df.rename(columns=dict([[(x,y), 'c_'+y] for (x,y) in df.columns if x=='call'] + \
                             [[(x,y), x] for (x,y) in df.columns if x not in ['put', 'call']] + \
                             [[(x,y), 'p_'+y] for (x,y) in df.columns if x=='put']), inplace=True)
        columns = [x for x in reversed(df.columns) if x.startswith('c_')] + \
                  ['date', 'strike', 'symbol','day'] + \
                  [x for x in df.columns if x.startswith('p_')]
        df = df[columns]

        if d[0]==None:
            dl = df['date'].values[0]
        else:
            dl = pd.to_datetime(d[0]).date()
        if d[1]==None:
            dh = df['date'].values[-1]
        else:
            dh = pd.to_datetime(d[1]).date()        
        strikes = df.strike.values
        if s[0]==None:
            lo = min(strikes)
        else:
            lo = s[0]
        if s[1]==None:
            hi = max(strikes)
        else:
            hi = s[1]
        df = df[(df.strike>=lo)&(df.strike<=hi)&(df['date']>=dl)&(df['date']<=dh)]
        return df

    # pos = 'AEO 06/17/2022 13.00 P'
    # mid = lower bound of middle price
    # roll to date >= d
    def roll_option(self, pos, mid=None, d=[None,None], s=[None, None]):
        today = datetime.datetime.today()
        symbol = pos.split(' ')[0].strip()
        date = pd.to_datetime(pos.split(' ')[1].strip()).date()
        strike = float(pos.split(' ')[2].strip())
        if pos.split(' ')[3].strip().lower()=='p':
            typ = 'put'
        else:
            typ = 'call'
        df = self.get_raw_options(symbol)

        df = df[df.put_call==typ]
        df = df[(df.date==date)&(df.put_call==typ)&(df.strike==strike)].merge(df, how='cross',suffixes=('_c', '_r'))
        df['bid'] = round(df['bid_r'] - df['ask_c'],2)
        df['ask'] = round(df['ask_r'] - df['bid_c'],2)
        df['mid'] = round((df['ask']+df['bid'])/2,3)
        df['day'] = (df.date_r - df.date_c).dt.days+1
        df['yr%'] = round((df['mid']-1.3/100)/df['strike_r']/df.day*365*100,2)
        df.drop(columns=['symbol_r','put_call_r','bid_r','ask_r','day_r','yr%_r','yr%_c','bid_c','ask_c','day_c'], inplace=True)
        df.rename(columns=dict([[c,c.rsplit('_',1)[0]]for c in df.columns if c.endswith('_c')]), inplace=True)
        columns = ['bid','mid','ask','yr%','date_r','strike_r','date','strike','put_call']
        df = df[columns]
        if d[0]==None:
            dl = min(df['date_r'].values)
        else:
            dl = pd.to_datetime(d[0]).date()
        if d[1]==None:
            dh = max(df['date_r'].values)
        else:
            dh = pd.to_datetime(d[1]).date()
        if mid==None:
            mid = min(df.mid.values)
        strikes = df.strike_r.values
        if s[0]==None:
            lo = min(strikes)
        else:
            lo = s[0]
        if s[1]==None:
            hi = max(strikes)
        else:
            hi = s[1]            
            
        df = df[(df['date_r']>=dl)&(df['date_r']<=dh)&(df.mid>=mid)&(df.strike_r>=lo)&(df.strike_r<=hi)]
        df = df.sort_values(by=['date_r','strike_r'])
        return df

    # list = ['aeo-p-5', 'aeo-c-10-10']
    def get_options(self, list):
        df = pd.DataFrame([l.split('-') for l in list], columns=['symbol','put_call','strike','basis'])
        df.put_call = np.where(df.put_call.str.lower()=='p','put','call')
        df[['strike','basis']] = df[['strike','basis']].apply(pd.to_numeric, errors='coerce')

        df = pd.concat([self._get_options(s, df[df.symbol==s]) for s in df.symbol.values], ignore_index=True)

        columns = ['date','put_call','symbol','strike','bid','ask','day','yr%','delta','iv','theta','gamma']  
        df = df[columns]
        df = df.sort_values(by=['date'])
        return df

    def _get_options(self, symbol, param):
        df = self.get_raw_options(symbol)
        df = df[((df.put_call=='put') & (df.strike.isin(np.unique(param[param.put_call=='put'].strike.values)))) |
                ((df.put_call=='call') & (df.strike.isin(np.unique(param[param.put_call=='call'].strike.values))))]
        df = df.merge(param[['strike','basis','put_call']], how='left', on=['strike','put_call'])
        df['yr%'] = np.where((df.put_call=='put') | (df.basis.isna()),
                             round((df['bid']-0.65/100)/df.strike/df.day*365*100,2),
                             round((df['bid']-0.65/100)/df.basis/df.day*365*100,2))      
        # columns = ['date','put_call','symbol','strike','bid','ask','day','yr%','delta','iv','theta','gamma']  
        # df = df[columns]
        return df
    
    def _get_alphavantage_data(self, url):
        if 'This is a premium endpoint' in requests.get(url).text:
            print('This is premium endpoint. Access is denied.')
            return None
        url = url + '&apikey=' + self.alphavantage_key
        
        res = requests.get(url)
        return res.json()
        
    def _get_alphavantage_stock_daily(self, symbol, outputsize='full'):
        url = self.alphavantage_url + "function=TIME_SERIES_DAILY&symbol={symbol}&outputsize={outputsize}"\
                  .format(symbol=symbol, outputsize=outputsize)
        return self._get_alphavantage_data(url)

    def _get_alphavantage_stock_weekly(self, symbol):
        url = self.alphavantage_url + "function=TIME_SERIES_WEEKLY_ADJUSTED&symbol={symbol}"\
                  .format(symbol=symbol)
        return self._get_alphavantage_data(url)

    def _get_alphavantage_stock_monthly(self, symbol):
        url = self.alphavantage_url + "function=TIME_SERIES_MONTHLY_ADJUSTED&symbol={symbol}"\
                  .format(symbol=symbol)
        return self._get_alphavantage_data(url)
    
    def _get_alphavantage_stock(self, symbol, option):
        url = self.alphavantage_url + "symbol={symbol}&{option}"\
                  .format(symbol=symbol, option=option)
        
        res = self._get_alphavantage_data(url)
        #print(res['Note'])
        
        df = pd.DataFrame(list(res.values())[-1]).T
        df.columns  = [col[col.find('.')+1:].strip() for col in df.columns]
        df.reset_index(inplace=True)
        df.rename(columns={'index':'date'}, inplace=True)
        df['date'] = df['date'].apply(pd.to_datetime, errors='coerce')
        df.sort_values(by=['date'], inplace=True)
        df.reset_index(inplace=True, drop=True)
        
        numeric_cols = [col for col in df.columns 
                            if col in ['open','close','low','high','volume', 'adjusted close']]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

        return df

class option_chain:
    def on_change(self, change):
        if change['type'] == 'change' and change['name'] == 'value':
            if 'Stock' in change['owner'].description:
                self.refresh()            
            if 'Cost Basis' in change['owner'].description and self.cbck.value==False:
                None
            else:
                self.show()

    def refresh(self):
        self.df_options = self.work.get_option_chain(self.symbol.value)
        self.df_quotes = self.work.get_raw_quotes(self.symbol.value)

        bid = self.df_quotes['bid'].values[0]
        strikes = np.unique(self.df_options.strike.values)
        idx = (np.abs(strikes - bid)).argmin()
        self.lsp.value=strikes[max(0,idx-5)]
        self.hsp.value=strikes[min(idx+5, len(strikes))]
        self.cb.value=strikes[idx]
        self.cbck.value=False
        self.expiration.options=np.unique(self.df_options.date)
        self.expiration.index=[0,1,2,3]

        self.show()

    def on_click(self, change):
        self.refresh()

    def show(self):
        lo_p = float(self.lsp.value)
        hi_p = float(self.hsp.value)
        cb = float(self.cb.value)
        if hi_p<lo_p:
            self.hsp.value=lo_p
        dts = self.expiration.value
        self.df = self.df_options[(self.df_options.strike>=lo_p)&(self.df_options.strike<=hi_p)&\
                                  (self.df_options.date.isin(np.array([pd.to_datetime(d).date() for d in dts])))]
        if self.cbck.value:
            self.df['c_yr%'] = round(self.df['c_yr%']*self.df.strike/cb,2)
        else:
            self.df['c_yr%'] = round((self.df['c_bid']-0.65/100)/self.df.strike/self.df.day*365*100,2)

        clear_output()
        columns = ['bid', 'ask', 'time','cl','lo','opn','hi','chg']
        display(self.df_quotes[columns])
        display(self.board)
        
        if _COLAB:
            display(data_table.DataTable(self.df,include_index=False,num_rows_per_page=20))
        else:
            display(self.df)

    def __init__(self, watch_list, credential):
        self.work = InvestAPI(credential)
        self.df_options = self.work.get_option_chain('gps')
        self.df_quotes = self.work.get_raw_quotes('gps')
        self.df = None

        self.symbol = widgets.Dropdown(
            options=watch_list['symbols'],
            value='GPS',
            description='Stock:',
            width = 100,
            layout=Layout(width='180px'),
        )
        self.button = widgets.Button(
            description='Refresh',
            disabled=False,
            button_style='success', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Refresh Quote',
            icon='check', # (FontAwesome names without the `fa-` prefix)
            layout=Layout(width='100px'),
            style={'description_width': '10px'},
        )

        bid = self.df_quotes['bid'].values[0]
        strikes = np.unique(self.df_options.strike.values)
        idx = (np.abs(strikes - bid)).argmin()
        self.lsp = widgets.FloatText(
            value=strikes[max(0,idx-5)],
            description='Strike from:',
            disabled=False,
            continuous_update=False,
            layout=Layout(width='180px'),
        )
        self.hsp = widgets.FloatText(
            value=strikes[min(idx+5, len(strikes))],
            description='to:',
            disabled=False,
            continuous_update=False,
            layout=Layout(width='180px'),
        )
        self.cb = widgets.FloatText(
            value=strikes[idx],
            description='Cost Basis:',
            disabled=False,
            continuous_update=False,
            layout=Layout(width='180px'),
        )
        self.cbck = widgets.Checkbox(
            value=False,
            description='Apply',
            disabled=False,
            layout=Layout(width='180px', position='left',display='flex'),
        )
        self.expiration = widgets.SelectMultiple(
            options=np.unique(self.df_options.date),
            index=[0,1,2,3],
            rows=8,
            description='Expire:',
            disabled=False,
            continuous_update=False,
            layout=Layout(width='220px'),
        )
        self.symbol.observe(self.on_change)
        self.lsp.observe(self.on_change)
        self.hsp.observe(self.on_change)
        self.expiration.observe(self.on_change)
        self.cb.observe(self.on_change)
        self.cbck.observe(self.on_change)
        self.button.on_click(self.on_click)
        self.board = widgets.HBox([widgets.VBox([self.symbol,self.cb,self.cbck]),
                                   widgets.VBox([self.lsp,self.hsp]),
                                   self.expiration,self.button])
        self.show()
        
        
class option_roll:
    def on_change(self, change):
        if change['type'] == 'change' and change['name'] == 'value':
            if 'Position' in change['owner'].description:
                pos = self.pos.value
                self.df_options = self.work.roll_option(pos)
                self.df_quotes = self.work.get_raw_quotes(pos.split(' ')[0].strip())

            self.show()

    def on_click(self, change):
        pos = self.pos.value
        self.df_options = self.work.roll_option(pos)
        self.df_quotes = self.work.get_raw_quotes(pos.split(' ')[0].strip())
        
        self.show()

    def show(self):
        dl = pd.to_datetime(self.datel.value).date()
        dh = pd.to_datetime(self.dateh.value).date()
        mid = float(self.mid.value)
        clear_output()
        # columns = ['bid', 'ask', 'ask_time','cl','lo','opn','hi','chg']
        # display(quote[columns])
        display(self.board)
        df = self.df_options[(self.df_options['date_r']>=dl)&(self.df_options['date_r']<=dh)&
                             (self.df_options.mid>=mid)
                            ]
        if self.check.value:
            df = df[(df.strike_r<=df.strike)&(df.put_call=='put')|
                    (df.strike_r>=df.strike)&(df.put_call=='call')]
        df = df.sort_values(by=['date_r','strike_r'])
        if _COLAB:
            display(data_table.DataTable(df,include_index=False,num_rows_per_page=20))
        else:
            display(df) 

    def __init__(self, watch_list, credential):
        self.work = InvestAPI(credential)
        pos = watch_list['positions'][0]
        self.df_options = self.work.roll_option(pos)
        self.df_quotes = self.work.get_raw_quotes(pos.split(' ')[0].strip())  

        self.pos = widgets.Dropdown(
            options=watch_list['positions'],
            value=watch_list['positions'][0],
            description='Position:',
            width = 100,
            layout=Layout(width='280px'),
        )
        self.datel = widgets.Dropdown(
            options=np.unique(self.df_options.date_r.values),
            index=1,
            description='Date from:',
            width = 100,
            layout=Layout(width='200px'),
        )
        self.dateh = widgets.Dropdown(
            options=np.unique(self.df_options.date_r.values),
            index=len(np.unique(self.df_options.date_r.values))-1,
            description='to:',
            width = 100,
            layout=Layout(width='200px'),
        )
        self.mid = widgets.FloatText(
            value=0,
            description='Mid:',
            disabled=False,
            step=0.02,
            continuous_update=False,
            layout=Layout(width='180px'),
        )
        self.check = widgets.Checkbox(
            value=True,
            description='Compact',
            disabled=False,
            layout=Layout(width='180px', position='left',display='flex'),
        )        
        self.refresh = widgets.Button(
            description='Refresh',
            disabled=False,
            button_style='success', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Refresh Quote',
            icon='check', # (FontAwesome names without the `fa-` prefix)
            layout=Layout(width='100px'),
            style={'description_width': '10px'},
        )

        self.pos.observe(self.on_change)
        self.datel.observe(self.on_change)
        self.datel.observe(self.on_change)
        self.mid.observe(self.on_change)
        self.check.observe(self.on_change)
        self.refresh.on_click(self.on_click)
        self.board = widgets.HBox([self.pos,widgets.VBox([self.datel,self.dateh]),widgets.VBox([self.mid,self.check]),self.refresh])

        self.show()
