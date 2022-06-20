from ipywidgets.widgets.widget_description import DescriptionStyle
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
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

# if _COLAB:
#     from google.colab import data_table
    
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

class InvestmentAPI:
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

    def get_options(self, symbol):
        today = datetime.datetime.today()
        df = self._get_ally_option(symbol, 'xdate-gte:'+today.strftime('%Y%m%d'))
        
        # select columns from raw data
        columns = ['xdate','strikeprice','bid','ask','idelta','imp_Volatility','itheta','igamma','rootsymbol','put_call']
        df = df.loc[:, columns]
        df.rename(columns={'strikeprice':'strike', 'rootsymbol':'symbol','xdate':'date',
                           'imp_Volatility':'iv','idelta':'delta','igamma':'gamma','itheta':'theta'}, inplace=True)
        df['date'] = df.date.dt.date
        df['day'] = (df.date-pd.to_datetime(today.strftime('%Y-%m-%d')).date()).dt.days+1
        df['yr%'] = round((df['bid']-0.65/100)/df.strike/df.day*365*100,2)
        return df

    def get_quotes(self, symbols):
        df = self._get_ally_quote(symbols)
        df.set_index('date', inplace=True)
        df.index.name = None
        df['time'] = df.datetime.dt.strftime('%I:%M %p')
        
        # select columns from raw data
        columns = ['symbol','bid','ask','time','cl','lo','opn','hi','chg','wk52hi','wk52lo']
        df = df.loc[:, columns].sort_values(by=['symbol'])        
        return df

    # s[strike price lower/upper bnd]
    # d[expiration date range]
    def get_option_chain(self, symbol, s=[None, None], d=[None,None]):
        df = self.get_options(symbol)
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

    # list = ['aeo-p-5', 'aeo-c-10-10']
    def get_selected_options(self, list):
        df = pd.DataFrame([l.split('-') for l in list], columns=['symbol','put_call','strike','basis'])
        df.put_call = np.where(df.put_call.str.lower()=='p','put','call')
        df[['strike','basis']] = df[['strike','basis']].apply(pd.to_numeric, errors='coerce')

        df = pd.concat([self.get_selected_option(s, df[df.symbol==s]) for s in df.symbol.values], ignore_index=True)

        columns = ['date','put_call','symbol','strike','bid','ask','day','yr%','delta','iv','theta','gamma']  
        df = df[columns]
        df = df.sort_values(by=['date'])
        return df

    def get_selected_option(self, symbol, param):
        df = self.get_options(symbol)
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

class tab_option_chain:
    
    def on_change(self, change):
        if change['type'] == 'change' and change['name'] == 'value':
            if 'Stock' in change['owner'].description:
                self.df_options = self.work.get_option_chain(self.widget_symbol.value)
                self.df_quotes  = self.work.get_quotes(self.widget_symbol.value)                
                bid = self.df_quotes['bid'].values[0]
                strikes = np.unique(self.df_options.strike.values)
                idx = (np.abs(strikes - bid)).argmin()
                self.widget_lo_strike_price.value=strikes[max(0,idx-5)]
                self.widget_hi_strike_price.value=strikes[min(idx+5, len(strikes))]
                self.widget_cost_basis.value=strikes[idx]
                self.widget_expiration_dates.options=np.unique(self.df_options.date)
                self.widget_expiration_dates.index=[0,1,2,3]
                self.page = 0
            self.show()

    def on_click(self, change):
        self.df_options = self.work.get_option_chain(self.widget_symbol.value)
        self.df_quotes  = self.work.get_quotes(self.widget_symbol.value)
        self.show()
        
    def on_flip_left(self, change):
        self.page = max(0,self.page - 10)
        self.show()
        
    def on_flip_right(self, change):
        self.page = self.page + 10
        self.show()
        
    def show(self):
        lo_strike_price = float(self.widget_lo_strike_price.value)
        hi_strike_price = float(self.widget_hi_strike_price.value)
        cost_basis = float(self.widget_cost_basis.value)
        if hi_strike_price<lo_strike_price:
            self.widget_hi_strike_price.value=lo_strike_price
        expiration_dates = self.widget_expiration_dates.value

        # for call - use cost basis to calculate yield
        self.df_options['c_yr%'] = \
            round((self.df_options['c_bid']-0.65/100)/cost_basis/self.df_options.day*365*100,2)
            
        df = self.df_options.loc[\
                (self.df_options.strike>=lo_strike_price)&\
                (self.df_options.strike<=hi_strike_price)&\
                (self.df_options.date.isin(np.array([pd.to_datetime(d).date() for d in expiration_dates]))),:]
        
        sort_columns = []
        sort_order = []
        for i in range(3):
            if len(self.widget_sort[i].value.strip())>0:
                sort_columns.append(self.widget_sort[i].value)
                sort_order.append(not self.widget_sort_checkbox[i].value)
            else:
                break
        if len(sort_columns)>0:
            df = df.sort_values(by=sort_columns, ascending=sort_order)
        
        if self.page>len(df)-10:
            self.page=max(0,len(df)-10)
        self.widget_rows.value = 'Total Rows: '+str(len(df))
        df = df.reset_index(drop=True)
        
        self.output.quote.clear_output()
        with self.output.quote:
            columns = ['bid', 'ask', 'time','cl','lo','opn','hi','chg']
            display(self.df_quotes[columns])
            
        self.output.option.clear_output()
        with self.output.option:
            # display(data_table.DataTable(df,include_index=False,num_rows_per_page=20))
            display(df.iloc[self.page:self.page+10,:])
            # display(df)

    def __init__(self, work, watch_list, output, flag=None):
        self.work = work
        default_symbol = watch_list.symbols[0]
        self.page = 0
        
        self.df_options = self.work.get_option_chain(default_symbol)
        self.df_quotes  = self.work.get_quotes(default_symbol)
        self.output = output

        self.widget_symbol = widgets.Dropdown(
            options=watch_list.symbols,
            value=default_symbol,
            description='Stock:',
            width = 100,
            layout=Layout(width='180px'),
        )
        bid = self.df_quotes['bid'].values[0]
        strikes = np.unique(self.df_options.strike.values)
        idx = (np.abs(strikes - bid)).argmin()
        self.widget_lo_strike_price = widgets.FloatText(
            value=strikes[max(0,idx-3)],
            description='Strike from:',
            disabled=False,
            continuous_update=False,
            layout=Layout(width='180px'),
        )
        self.widget_hi_strike_price = widgets.FloatText(
            value=strikes[min(idx+4, len(strikes))],
            description='to:',
            disabled=False,
            continuous_update=False,
            layout=Layout(width='180px'),
        )
        self.widget_cost_basis = widgets.FloatText(
            value=strikes[idx],
            description='Cost Basis:',
            disabled=False,
            continuous_update=False,
            layout=Layout(width='180px'),
        )
        self.widget_expiration_dates = widgets.SelectMultiple(
            options=np.unique(self.df_options.date),
            index=[0,1,2,3],
            rows=8,
            description='Expire:',
            disabled=False,
            continuous_update=False,
            layout=Layout(width='220px', margin='0px 0 0 -20px'),
        )
        
        columns = [' ', 'date', 'strike', 'c_yr%', 'p_yr%']
        self.widget_sort = [
            widgets.Dropdown(
                options=columns,
                value=columns[0],
                description='',
                layout=Layout(width='100px'),
            ) for _ in range(3)
        ]
        self.widget_sort_checkbox = [
            widgets.Checkbox(
                value=False,
                description='Descending',
                disabled=False,
                # disabled=False,
                layout=Layout(width='max-content', margin='0px 0 0 -85px'),
            ) for _ in range(3)
        ]
        self.widget_refresh_button = widgets.Button(
            description='Refresh',
            disabled=False,
            button_style='success', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Refresh Quote',
            icon='refresh', # (FontAwesome names without the `fa-` prefix)
            layout=Layout(width='90px', margin='0 0px 0 20px'),
            # style={'description_width': '10px', },
        )        
        self.widget_left_flip_button = widgets.Button(
            description='',
            disabled=False,
            button_style='warning', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Left',
            icon='caret-left', # (FontAwesome names without the `fa-` prefix)
            layout=Layout(height='20px', width='20px', margin='10px 5px 0 20px'),
            # style={'description_width': '10px', },
        )
        self.widget_right_flip_button = widgets.Button(
            description='',
            disabled=False,
            button_style='warning', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Right',
            icon='caret-right', # (FontAwesome names without the `fa-` prefix)
            layout=Layout(height='20px', width='20px', margin='10px 0px 0 0px'),
            # style={'description_width': '10px', },
        )
        self.widget_rows = widgets.Label(
            value='Total Row',
            layout=Layout(margin='0px 0px 0 20px')
        )        
        self.widget_symbol.observe(self.on_change)
        self.widget_lo_strike_price.observe(self.on_change)
        self.widget_hi_strike_price.observe(self.on_change)
        self.widget_expiration_dates.observe(self.on_change)
        self.widget_cost_basis.observe(self.on_change)
        for i in range(3):
            self.widget_sort[i].observe(self.on_change)
            self.widget_sort_checkbox[i].observe(self.on_change)
        self.widget_refresh_button.on_click(self.on_click)
        self.widget_left_flip_button.on_click(self.on_flip_left)
        self.widget_right_flip_button.on_click(self.on_flip_right)
       
        self.board = widgets.HBox([
                            widgets.VBox([widgets.HBox([widgets.VBox([self.widget_symbol,
                                                                      self.widget_cost_basis]),
                                                       widgets.VBox([self.widget_lo_strike_price,
                                                                     self.widget_hi_strike_price])
                                                      ]),
                                          widgets.HBox([widgets.Label(value="Columns:",
                                                                      layout=Layout(margin='0px 7px 0 27px'))] + 
                                                       [widgets.VBox([self.widget_sort[i], self.widget_sort_checkbox[i]])
                                                        for i in range(3)
                                                      ])
                                         ]),
                            self.widget_expiration_dates,
                            widgets.VBox([self.widget_refresh_button, 
                                          widgets.HBox([self.widget_left_flip_button, 
                                                        self.widget_right_flip_button]),
                                          self.widget_rows
                                         ])
        ])
        self.show()
        
class tab_option_roll:
    def on_change(self, change):
        if change['type'] == 'change' and change['name'] == 'value':
            if 'Position:' in change['owner'].description or 'Date:' in change['owner'].description:
                position = self.widget_position.value + ' ' + self.widget_date.value.strftime("%m/%d/%Y")
                self.df_options = self.get_option_roll(position)
                self.df_quotes  = self.work.get_quotes(position.split(' ')[0].strip())
                
                strike = float(position.split(' ')[1].strip())
                strikes = np.unique(self.df_options.strike_r.values)
                idx = (np.abs(strikes - strike)).argmin()
                lo_strike_price = strikes[0]
                hi_strike_price = strikes[-1]
                if position.split(' ')[2].strip().lower()=='p':
                    self.widget_hi_strike_price.value = strikes[idx]
                else:
                    self.widget_lo_strike_price.value = strikes[idx]
                    
                expiration_dates = np.unique(self.df_options.date_r.values)
                self.widget_date_from.index = min(expiration_dates.tolist().index(self.widget_date.value)+1,len(expiration_dates)-1)
                self.widget_date_to.index = len(expiration_dates)-1

            lo_strike_price = float(self.widget_lo_strike_price.value)
            hi_strike_price = float(self.widget_hi_strike_price.value)                
            if 'Strike from:' in change['owner'].description and lo_strike_price>=hi_strike_price:
                  self.widget_hi_strike_price.index=min(self.widget_hi_strike_price.options.index(self.widget_lo_strike_price.value)+1,
                                                      len(self.widget_hi_strike_price.options)-1)
            if 'to:' in change['owner'].description and '160px' in change['owner'].layout.width and lo_strike_price>=hi_strike_price:
                self.widget_lo_strike_price.index=max(self.widget_lo_strike_price.options.index(self.widget_hi_strike_price.value)-1,
                                                      0)
            self.page = 0                
            self.show()

    def on_click(self, change):
        position = self.widget_position.value + ' ' + self.widget_date.value.strftime("%m/%d/%Y")
        self.df_options = self.get_option_roll(position)
        self.df_quotes = self.work.get_quotes(position.split(' ')[0].strip())
        self.show()
        
    def on_flip_left(self, change):
        self.page = max(0,self.page - 10)
        self.show()
        
    def on_flip_right(self, change):
        self.page = self.page + 10
        self.show()        

    def show(self):
        date_from = pd.to_datetime(self.widget_date_from.value).date()
        date_to   = pd.to_datetime(self.widget_date_to.value).date()
        lo_strike_price = float(self.widget_lo_strike_price.value)
        hi_strike_price = float(self.widget_hi_strike_price.value)         
        lo_mid_price = float(self.widget_lo_mid_price.value)
        df = self.df_options[(self.df_options.date_r>=date_from)&(self.df_options.date_r<=date_to)&\
                             (self.df_options.strike_r>=lo_strike_price)&\
                             (self.df_options.strike_r<=hi_strike_price)&\
                             (self.df_options.mid>=lo_mid_price)
                            ].rename(columns={'date_r': 'date', 'strike_r': 'strike'})
        columns = ['bid','mid','ask','yr%','date','strike']
        df = df[columns]                                     
        if self.page>len(df)-10:
            self.page=max(0,len(df)-10)
        self.widget_rows.value = 'Total Rows: '+str(len(df))
        
        sort_columns = []
        sort_order = []
        for i in range(3):
            if len(self.widget_sort[i].value.strip())>0:
                sort_columns.append(self.widget_sort[i].value)
                sort_order.append(not self.widget_sort_checkbox[i].value)
            else:
                break
        if len(sort_columns)>0:
            df = df.sort_values(by=sort_columns, ascending=sort_order) 
        df = df.reset_index(drop=True)  
        
        self.output.quote.clear_output()
        with self.output.quote:
            columns = ['bid', 'ask', 'time','cl','lo','opn','hi','chg']
            display(self.df_quotes[columns])

        self.output.option.clear_output()
        with self.output.option:            
            # display(HTML(df.to_html(index=False)))
            display(df.iloc[self.page:self.page+10,:])

    # pos = 'AEO 13.00 P 06/17/2022'
    # mid = lower bound of middle price
    # roll to date >= d
    def get_option_roll(self, pos):
        today = datetime.datetime.today()
        symbol = pos.split(' ')[0].strip()
        df = self.work.get_options(symbol)
        
        strike = float(pos.split(' ')[1].strip())
        if pos.split(' ')[2].strip().lower()=='p':
            typ = 'put'
        else:
            typ = 'call'
        date = pd.to_datetime(pos.split(' ')[3].strip()).date()
        
        df = df[df.put_call==typ]
        df = df[(df.date==date)&(df.put_call==typ)&(df.strike==strike)].merge(df, how='cross',suffixes=('_c', '_r'))
        df['bid'] = round(df['bid_r'] - df['ask_c'],2)
        df['ask'] = round(df['ask_r'] - df['bid_c'],2)
        df['mid'] = round((df['ask']+df['bid'])/2,3)
        df['day'] = (df.date_r - df.date_c).dt.days+1
        df['yr%'] = round((df['mid']-1.3/100)/df['strike_r']/df.day*365*100,2)
        df = df.sort_values(by=['date_r','strike_r'])
        return df
                
            
    def __init__(self, work, watch_list, output, flag=None):
        
        self.work = work
        self.page = 0
        position = watch_list.option_positions[0]
        symbol = position.split(' ')[0].strip()
        
        df_options = self.work.get_options(symbol)
        expiration_dates = np.unique(df_options.date.values)
        self.widget_position = widgets.Dropdown(
            options=watch_list.option_positions,
            value=watch_list.option_positions[0],
            description='Position:',
            layout=Layout(width='210px'),
        )
        self.widget_date = widgets.Dropdown(
            options=expiration_dates,
            value=expiration_dates[0],
            description='Date:',
            layout=Layout(width='210px'),
        )
        self.widget_date_from = widgets.Dropdown(
            options=expiration_dates,
            index=1,
            description='Date from:',
            width = 100,
            layout=Layout(width='190px'),
        )
        self.widget_date_to = widgets.Dropdown(
            options=expiration_dates,
            index=len(expiration_dates)-1,
            description='to:',
            width = 100,
            layout=Layout(width='190px'),
        )
        
        # if the position is put
        # if the position is call
        strike = float(position.split(' ')[1].strip())
        strikes = np.unique(df_options.strike.values)
        idx = (np.abs(strikes - strike)).argmin()
        lo_strike_price = strikes[0]
        hi_strike_price = strikes[-1]
        if position.split(' ')[2].strip().lower()=='p':
            hi_strike_price = strikes[idx]
        else:
            lo_strike_price = strikes[idx]
        self.widget_lo_strike_price = widgets.Dropdown(
            options=strikes,
            value=lo_strike_price,
            description='Strike from:',
            disabled=False,
            continuous_update=False,
            layout=Layout(width='160px'),
        )
        self.widget_hi_strike_price = widgets.Dropdown(
            options=strikes,
            value=hi_strike_price,
            description='to:',
            disabled=False,
            continuous_update=False,
            layout=Layout(width='160px'),
        )
        self.widget_lo_mid_price = widgets.FloatText(
            value=0,
            description='Mid:',
            disabled=False,
            step=0.02,
            continuous_update=False,
            layout=Layout(width='140px', margin='0px 0px 0 -30px'),
        )
        columns = [' ', 'date', 'strike', 'yr%', 'mid']
        self.widget_sort = [
            widgets.Dropdown(
                options=columns,
                value=columns[0],
                description='',
                layout=Layout(width='100px'),
            ) for _ in range(3)
        ]
        self.widget_sort_checkbox = [
            widgets.Checkbox(
                value=False,
                description='Descending',
                disabled=False,
                # disabled=False,
                layout=Layout(width='max-content', margin='0px 0 0 -85px'),
            ) for _ in range(3)
        ]
        self.widget_refresh_button = widgets.Button(
            description='Refresh',
            disabled=False,
            button_style='success', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Refresh Quote',
            icon='refresh', # (FontAwesome names without the `fa-` prefix)
            layout=Layout(width='90px', margin='0 0px 0 20px'),
            # style={'description_width': '10px'},
        )
        self.widget_left_flip_button = widgets.Button(
            description='',
            disabled=False,
            button_style='warning', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Left',
            icon='caret-left', # (FontAwesome names without the `fa-` prefix)
            layout=Layout(height='20px', width='20px', margin='10px 5px 0 20px'),
            # style={'description_width': '10px', },
        )
        self.widget_right_flip_button = widgets.Button(
            description='',
            disabled=False,
            button_style='warning', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Right',
            icon='caret-right', # (FontAwesome names without the `fa-` prefix)
            layout=Layout(height='20px', width='20px', margin='10px 0px 0 0px'),
            # style={'description_width': '10px', },
        )
        self.widget_rows = widgets.Label(
            value='',
            layout=Layout(margin='0px 0px 0 20px')
        )
        position = self.widget_position.value + ' ' + self.widget_date.value.strftime("%m/%d/%Y")
        self.df_options = self.get_option_roll(position)
        self.df_quotes  = self.work.get_quotes(symbol)        
        self.output = output

        self.widget_position.observe(self.on_change)
        self.widget_date.observe(self.on_change)
        self.widget_date_from.observe(self.on_change)
        self.widget_date_to.observe(self.on_change)
        self.widget_lo_strike_price.observe(self.on_change)
        self.widget_hi_strike_price.observe(self.on_change)
        self.widget_lo_mid_price.observe(self.on_change)
        for i in range(3):
            self.widget_sort[i].observe(self.on_change)
            self.widget_sort_checkbox[i].observe(self.on_change)        
        self.widget_refresh_button.on_click(self.on_click)
        self.widget_left_flip_button.on_click(self.on_flip_left)
        self.widget_right_flip_button.on_click(self.on_flip_right)        
        self.board = widgets.HBox([widgets.VBox([widgets.HBox([widgets.VBox([self.widget_position, self.widget_date]),
                                                               widgets.VBox([self.widget_date_from,self.widget_date_to]),
                                                               widgets.VBox([self.widget_lo_strike_price,self.widget_hi_strike_price])]),
                                                 widgets.HBox([widgets.Label(value="Columns:   ",
                                                                             layout=Layout(margin='0px 8px 0 26px'))] + 
                                                              [widgets.VBox([self.widget_sort[i], self.widget_sort_checkbox[i]])
                                                               for i in range(3)
                                                              ])
                                                ]),
                                   self.widget_lo_mid_price,
                                   widgets.VBox([self.widget_refresh_button, 
                                                 widgets.HBox([self.widget_left_flip_button, 
                                                               self.widget_right_flip_button]),
                                                 self.widget_rows,
                                                ])
                                  ])
        # self.show()
        
class tab_view:

    def on_update_selected_index(self, change):
        index = change['new']
        if self.tab.get_title(index).lower()=='option chain':
            self.tab_chain.show()        
        if self.tab.get_title(index).lower()=='roll':
            self.tab_roll.show()
            
    def __init__(self, credential, watch_list):
        
        class output:
            quote  = widgets.Output()
            option = widgets.Output()        
        
        self.work = InvestmentAPI(credential)
        self.tab_chain = tab_option_chain(self.work, watch_list, output) 
        self.tab_roll  = tab_option_roll(self.work, watch_list, output)
        
        output_selected_options = widgets.Output()
        with output_selected_options:
            print('here')

        self.tab = widgets.Tab([self.tab_chain.board, self.tab_roll.board,
                                output_selected_options])
        self.tab.set_title(0, 'Option Chain')
        self.tab.set_title(1, 'Roll')
        self.tab.set_title(2, 'Selected Options')   
        
        self.tab.observe(self.on_update_selected_index, names='selected_index')
        
        display(output.quote)
        display(self.tab)
        display(output.option)
        
