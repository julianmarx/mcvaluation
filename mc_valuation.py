import pandas as pd
from yahoo_fin import stock_info as si
import streamlit as st
import numpy as np
from matplotlib import pyplot as plt

def comma_format(number):
    if not pd.isna(number) and number != 0:
        return '{:,.0f}'.format(number)

def percentage_format(number):
    if not pd.isna(number) and number != 0:
        return '{:.1%}'.format(number) 

def calculate_value_distribution(parameter_dict_1, parameter_dict_2, parameter_dict_distribution):
    parameter_list = []
    parameter_list.append(parameter_dict_1['latest revenue'])
    for i in parameter_dict_2:
        if parameter_dict_distribution[i] == 'normal':
            parameter_list.append((np.random.normal(parameter_dict_1[i], parameter_dict_2[i]))/100)
        if parameter_dict_distribution[i] == 'triangular':
            lower_bound = parameter_dict_1[i]
            mode = parameter_dict_2[i]
            parameter_list.append((np.random.triangular(lower_bound, mode, 2*mode-lower_bound))/100)
        if parameter_dict_distribution[i] == 'uniform':
            parameter_list.append((np.random.uniform(parameter_dict_1[i], parameter_dict_2[i]))/100)
    parameter_list.append(parameter_dict_1['net debt'])
    return parameter_list

class Company:

    def __init__(self, ticker):
        self.income_statement = si.get_income_statement(ticker)
        self.balance_sheet = si.get_balance_sheet(ticker)
        self.cash_flow_statement = si.get_cash_flow(ticker)
        self.inputs = self.get_inputs_df()

    def get_inputs_df(self):
        income_statement_list = ['totalRevenue', 'ebit', 
        'incomeBeforeTax', 'incomeTaxExpense'
        ]
        balance_sheet_list = ['totalCurrentAssets', 'cash',
        'totalCurrentLiabilities', 'shortLongTermDebt',
        'longTermDebt'
        ]
        balance_sheet_list_truncated = ['totalCurrentAssets', 'cash',
        'totalCurrentLiabilities', 'longTermDebt'
        ]
        balance_sheet_list_no_debt = ['totalCurrentAssets', 'cash',
        'totalCurrentLiabilities'
        ]

        cash_flow_statement_list = ['depreciation', 
        'capitalExpenditures'
        ]

        income_statement_df = self.income_statement.loc[income_statement_list]
        try:
            balance_sheet_df = self.balance_sheet.loc[balance_sheet_list]
        except KeyError:
            try:
                balance_sheet_df = self.balance_sheet.loc[balance_sheet_list_truncated]
            except KeyError:
                balance_sheet_df = self.balance_sheet.loc[balance_sheet_list_no_debt]
        cash_flow_statement_df = self.cash_flow_statement.loc[cash_flow_statement_list]

        df = income_statement_df.append(balance_sheet_df)
        df = df.append(cash_flow_statement_df)        

        columns_ts = df.columns
        columns_str = [str(i)[:10] for i in columns_ts]
        columns_dict = {}
        for i,f in zip(columns_ts, columns_str):
            columns_dict[i] = f
        df.rename(columns_dict, axis = 'columns', inplace = True)

        columns_str.reverse()
        df = df[columns_str]

        prior_revenue_list = [None]
        for i in range(len(df.loc['totalRevenue'])):
            if i != 0 and i != len(df.loc['totalRevenue']):
                prior_revenue_list.append(df.loc['totalRevenue'][i-1])

        df.loc['priorRevenue'] = prior_revenue_list
        df.loc['revenueGrowth'] = (df.loc['totalRevenue'] - df.loc['priorRevenue']) / df.loc['priorRevenue']
        df.loc['ebitMargin'] = df.loc['ebit']/df.loc['totalRevenue'] 
        df.loc['taxRate'] = df.loc['incomeTaxExpense']/df.loc['incomeBeforeTax'] 
        df.loc['netCapexOverSales'] = (- df.loc['capitalExpenditures'] - df.loc['depreciation']) / df.loc['totalRevenue']
        try:
            df.loc['nwc'] = (df.loc['totalCurrentAssets'] - df.loc['cash']) - (df.loc['totalCurrentLiabilities'] - df.loc['shortLongTermDebt'])
        except KeyError:
            df.loc['nwc'] = (df.loc['totalCurrentAssets'] - df.loc['cash']) - (df.loc['totalCurrentLiabilities'])
        df.loc['nwcOverSales'] = df.loc['nwc']/df.loc['totalRevenue']
        try:
            df.loc['netDebt'] = df.loc['shortLongTermDebt'] + df.loc['longTermDebt'] - df.loc['cash']
        except KeyError:
            try:
                df.loc['netDebt'] = df.loc['longTermDebt'] - df.loc['cash']
            except KeyError:
                df.loc['netDebt'] = - df.loc['cash']
        df = df[12:len(df)].drop('nwc')
        df['Historical average'] = [df.iloc[i].mean() for i in range(len(df))]
        return df

    def get_free_cash_flow_forecast(self, parameter_list):
        df = pd.DataFrame(columns = [1, 2, 3, 4, 5])
        revenue_list = []
        for i in range(5):
            revenue_list.append(parameter_list[0] * (1 + parameter_list[1]) ** (i+1))
        df.loc['Revenues'] = revenue_list
        ebit_list = [i * parameter_list[2] for i in df.loc['Revenues']]
        df.loc['EBIT'] = ebit_list
        tax_list = [i * parameter_list[3] for i in df.loc['EBIT']]
        df.loc['Taxes'] = tax_list
        nopat_list = df.loc['EBIT'] - df.loc['Taxes']
        df.loc['NOPAT'] = nopat_list
        net_capex_list = [i * parameter_list[4] for i in df.loc['Revenues']]
        df.loc['Net capital expenditures'] = net_capex_list
        nwc_list = [i * parameter_list[5] for i in df.loc['Revenues']]
        df.loc['Changes in NWC'] = nwc_list
        free_cash_flow_list = df.loc['NOPAT'] - df.loc['Net capital expenditures'] - df.loc['Changes in NWC']
        df.loc['Free cash flow'] = free_cash_flow_list
        return df

    def discount_free_cash_flows(self, parameter_list, discount_rate, terminal_growth):
        free_cash_flow_df = self.get_free_cash_flow_forecast(parameter_list)
        df = free_cash_flow_df
        discount_factor_list = [(1 + discount_rate) ** i for i in free_cash_flow_df.columns]
        df.loc['Discount factor'] = discount_factor_list
        present_value_list = df.loc['Free cash flow'] / df.loc['Discount factor']
        df.loc['PV free cash flow'] = present_value_list
        df[0] = [0 for i in range(len(df))]
        df.loc['Sum PVs', 0] = df.loc['PV free cash flow', 1:5].sum()
        df.loc['Terminal value', 5] = df.loc['Free cash flow', 5] * (1 + terminal_growth) / (discount_rate - terminal_growth)
        df.loc['PV terminal value', 0] = df.loc['Terminal value', 5] / df.loc['Discount factor', 5]
        df.loc['Company value (enterprise value)', 0] = df.loc['Sum PVs', 0] + df.loc['PV terminal value', 0]
        df.loc['Net debt', 0] = parameter_list[-1]
        df.loc['Equity value', 0] = df.loc['Company value (enterprise value)', 0] - df.loc['Net debt', 0]
        equity_value = df.loc['Equity value', 0] 
        df = df.applymap(lambda x: comma_format(x))
        df = df.fillna('')
        column_name_list = range(6)
        df = df[column_name_list]
        return df, equity_value


st.title('Monte Carlo Valuation App')

with st.beta_expander('How to Use'):
    st.write('This application allows you to conduct a **probabilistic** \
        valuation of companies you are interested in. Please enter the \
        **stock ticker** of your company. Subsequently, the program will \
        provide you with **historical key metrics** you can use to specify \
        key inputs required for valuing the company of your choice. \
        In addition, you need to provide a **discount rate** and a **terminal \
        growth rate** at which your company is assumed to grow after year 5 \
        into the future.')

st.header('General company information')
ticker_input = st.text_input('Please enter your company ticker here:')
status_radio = st.radio('Please click Search when you are ready.', ('Entry', 'Search'))


@st.cache
def get_company_data():
    company = Company(ticker_input)
    return company

if status_radio == 'Search':
    company = get_company_data()
    st.header('Key Valuation Metrics')
    st.dataframe(company.inputs)


with st.beta_expander('Monte Carlo Simulation'):

    st.subheader('Random variables')
    st.write('When conducting a company valuation through a Monte Carlo simulation, \
        a variety of input metrics can be treated as random variables. Such \
        variables can be distributed according to different distributions. \
        Below, please specify the distribution from which the respective \
        variable values should be drawn.')

    parameter_dict_1 = {
        'latest revenue' : 0,
        'revenue growth': 0,
        'ebit margin' : 0,
        'tax rate' : 0,
        'capex ratio' : 0,
        'NWC ratio' : 0,
        'net debt' : 0
    }

    parameter_dict_2 = {
        'latest revenue' : 0,
        'revenue growth': 0,
        'ebit margin' : 0,
        'tax rate' : 0,
        'capex ratio' : 0,
        'NWC ratio' : 0
    }

    parameter_dict_distribution = {
        'latest revenue' : '',
        'revenue growth': '',
        'ebit margin' : '',
        'tax rate' : '',
        'capex ratio' : '',
        'NWC ratio' : ''
    }


    col11, col12, col13 = st.beta_columns(3)


    with col11:
        st.subheader('Revenue growth')
        radio_button_revenue_growth = st.radio('Choose growth rate distribution', ('Normal', 'Triangular', 'Uniform'))

        if radio_button_revenue_growth == 'Normal':
            mean_input = st.number_input('Mean revenue growth rate (in %)')
            stddev_input = st.number_input('Revenue growth rate std. dev. (in %)')
            parameter_dict_1['revenue growth'] = mean_input
            parameter_dict_2['revenue growth'] = stddev_input
            parameter_dict_distribution['revenue growth'] = 'normal'

        elif radio_button_revenue_growth == 'Triangular':
            lower_input = st.number_input('Lower end growth rate (in %)')
            mode_input = st.number_input('Mode growth rate (in %)')
            parameter_dict_1['revenue growth'] = lower_input
            parameter_dict_2['revenue growth'] = mode_input
            parameter_dict_distribution['revenue growth'] = 'triangular'

        elif radio_button_revenue_growth == 'Uniform':
            lower_input = st.number_input('Lower end growth rate (in %)')
            upper_input = st.number_input('Upper end growth rate (in %)')
            parameter_dict_1['revenue growth'] = lower_input
            parameter_dict_2['revenue growth'] = upper_input
            parameter_dict_distribution['revenue growth'] = 'uniform'
        

    with col12:
        st.subheader('EBIT margin')
        radio_button_ebit_margin = st.radio('Choose EBIT margin distribution', ('Normal', 'Triangular', 'Uniform'))

        if radio_button_ebit_margin == 'Normal':
            mean_input = st.number_input('Mean EBIT margin (in %)')
            stddev_input = st.number_input('EBIT margin std. dev. (in %)')
            parameter_dict_1['ebit margin'] = mean_input
            parameter_dict_2['ebit margin'] = stddev_input
            parameter_dict_distribution['ebit margin'] = 'normal'

        elif radio_button_ebit_margin == 'Triangular':
            lower_input = st.number_input('Lower end EBIT margin (in %)')
            mode_input = st.number_input('Mode EBIT margin (in %)')
            parameter_dict_1['ebit margin'] = lower_input
            parameter_dict_2['ebit margin'] = mode_input
            parameter_dict_distribution['ebit margin'] = 'triangular'

        elif radio_button_ebit_margin == 'Uniform':
            lower_input = st.number_input('Lower end EBIT margin (in %)')
            upper_input = st.number_input('Upper end EBIT margin (in %)')
            parameter_dict_1['ebit margin'] = lower_input
            parameter_dict_2['ebit margin'] = upper_input
            parameter_dict_distribution['ebit margin'] = 'uniform'


    with col13:
        st.subheader('Tax rate')
        radio_button_tax_rate = st.radio('Choose tax rate distribution', ('Normal', 'Triangular', 'Uniform'))

        if radio_button_tax_rate == 'Normal':
            mean_input = st.number_input('Mean tax rate (in %)')
            stddev_input = st.number_input('Tax rate std. dev. (in %)')
            parameter_dict_1['tax rate'] = mean_input
            parameter_dict_2['tax rate'] = stddev_input
            parameter_dict_distribution['tax rate'] = 'normal'

        elif radio_button_tax_rate == 'Triangular':
            lower_input = st.number_input('Lower end tax rate (in %)')
            mode_input = st.number_input('Mode tax rate (in %)')
            parameter_dict_1['tax rate'] = lower_input
            parameter_dict_2['tax rate'] = mode_input
            parameter_dict_distribution['tax rate'] = 'triangular'

        elif radio_button_tax_rate == 'Uniform':
            lower_input = st.number_input('Lower end tax rate (in %)')
            upper_input = st.number_input('Upper end tax rate (in %)')
            parameter_dict_1['tax rate'] = lower_input
            parameter_dict_2['tax rate'] = upper_input
            parameter_dict_distribution['tax rate'] = 'uniform'

        
    col21, col22, col23 = st.beta_columns(3)

    with col21:
        st.subheader('Net capex/sales')
        radio_button_tax_rate = st.radio('Choose capex ratio distribution', ('Normal', 'Triangular', 'Uniform'))

        if radio_button_tax_rate == 'Normal':
            mean_input = st.number_input('Mean capex ratio (in %)')
            stddev_input = st.number_input('capex ratio std. dev. (in %)')
            parameter_dict_1['capex ratio'] = mean_input
            parameter_dict_2['capex ratio'] = stddev_input
            parameter_dict_distribution['capex ratio'] = 'normal'

        elif radio_button_tax_rate == 'Triangular':
            lower_input = st.number_input('Lower end capex ratio (in %)')
            mode_input = st.number_input('Mode capex ratio (in %)')
            parameter_dict_1['capex ratio'] = lower_input
            parameter_dict_2['capex ratio'] = mode_input
            parameter_dict_distribution['capex ratio'] = 'triangular'

        elif radio_button_tax_rate == 'Uniform':
            lower_input = st.number_input('Lower end capex ratio (in %)')
            upper_input = st.number_input('Upper end capex ratio (in %)')
            parameter_dict_1['capex ratio'] = lower_input
            parameter_dict_2['capex ratio'] = upper_input
            parameter_dict_distribution['capex ratio'] = 'uniform'

    with col22:
        st.subheader('NWC/sales')
        radio_button_tax_rate = st.radio('Choose NWC ratio distribution', ('Normal', 'Triangular', 'Uniform'))

        if radio_button_tax_rate == 'Normal':
            mean_input = st.number_input('Mean NWC ratio (in %)')
            stddev_input = st.number_input('NWC ratio std. dev. (in %)')
            parameter_dict_1['NWC ratio'] = mean_input
            parameter_dict_2['NWC ratio'] = stddev_input
            parameter_dict_distribution['NWC ratio'] = 'normal'

        elif radio_button_tax_rate == 'Triangular':
            lower_input = st.number_input('Lower end NWC ratio (in %)')
            mode_input = st.number_input('Mode NWC ratio (in %)')
            parameter_dict_1['NWC ratio'] = lower_input
            parameter_dict_2['NWC ratio'] = mode_input
            parameter_dict_distribution['NWC ratio'] = 'triangular'

        elif radio_button_tax_rate == 'Uniform':
            lower_input = st.number_input('Lower end NWC ratio (in %)')
            upper_input = st.number_input('Upper end NWC ratio (in %)')
            parameter_dict_1['NWC ratio'] = lower_input
            parameter_dict_2['NWC ratio'] = upper_input
            parameter_dict_distribution['NWC ratio'] = 'uniform'

    with col23:
        st.subheader('Additional inputs')
        discount_rate = (st.number_input('Discount rate:')/100)
        terminal_growth = (st.number_input('Terminal growth rate:')/100)
        simulation_iterations = (st.number_input('Number of simulation iterations (<= 1000):'))
        inputs_radio = st.radio('Please click Search if you are ready.', ('Entry', 'Search'))

    equity_value_list = []
    revenue_list_of_lists = []
    ebit_list_of_lists = []
    if inputs_radio == 'Search':
        parameter_dict_1['latest revenue'] = company.income_statement.loc['totalRevenue', company.income_statement.columns[-1]]
        parameter_dict_1['net debt'] = company.inputs.loc['netDebt', 'Historical average']
        if simulation_iterations > 1000:
            simulation_iterations = 1000
        elif simulation_iterations < 0:
            simulation_iterations = 100
        for i in range(int(simulation_iterations)):
            model_input = calculate_value_distribution(parameter_dict_1, parameter_dict_2, parameter_dict_distribution)
            forecast_df = company.get_free_cash_flow_forecast(model_input)
            revenue_list_of_lists.append(forecast_df.loc['Revenues'])
            ebit_list_of_lists.append(forecast_df.loc['EBIT'])
            model_output, equity_value = company.discount_free_cash_flows(model_input, discount_rate, terminal_growth)
            equity_value_list.append(equity_value)
    
    st.header('MC Simulation Output')

    mean_equity_value = np.mean(equity_value_list)
    stddev_equity_value = np.std(equity_value_list)
    st.write('Mean equity value: $' + str(comma_format(mean_equity_value )))
    st.write('Equity value std. deviation: $' + str(comma_format(stddev_equity_value)))

    font_1 = {
        'family' : 'Arial',
            'size' : 12
    }

    font_2 = {
        'family' : 'Arial',
            'size' : 14
    }

    fig1 = plt.figure()
    plt.style.use('seaborn-whitegrid')
    plt.title(ticker_input + ' Monte Carlo Simulation', fontdict = font_1)
    plt.xlabel('Equity value (in $)', fontdict = font_1)
    plt.ylabel('Number of occurences', fontdict = font_1)
    plt.hist(equity_value_list, bins = 50, color = '#006699', edgecolor = 'black')
    st.pyplot(fig1)


    col31, col32 = st.beta_columns(2)
    with col31:
        fig2 = plt.figure()
        x = range(6)[1:6]
        plt.style.use('seaborn-whitegrid')
        plt.title('Revenue Forecast Monte Carlo Simulation', fontdict = font_2)
        plt.xticks(ticks = x)
        plt.xlabel('Year', fontdict = font_2)
        plt.ylabel('Revenue (in $)', fontdict = font_2)
        for i in revenue_list_of_lists:
            plt.plot(x, i)
        st.pyplot(fig2)
    
    with col32:
        fig3 = plt.figure()
        x = range(6)[1:6]
        plt.style.use('seaborn-whitegrid')
        plt.title('EBIT Forecast Monte Carlo Simulation', fontdict = font_2)
        plt.xticks(ticks = x)
        plt.xlabel('Year', fontdict = font_2)
        plt.ylabel('EBIT (in $)', fontdict = font_2)
        for i in ebit_list_of_lists:
            plt.plot(x, i)
        st.pyplot(fig3)

st.write('Disclaimer: Information and output provided on this site do \
    not constitute investment advice.')
st.write('Copyright (c) 2021 Julian Marx')
