# --------------
# Import the required Libraries
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import calendar
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# Generate a line chart that visualizes the readings in the months

def line_chart(df,period,col):
    """ A line chart that visualizes the readings in the months
    
    This function accepts the dataframe df ,period(day/month/year) and col(feature), which plots the aggregated value of the feature based on the periods. Ensure the period labels are properly named.
    
    Keyword arguments:
    df - Pandas dataframe which has the data.
    period - Period of time over which you want to aggregate the data
    col - Feature of the dataframe
    
    """
    if (period == 'Month'):
        df1 = df.groupby(df.index.month).mean()
    elif (period == 'Year'):
        df1 = df.groupby(df.index.year).mean()
    elif (period == 'Day'):
        df1 = df.groupby(df.index.day).mean()
    
    xvalues = calendar.month_name[1:]
    yvalues = df1[col]

    plt.plot(xvalues,yvalues)
    plt.title("Temperature Trend, 2012")
    plt.xlabel(period)
    plt.xticks(rotation = 90)
    plt.ylabel("Temp (C)")
    plt.show()
    
# Function to perform univariate analysis of categorical columns
def plot_categorical_columns(df):
    """ Univariate analysis of categorical columns
    
    This function accepts the dataframe df which analyzes all the variable in the data and performs the univariate analysis using bar plot.
    
    Keyword arguments:
    df - Pandas dataframe which has the data.
    
    """
    categorical_var = df.select_dtypes(include = 'object').columns.tolist()
    for categoric in categorical_var:
        df[categoric].value_counts().plot(kind = 'bar')
    
# Function to plot continous plots
def plot_cont(df,plt_typ):
    """ Univariate analysis of Numerical columns
    
    This function accepts the dataframe df, plt_type(boxplot/distplot) which analyzes all the variable in the data and performs the univariate analysis using boxplot or distplot plot.
    
    Keyword arguments:
    df - Pandas dataframe which has the data.
    plt_type - type of plot through which you want to visualize the data
    
    """
    numerical_var = df.select_dtypes(include = 'number').columns.tolist()
    print(numerical_var)
    for x in range(0,len(numerical_var),2):
        plt.figure(figsize = (10,4))
        plt.subplot(121)
        if plt_typ == 'boxplot':
            sns.boxplot(df[numerical_var[x]])
            plt.subplot(122)
            sns.boxplot(df[numerical_var[x+1]])
        elif plt_typ == 'distplot':
            sns.distplot(df[numerical_var[x]])
            plt.subplot(122)
            sns.distplot(df[numerical_var[x+1]])
        else:
            print("invalid plot type selection")

# Function to plot grouped values based on the feature
def group_values(df,col1,agg1,col2):
    """ Agrregate values by grouping
    
    This function accepts a dataframe, 2 column(feature) and aggregated function(agg1) which groupby the dataframe based on the column and plots the bar plot.
   
    Keyword arguments:
    df - Pandas dataframe which has the data.
    col1 - Feature of the dataframe on which values will be aggregated.
    agg1 - Dictionary of aggregate functions with feature as the key and func as the value
    col2 - Feature of the dataframe to be plot against grouped data.
    
    Returns:
    grouping - Dataframe with all columns on which it is grouped on.
    """    
    aggregate = {'mean':np.mean,'max':np.max,'min':np.min}
    group = df.groupby(col1).agg(aggregate[agg1])
    plt.figure(figsize = (10,10))
    group[col2].plot(kind = 'bar')
    plt.ylabel(col2)
    plt.show()
# Read the Data and pass the parameter as parse_dates=True, index_col='Date/Time'
weather_df = pd.read_csv(path, parse_dates = True, index_col = 'Date/Time')
weather_df = pd.DataFrame(weather_df)
#print(weather_df.head())
#print(weather_df.shape)

#Checking categorical and numerical variables
catergorical_var = weather_df.select_dtypes(include = 'object').columns.tolist()
#print(catergorical_var)
numerical_var = weather_df.select_dtypes(include = 'number').columns.tolist()
#print(numerical_var)

# Lets try to generate a line chart that visualizes the temperature readings in the months.
# Call the function line_chart() with the appropriate parameters.
line_chart(weather_df,'Month','Temp (C)')

# Now let's perform the univariate analysis of categorical features.
# Call the "function plot_categorical_columns()" with appropriate parameters.
plot_categorical_columns(weather_df)

# Let's plot the Univariate analysis of Numerical columns.
# Call the function "plot_cont()" with the appropriate parameters to plot distplot
plot_cont(weather_df,'distplot')

# Call the function "plot_cont()" with the appropriate parameters to plot boxplot
plot_cont(weather_df,'boxplot')

# Groupby the data by Weather and plot the graph of the mean visibility during different weathers. Call the function group_values to plot the graph.
# Feel free to try on diffrent features and aggregated functions like max, min.
group_values(weather_df,'Weather','mean','Visibility (km)')



