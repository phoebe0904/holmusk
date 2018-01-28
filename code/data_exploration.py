import scipy
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


def data_exp_bin(df,var):
    print('Data exploration for binary variable: ', var.upper())
    # frequency
    print(df.groupby(var).size())   
    # no. of missing
    print('No. of missing = ', df[var].isnull().sum(), '(', round(100*df[var].isnull().sum().astype('float64')/float(len(df[var])),4), "%)")
    print('\n')

def data_exp_cat(df, var):
    print("Data exploration for categorical variable: ", var.upper())
    # frequency
    print(df.groupby(var).size())
    # plot frequency
    V = sns.countplot(y=var, data=df, palette="PuBuGn_d");
    V.set(ylabel=var, title='Frequency of ' + var)
    plt.show()
    # no. of missing
    print('No. of missing = ', df[var].isnull().sum(), '(', round(100*df[var].isnull().sum().astype('float64')/float(len(df[var])),4), '%)')
    print('\n')
    
def data_exp_num(df,var,target=None):
    print('Data exploration for numeric variable: ', var.upper())
    # basic stat
    print('Basic stat: \n', df[var].describe())
    # no. of missing
    print('No. of missing = ', df[var].isnull().sum(), '(', round(100*df[var].isnull().sum().astype('float64')/float(len(df[var])),4), '%)')
    if target is None:
        pass
    else:
        # correlation with total amount
        print('Correlation with ', target, ': \n', scipy.stats.pearsonr(df.ix[df[var].notnull(),var], df.ix[df[var].notnull(),target]))
    # plot histogram    
    plt.title(('Histogram for ' + var), loc='center')
    plt.hist(df[var], bins=50)
    plt.xlabel(var)
    plt.ylabel('count')
    plt.show() 
    if target is None:
        pass
    else:
        # scatter plot with total amount    
        plt.plot(df[var], df[target],"o")
        plt.title(('Scatter plot of ' + var + ' with ' + target + ': '), loc='center')
        plt.xlabel(var)
        plt.ylabel(target)
        plt.show()  
    print('\n')     















