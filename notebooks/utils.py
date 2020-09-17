import jax.numpy as np
import numpy as onp
import pandas as pd
import janitor
import altair as alt
from numpyro.diagnostics import hpdi, print_summary

PD = pd.DataFrame
def j_summary(samples, ctype='hist', properties={'width':800}):
#     print(vcov)
    if type(samples) == dict:
        print_summary(samples, 0.89, False)
        df = pd.DataFrame(samples).clean_names()
    else:
        print_summary(dict(zip(samples.columns, samples.T.values)), 0.89, False)
        df = samples
    display(df.corr())
    df = df if len(df)< 5000 else df.sample(n=4000)
    base = alt.Chart(df).properties(height=30)

    if ctype=='density':
        l = [ base.mark_line().transform_density(
            row, as_ = [ row, 'density'], 
            ).encode(
            alt.X(f'{row}:Q'),
            alt.Y('density:Q')
        )  for row in df.columns]

        density = alt.vconcat(*l)
        return_chart = density

        
    
    if ctype=='hist':
        hist = base.mark_bar().encode(
            alt.X(bin=alt.Bin(maxbins=20), field=alt.repeat("row"), type='quantitative'),
            y = alt.Y(title=None, aggregate='count', type='quantitative')
        ).repeat(row=[c for c in df.columns])
        return_chart = hist
        
    display(return_chart)
#     return return_chart
   

def alt_plot(domain, f, mark= 'line', properties={'width':800}):
    """
    description:
    ------------
    plot a function on a given domain (or support?)
    
    parameters:
    -----------
    domain (1-d array): an array containg the values that f will be calculated on
    f : a function that will calculate the curve values
    mark (str): the kind of mark the chart will use
    properties (dict) : a dictionary containing attributes for the altair chart
    
    returns:
    -------
    an altair chart
    """
    df = pd.DataFrame({'x': domain, 'y': f(domain)})
    c = alt.Chart(df).mark_line().encode(x='x:Q',y='y:Q').properties(**properties)
    return c


def plot_lines( data, xs=None, options={}):
    """
    description:
    ------------
    plot multiple curves on a single altair plot
    
    parameters:
    -----------
    data: a array with columns representing values of a curve
    xs (optional): an array that will serve as x values on the plotted curve
    options (optional): a dictionary to pass options to altair.mark_line()
    
    returns:
    -------
    an altair chart
    """
    
    # basic idea: 
    # take columnar data eg. shape = (kind,samples)
    # transpose it, shape = (samples, kind) 
    # using melt and put in "long form". shape = (kind x samples, 2)
    # then use altair "detail" to segregate otherwise similar data (could use color, or other quality too)
    if isinstance(data, pd.DataFrame):
        _df = data 
    else:
        _df = pd.DataFrame(data = data, index =  xs if xs is not None else np.arange(len(data)) )
    _df = _df.T.melt().assign(v_name = lambda r:  _df.columns[r.index % len(_df.columns)])
    return alt.Chart(_df).mark_line(**options).encode(
        x= alt.X('variable:Q', title='x-axis'),
        y= alt.Y('value:Q', title='y-axis'),
        detail='v_name:N')

