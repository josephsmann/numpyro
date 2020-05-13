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
    df = pd.DataFrame({'x': domain, 'y': f(domain)})
    c = alt.Chart(df).mark_line().encode(x='x:Q',y='y:Q').properties(**properties)
    return c

# def alt_density()

# alt.Chart(source).transform_fold(
#     ['petalWidth',
#      'petalLength',
#      'sepalWidth',
#      'sepalLength'],
#     as_ = ['Measurement_type', 'value']
# ).transform_density(
#     density='value',
#     bandwidth=0.3,
#     groupby=['Measurement_type'],
#     extent= [0, 8]
# ).mark_area().encode(
#     alt.X('value:Q'),
#     alt.Y('density:Q'),
#     alt.Row('Measurement_type:N')
# ).properties(width=300, height=50)