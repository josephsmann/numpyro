import pandas as pd
import janitor
import altair as alt
from numpyro.diagnostics import hpdi, print_summary

def j_summary(samples):
    print_summary(samples, 0.89, False)
    df = pd.DataFrame(samples).clean_names()
    df = df if len(df)< 5000 else df.sample(n=4000)
    base = alt.Chart(df).mark_bar().properties(height=30)

    return base.encode(
        alt.X(bin=alt.Bin(maxbins=20), field=alt.repeat("row"), type='quantitative'),
        y = alt.Y(title=None, aggregate='count', type='quantitative')
    ).repeat(row=[c for c in df.columns])