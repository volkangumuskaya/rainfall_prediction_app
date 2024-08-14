import streamlit as st
import pandas as pd
import math
from pathlib import Path

# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='volkan-ai',
    page_icon=':volcano:', # This is an emoji shortcode. Could be a URL too.
)

# -----------------------------------------------------------------------------
# Declare some useful functions.

@st.cache_data
def get_df():
    """Grab GDP data from a CSV file.

    This uses caching to avoid having to read the file every time. If we were
    reading from an HTTP endpoint instead of a file, it's a good idea to set
    a maximum age to the cache with the TTL argument: @st.cache_data(ttl='1d')
    """
    print('Reading df')
    # Instead of a CSV on disk, you could read from an HTTP endpoint here too.
    DATA_FILENAME = Path(__file__).parent/'files/daily_prediction.csv'
    tmp_df = pd.read_csv(DATA_FILENAME)
    print ('df read with shape ',tmp_df.shape,' and type ',type(tmp_df))
    print ('df min and max ',tmp_df['date'].min(),tmp_df['date'].max())
    return tmp_df

df = get_df()

# -----------------------------------------------------------------------------
# Draw the actual page

# Set the title that appears at the top of the page.
'''
# :rain_cloud: Daily rain predictions for the cities in Netherlands

This is an example project to demonstrate MLOps, data visualisation and DS skills. The prediction algorithm is a simple model in the sole purpose of demosntration purposes. 
This is not a full-blown weather model that aims to provide industry standard predictions. 
More can be found in the [github repo here](https://github.com/volkangumuskaya/rainfall_prediction_app/blob/main/README.md)

The data is streamed using [KNMI API](https://daggegevens.knmi.nl/klimatologie/daggegevens)
'''

# Add some spacing
''
''
df=df.fillna(0)
df = df.replace('',0)
min_value = df['date'].min()
max_value = df['date'].max()
print("min",min_value,'max',max_value)

start_color, end_color = st.select_slider(
    "Select a range of color wavelength",
    options=df.date.sort_values().unique(),
    value=(min_value, max_value),
)

from_year, to_year = st.slider(
    'Which dates are you interested in?',
    min_value=min_value,
    max_value=max_value,
    value=[min_value, max_value])

stations = df['station_name'].sort_values().unique()

if not len(stations):
    st.warning("Select at least one station")

selected_stations = st.multiselect(
    'Which station would you like to view?',
    stations,
    ["Eindhoven"])

''
''
''

# Filter the data
filtered_df = df[
    (df['station_name'].isin(selected_stations))
    & (df['date'] <= to_year)
    & (from_year <= df['date'])
]
print(filtered_df)
print(df)
print(stations)
st.header('Rainfall predictions', divider='gray')

''
filtered_df['date']=filtered_df['date'].astype('str')
st.line_chart(
    filtered_df,
    x='date',
    y='rain_amount_mm_prediction',
    color='station_name',
)

''
''

import pandas as pd
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go

selected_stat = st.selectbox(
    'Which station to be inspected?',
    stations
)
#plot heatmap
# print('Reading daily_prediction.csv')
# filename='files/daily_prediction.csv'
# df=pd.read_csv(filename)
df=df[df['station_name']==selected_stat].copy()
df['date']=df['date'].astype('str')
y_max=np.ceil(max(df.rain_amount_mm_prediction.max(),df.next_day_rain_mm.max())/20)*20
y_min=-y_max

kwargs = {
    'cbar': False,
    'linewidths': 0.2,
    'linecolor': 'white',
    'annot': True}

df['error']=df['rain_amount_mm_prediction']-df['next_day_rain_mm']

fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(
    go.Bar(x=df.date, y=df.error,
           # text=round(metrics.R2, 2),
           marker_color='dodgerblue', opacity=0.9,
           name="Error"),
    secondary_y=False
)

fig.add_trace(
    go.Scatter(x=df.date, y=df.next_day_rain_mm,
               mode='markers', name="Next day rain"),
    secondary_y=True
)

fig.add_trace(
    go.Scatter(x=df.date, y=df.rain_amount_mm_prediction,
               mode='markers', name="Rain prediction"),
    secondary_y=True
)


fig.update_traces(marker=dict(size=6,
                              line=dict(width=1, color='black')),
                  selector=dict(mode='markers'))
fig.update_layout(
    title="Detailed predictions, actuals and errors for selected station",
    xaxis_title="Date",
    yaxis_title="Rain amount",
    legend_title="Legend",
)
fig.update_yaxes(title_text="Rain amount", secondary_y=True)
fig.update_layout(xaxis_type='category')
fig.update_xaxes(tickangle=270)
fig.update_yaxes(range=[y_min,y_max], secondary_y=False)
fig.update_yaxes(range=[y_min,y_max], secondary_y=True)
st.plotly_chart(fig)
