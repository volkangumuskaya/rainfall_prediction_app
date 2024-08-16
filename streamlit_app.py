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
st.image('images/el-chalten.jpg','El Chalten, Patagonia')
st.sidebar.title("")
'''
# :rain_cloud: Daily rain predictions for the cities in Netherlands

This is an example project to demonstrate MLOps, data visualisation and DS capabilities. The prediction algorithm is a simple model in the sole purpose of demonstration purposes. 
As such, it is not a full-blown weather model that aims to provide industry standard predictions. 
More can be found in the [github repo here](https://github.com/volkangumuskaya/rainfall_prediction_app/blob/main/README.md)

The data is provided via the [KNMI API](https://daggegevens.knmi.nl/klimatologie/daggegevens)
'''

# Add some spacing
''
''
df=df.fillna(0)
df = df.replace('',0)
min_value = df['date'].min()
max_value = df['date'].max()
print("min",min_value,'max',max_value)

from_year, to_year = st.select_slider(
    "Select a range of date",
    options=df.date.sort_values().unique(),
    value=(min_value, max_value),
)

stations = df['station_name'].sort_values().unique()

if not len(stations):
    st.warning("Select at least one station")

selected_stations = st.multiselect(
    'Which station would you like to view?',
    stations,
    ["Eindhoven","Rotterdam"])

''


# Filter the data
filtered_df = df[
    (df['station_name'].isin(selected_stations))
    & (df['date'] <= to_year)
    & (from_year <= df['date'])
]
# print(filtered_df)
# print(df)
# print(stations)
st.header('Rainfall predictions in the selected dates&stations', divider='gray')

''
filtered_df['date']=filtered_df['date'].astype('str')
st.line_chart(
    filtered_df,
    x='date',
    y='rain_amount_mm_prediction',
    color='station_name',
)

''

import numpy as np
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go

selected_stat = st.selectbox(
    'Which station to be inspected in detail?',
    stations
)

#plot detailed
df = df[
    (df['station_name']==selected_stat)
    & (df['date'] <= to_year)
    & (from_year <= df['date'])
].copy()
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


fig.update_traces(marker=dict(size=5.5,
                              line=dict(width=0.3, color='black')),
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

# col1,col2,col3 = st.columns(3)
# col1.metric("Temperature", "70 °F", "1.2 °F")
# col2.metric("Wind", "9 mph", "-8%")
# col3.metric("Humidity", "86%", "4%")

# st.subheader('Eind', divider='gray')
# col1,col2,col3 = st.columns(3)
# col1.metric("Temperature", "70 °F", "1.2 °F")
# col2.metric("Wind", "9 mph", "-8%")
# col3.metric("Humidity", "86%", "4%")

###
del(df)
print('Measurements part start')
print('Reading df')
# Instead of a CSV on disk, you could read from an HTTP endpoint here too.
DATA_FILENAME = Path(__file__).parent/'files/latest_measurements.csv'
df = pd.read_csv(DATA_FILENAME)
print ('df read with shape ',df.shape,' and type ',type(df))
print (df.head(2))


cols=['stationname','Temperature','Rainfall_Duration_last_hour_minutes','Amount_Rainfall_last_Hour_in_mm','Total_cloud_cover_percentage',
      'Air_pressure_in_hPa','Wind_Speed_kmh','Wind_Direction']
df=df[cols].copy()
col_names_dict = {
    "stationname": "stationname",
    "Rainfall_Duration_last_hour_minutes": "Rainfall duration (mins) ",
    "Amount_Rainfall_last_Hour_in_mm": "Rainfall amount (mm)",
    'Wind_Direction':"Wind Direction",
    "Wind_Speed_kmh":'Windspeed (km/h)',
    "Air_pressure_in_hPa":'Pressure (hPA)',
    "Total_cloud_cover_percentage":'Cloud cover (%)',
    'Temperature':'Temperature (C)'
}

df=df.rename(columns=col_names_dict)
st.header('Latest measurements', divider='red')

# for i in range(0, len(df)):
#     print ('row:',i)
#     for j in range(0, len(df.columns)-1):
#         print('col',j,'**rank', j%3+1,df.columns[j+1])

for i in range(0, len(df)):
    print ('row:',i)
    st.subheader(df.iloc[i]['stationname'], divider='gray')
    cols = st.columns(3)
    for j in range(0, len(df.columns)-1):
        col = cols[j % len(cols)+1]
        with col:
            st.metric(
                label=df.columns[j+1],
                value=df.iloc[i][j+1]
            )


# for i in 
# cols = st.columns(len(df.columns)
# for i, city in enumerate(df.stationname):
    
#     col = cols[i % len(cols)]

#     with col:
#         st.metric(
#             label=f'{City}',
#             value=f'{}B',
#             delta=growth,
#             delta_color=delta_color
#         )
