import streamlit as st
import pandas as pd
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go

#####################
## Monthly dashboards
#####################
st.subheader('Max - min temperature plots', divider='gray')
selected_station_t = st.selectbox(
    'Which station would you like to view for temperature plots?',
    stations)

from_year_t, to_year_t = st.select_slider(
    "Select a range of years for monthly temp plots",
    options=df[df.year>=min_year].year.sort_values().unique(),
    value=(min_year, max_year),
)

#plot
def plot_max_min_temps(selected_stations,fromyear,toyear,df_f,selected_palet_max,selected_palet_min):
    cols=['station','station_name','year','month','month_name','monthly_rain_mm','monthly_max_temp','monthly_min_temp']
    tmp=df_f[(df_f.station_name==selected_stations)&(df_f.year>=fromyear)&(df_f.year<=toyear)].drop_duplicates(subset=['year','month'])[cols]
    tmp_min=tmp.groupby(['month','month_name','station'])[['monthly_max_temp']].mean().reset_index()
    tmp_max=tmp.groupby(['month','month_name','station'])[['monthly_min_temp']].mean().reset_index()

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    df_len=len(tmp.year.unique())
    interval_=int(79/(df_len-1))
    x_categories = tmp.sort_values('month').month.to_list()
    for yr,color_index in zip(tmp.year.unique(),range(20, interval_*df_len, interval_)):
        sel_tmp=tmp[tmp.year==yr]
        fig.add_trace(
            go.Scatter(x=sel_tmp.month_name, y=sel_tmp.monthly_max_temp,
                       line=dict(width=1.7),
                       marker_size=6, marker_color=selected_palet_max[color_index],
                   mode='lines+markers',
                   name=str(yr)),
            secondary_y=False
        )

        fig.add_trace(
            go.Scatter(x=sel_tmp.month_name, y=sel_tmp.monthly_min_temp,
                       line=dict(width=1.7),
                       marker_size=6, marker_color=selected_palet_min[color_index],
                       mode='lines+markers',
                       name=str(yr),showlegend=False),
            secondary_y=True
        )

    fig.add_trace(
        go.Scatter(x=tmp_min.month_name, y=tmp_min.monthly_max_temp,
               line = dict(dash='dot',width=1.2,color='black'),
               mode='lines+markers',
               name="Average min from "+str(fromyear)+" to "+str(toyear)),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=tmp_max.month_name, y=tmp_max.monthly_min_temp,
               line = dict(dash='dot',width=1.2,color='black'),
               mode='lines+markers',
               name="Average max from "+str(fromyear)+" to "+str(toyear)),
        secondary_y=False
    )
    fig.add_hline(y=0, line_width=1.0, line_color="black")

    fig.update_xaxes(type="category",categoryorder='array', categoryarray= x_categories)
    fig.update_layout(
        title="Max and min temp per month in the years "+str(fromyear)+'-'+str(toyear)+" in "+ selected_stations,
        xaxis_title="Month",
        yaxis_title="Celcius degree",
        legend_title="Years",
    )
    fig.update_yaxes(title_text="Celcius degree", secondary_y=True)
    fig.update_yaxes(range=[-20, 50], secondary_y=True)
    fig.update_yaxes(range=[-20, 50], secondary_y=False)
    return fig

from plotly.express.colors import sample_colorscale
from sklearn.preprocessing import minmax_scale
colors_ = np.linspace(1, 10, 100)
discrete_colors_mx = sample_colorscale('Reds', minmax_scale(colors_))
discrete_colors_mn = sample_colorscale('Blues', minmax_scale(colors_))

figure=plot_max_min_temps(selected_stations=selected_station_t,
                          fromyear=from_year_t,toyear=to_year_t,
                          df_f=df,
                          selected_palet_max=discrete_colors_mx,selected_palet_min=discrete_colors_mn
                         )
st.plotly_chart(figure)
