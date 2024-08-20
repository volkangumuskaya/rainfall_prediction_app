

print('Importing libs..')
#Import libraries

import numpy as np
import pandas as pd
import os
import calendar
import datetime
import io
import requests
from requests.exceptions import HTTPError

#set training end date to 1 day ago
end_date=(datetime.date.today()-datetime.timedelta(days=1))

#start_date is 10 years before end_date
start_date=end_date-datetime.timedelta(days=3650)

end_date=int(end_date.strftime('%Y%m%d'))
start_date=int(start_date.strftime('%Y%m%d'))

data = {
    'start': start_date,
    'end': end_date,
    'vars': 'ALL',
    'stns': '240:344:370:380:269',
}

URL = "https://daggegevens.knmi.nl/klimatologie/daggegevens"

try:
    response = requests.post(URL, data=data)
    response.raise_for_status()
except HTTPError as http_err:
    print(f"HTTP error occurred : {http_err}")
except Exception as err:
    print(f"Other error occurred: {err}")
else:
    print("Success with status code",response.status_code)


##Parse response as df
s=str(response.content)# response as string
s=s[s.rindex('#')+1:] #find last # and delete the first part
s=s.replace('\\n','\n') #replace //n with /n
s=s.replace(' ','') #delete space
s=s.replace("'",'') #delete '
df=pd.read_csv(io.StringIO(s), sep=",")
type(df)

#data preparation
selected_cols=['STN', 'YYYYMMDD', 'DDVEC', 'FG', 'TG', 'TN', 'TX',
          'DR', 'RH', 'RHX', 'RHXH',
          'PG', 'PX', 'PN','UX', 'UN']

col_names_dict = {
    "STN": "station",
    "YYYYMMDD": "date",
    "DDVEC": "wind_direction",
    'FG':"mean_wind_speed",
    "TG":'mean_temp',
    "TN":'min_temp',
    "TX":'max_temp',
    'DR':'rain_duration',
    'RH':'rain_amount_mm',
    'RHX':'max_hourly_rain_mm',
    'RHXH':'time_of_max_rain',
    'PG':'mean_pressure',
    'PX':'max_pressure',
    'PN':'min_pressure',
    'UX':'max_humidity',
    'UN':'min_humidity'
}


#filter out not selected columns
df=df[[x for x in df.columns if x in col_names_dict]]

stations_df=pd.read_csv('files/station_list_klimatologie.csv')
df=pd.merge(df,stations_df,how='left',on='STN')
df['station_name']=df['station_name'].fillna('unknown')

#Convert data to the expected format by LGBM model
df.columns = df.columns.str.strip()
df=df.rename(columns=col_names_dict)

df['rain_amount_mm']=df['rain_amount_mm']/10
df.loc[df.rain_amount_mm <=0, 'rain_amount_mm'] = 0

df['max_hourly_rain_mm']=df['max_hourly_rain_mm']/10
df.loc[df.max_hourly_rain_mm <=0, 'max_hourly_rain_mm'] = 0

df['mean_temp']=df['mean_temp']/10
df['min_temp']=df['min_temp']/10
df['max_temp']=df['max_temp']/10
df['mean_pressure']=df['mean_pressure']/10
df['min_pressure']=df['min_pressure']/10
df['max_pressure']=df['max_pressure']/10
df['rain_duration']=df['rain_duration']/10


df['year']=np.floor((df['date']/10000)).astype('int')
df['month']=(np.floor(df['date']/100)-np.floor((df['date']/10000))*100).astype('int')
df['day']=(df['date']-(np.floor(df['date']/100)*100)).astype('int')
df['daymonthyear']=df['date']
df['yearmonth']=df['year']*100+df['month']
df['date']=df['date'].astype('string')

#seasons
#map one column to another
conditions = [(df['month'].isin ([12,1,2])),
              (df['month'].isin ([3,4,5])),
              (df['month'].isin ([6,7,8])),
            (df['month'].isin ([9,10,11]))
              ]
choices = ['Winter', 'Spring', 'Summer','Fall']
df['season'] = np.select(conditions, choices,default='na')
df['month_name'] =df['month'].apply(lambda x: calendar.month_name[x])

df['monthly_rain_mm']=df.groupby(['year','month','station'])[['rain_amount_mm']].transform('sum')

df['monthly_max_temp']=df.groupby(['year','month','station'])[['max_temp']].transform('max')
df['monthly_min_temp']=df.groupby(['year','month','station'])[['min_temp']].transform('min')

import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.graph_objects as go


#plot
def plot_monthly_rain_per_years(selected_stations,fromyear,toyear,df_f):
    cols=['station','station_name','year','month','month_name','monthly_rain_mm']
    tmp=df_f[(df_f.station.isin(selected_stations))&(df_f.year>=fromyear)&(df_f.year<=toyear)].drop_duplicates(subset=['year','month'])[cols]
    tmp_mean=tmp.groupby(['month','month_name','station'])[['monthly_rain_mm']].mean().reset_index()
    max_monthly_rain_overall = np.ceil(tmp.monthly_rain_mm.max() / 20) * 20
    min_monthly_rain_overall = 0
    x_categories = tmp.sort_values('month').month.to_list()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for yr in tmp.year.unique():
        sel_tmp=tmp[tmp.year==yr]
        fig.add_trace(
            go.Scatter(x=sel_tmp.month_name, y=sel_tmp.monthly_rain_mm,
                       line=dict(width=1.1),marker_size=7,
                   mode='lines+markers',
                   name=str(yr)),
            secondary_y=False
        )


    fig.add_trace(
        go.Scatter(x=tmp_mean.month_name, y=tmp_mean.monthly_rain_mm,
               line = dict(dash='dot',width=2,color='black'),
               mode='lines+markers',
               name="Average from "+str(fromyear)+" to "+str(toyear)),
        secondary_y=True
    )

    fig.update_xaxes(type="category",categoryorder='array', categoryarray= x_categories)
    fig.update_layout(
        title="Total rain per month in the years "+str(fromyear)+'-'+str(toyear),
        xaxis_title="Month",
        yaxis_title="Monthly total rain (mm)",
        legend_title="Years",
    )
    fig.update_yaxes(title_text="Monthly total rain (mm)", secondary_y=True)
    fig.update_yaxes(range=[min_monthly_rain_overall, max_monthly_rain_overall], secondary_y=True)
    fig.update_yaxes(range=[min_monthly_rain_overall, max_monthly_rain_overall], secondary_y=False)
    return fig

figure2=plot_monthly_rain_per_years([370],2019,2024,df)
path='images/monthly_rain.png'
pio.write_image(figure2, path,width=1600, height=900)
print("fig saved to: ", path)


#plot
def plot_max_min_temps(selected_stations,fromyear,toyear,df_f,selected_palet):
    cols=['station','station_name','year','month','month_name','monthly_rain_mm','monthly_max_temp','monthly_min_temp']
    tmp=df_f[(df_f.station.isin(selected_stations))&(df_f.year>=fromyear)&(df_f.year<=toyear)].drop_duplicates(subset=['year','month'])[cols]
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
                       marker_size=6, marker_color=selected_palet[color_index],
                   mode='lines+markers',
                   name=str(yr)),
            secondary_y=False
        )

        fig.add_trace(
            go.Scatter(x=sel_tmp.month_name, y=sel_tmp.monthly_min_temp,
                       line=dict(width=1.7),
                       marker_size=6, marker_color=selected_palet[color_index],
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
        title="Max and min temp per month in the years "+str(fromyear)+'-'+str(toyear),
        xaxis_title="Month",
        yaxis_title="Celcius degrees",
        legend_title="Years",
    )
    fig.update_yaxes(title_text="Celcius degrees", secondary_y=True)
    fig.update_yaxes(range=[-20, 50], secondary_y=True)
    fig.update_yaxes(range=[-20, 50], secondary_y=False)
    return fig

from plotly.express.colors import sample_colorscale
from sklearn.preprocessing import minmax_scale
colors_ = np.linspace(1, 10, 100)
discrete_colors = sample_colorscale('Reds', minmax_scale(colors_))
figure=plot_max_min_temps(selected_stations=[370],
                          fromyear=2018,toyear=2024,
                          df_f=df,
                          selected_palet=discrete_colors)
path='images/monthly_temps.png'
pio.write_image(figure, path,width=1600, height=900)
print("fig saved to: ", path)




