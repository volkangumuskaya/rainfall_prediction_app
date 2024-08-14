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
# :rain_cloud: Daily rain predictions for some cities in Netherlands

This is an example project to demonstrate MLOps, data visualisation and DS skills. The prediction algorithm is a simple model in the sole purpose of demosntration purposes. 
This is not a full-blown weather model that aims to provide industry standard predictions. More can be found in the github repo: 

For detailed info please [check here](https://github.com/volkangumuskaya/rainfall_prediction_app/blob/main/README.md)
'''

# Add some spacing
''
''
df=df.fillna(0)
df = df.replace('',0)
min_value = df['date'].min()
max_value = df['date'].max()
print("min",min_value,'max',max_value)


from_year, to_year = st.slider(
    'Which dates are you interested in?',
    min_value=min_value,
    max_value=max_value,
    value=[min_value, max_value])

stations = df['station_name'].unique()

if not len(stations):
    st.warning("Select at least one station")

selected_stations = st.multiselect(
    'Which st would you like to view?',
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


# first_year = gdp_df[gdp_df['Year'] == from_year]
# last_year = gdp_df[gdp_df['Year'] == to_year]

# st.header(f'GDP in {to_year}', divider='gray')

# ''

# cols = st.columns(4)

# for i, country in enumerate(selected_countries):
#     col = cols[i % len(cols)]

#     with col:
#         first_gdp = first_year[first_year['Country Code'] == country]['GDP'].iat[0] / 1000000000
#         last_gdp = last_year[last_year['Country Code'] == country]['GDP'].iat[0] / 1000000000

#         if math.isnan(first_gdp):
#             growth = 'n/a'
#             delta_color = 'off'
#         else:
#             growth = f'{last_gdp / first_gdp:,.2f}x'
#             delta_color = 'normal'

#         st.metric(
#             label=f'{country} GDP',
#             value=f'{last_gdp:,.0f}B',
#             delta=growth,
#             delta_color=delta_color
#         )
# import plotly.graph_objects as go
# # Sample data
# categories = ['A', 'B', 'C', 'D', 'E']
# bar_values = [3, 7, 2, 5, 8]
# line_values = [2, 6, 4, 8, 7]
# # Create a figure
# fig = go.Figure()

# # Add line trace
# fig.add_trace(go.Scatter(
#     x=selected_df.date,
#     y=selected_df.next_rain_mm,
#     name='Line Chart',
#     mode='lines',
#     marker_color='red'
# ))

# # Add line trace
# fig.add_trace(go.Scatter(
#     x=selected_df.date,
#     y=selected_df.rain_amount_mm_prediction,
#     name='Line Chart',
#     mode='markers',
#     marker_color='red'
# ))

# # Update layout
# fig.update_layout(
#     title='Bar and Line Chart',
#     xaxis_title='Categories',
#     yaxis_title='Values'
# )

# # Display the chart in Streamlit
# st.plotly_chart(fig)

from plotly.subplots import make_subplots
import plotly.graph_objects as go
y_max=np.ceil(max(df.rain_amount_mm_prediction.max(),df.next_day_rain_mm.max())/20)*20
y_min=-y_max

kwargs = {
    'cbar': False,
    'linewidths': 0.2,
    'linecolor': 'white',
    'annot': True}
df.columns


df['error']=df['rain_amount_mm_prediction']-df['next_day_rain_mm']
df['date']=df['date'].astype('str')
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


fig.update_traces(marker=dict(size=10,
                              line=dict(width=1, color='black')),
                  selector=dict(mode='markers'))
fig.update_layout(
    title="Rain prediction for Eindhoven",
    xaxis_title="Date",
    yaxis_title="Rain amount",
    legend_title="Legend",
)
fig.update_yaxes(title_text="Rain amount", secondary_y=True)
fig.update_yaxes(range=[y_min,y_max], secondary_y=False)
fig.update_yaxes(range=[y_min,y_max], secondary_y=True)
