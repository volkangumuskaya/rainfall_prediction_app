import streamlit as st
import pandas as pd
import math
from pathlib import Path

# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='GDP dashboard here',
    page_icon=':earth_americas:', # This is an emoji shortcode. Could be a URL too.
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
    return tmp_df

df = get_df()

# -----------------------------------------------------------------------------
# Draw the actual page

# Set the title that appears at the top of the page.
'''
# :earth_americas: Daily predictions

Some explanations here
'''

# Add some spacing
''
''

min_value = df['date'].min()
max_value = df['date'].max()



from_year, to_year = st.slider(
    'Which dates are you interested in?',
    min_value=min_value,
    max_value=max_value,
    value=[min_value, max_value])

stations = df['station'].unique()

if not len(stations):
    st.warning("Select at least one station")

selected_stations = st.multiselect(
    'Which st would you like to view?',
    stations,
    [370])

''
''
''

# Filter the data
filtered_df = df[
    (df['station'].isin(selected_stations))
    & (df['date'] <= to_year)
    & (from_year <= df['date'])
]

st.header('GDP over time', divider='gray')

''

st.line_chart(
    filtered_df,
    x='date',
    y='rain_amount_mm_prediction',
    color='month_name',
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
