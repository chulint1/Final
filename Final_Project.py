# Rubric item focusing on in this homework: pandas, scikit-learn/Keras and Altair

import streamlit as st
import numpy as np
import pandas as pd
import sklearn
from pandas.api.types import is_numeric_dtype
import altair as alt


st.title('The ranking of Universities')

# Read Data

df = pd.read_csv("cwurData.csv")

# Manipulate Dataframe

df2 = df[df['year'] == 2015]
for i in df2.columns[4:-2]:
    df2 = df2[df2[i] < df2[i].max()]

df3 = df2[df2.notna().all(axis=1)].copy()
numeric_cols = [c for c in df3.columns if is_numeric_dtype(df3[c])]


# Scikit-learn

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df4 = df3[numeric_cols]
scaler = StandardScaler()
scaler.fit(df4)
df5 = pd.DataFrame(scaler.transform(df4), columns = df4.columns)

newkmeans = KMeans(10)
newkmeans.fit(df5)

newkmeans.predict(df5)

df3["Clusters"] = newkmeans.predict(df5)

# Select x-axis and y-axis from st.selectbox
x_axis = st.selectbox('Choose an x-value', numeric_cols)
y_axis = st.selectbox('Choose a y-value', numeric_cols)

row = st.slider('Select the rows you want plotted:', 0, len(df3), (0, int(len(df3)/2)))
st.write(f'You are looking at row {row}')

# Create the Chart
df_new = df3[row[0]:row[1]]
cha = alt.Chart(df_new).mark_circle().encode(
    x=x_axis,
    y=y_axis,
    color="Clusters:N",
    tooltip=["institution", "country"]
)

st.altair_chart(cha, use_container_width=True)
# Keras


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # suppress most warnings in TensorFlow
import tensorflow as tf
from tensorflow import keras
from keras import layers

if 'model_saved' in st.session_state:
    model = st.session_state['model_saved']
else:

# Importing Data

    data = df5

    x = data[data.columns[2:-2]]

    y = data.iloc[:,0]

    # Building the Model

    model = keras.Sequential()

    model.add(layers.Dense(1,input_dim = 8))

    model.summary()

    model.compile(loss="mse",
                  optimizer='adam'
                 )

    # Train the Data

    model.fit(x,y,epochs=200)
    st.session_state['model_saved'] = model


import plotly as py
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)

all_countries = sorted(set(list(df_new["country"])))

# input: a country. output: how many universities
def count_univs(u):
    return len(df_new[df_new["country"].map(lambda u_list: u in u_list)])

dic= {u:count_univs(u) for u in all_countries}

df_ = pd.DataFrame(list(dic.items()),columns = ['country','# of Univs'])

# Create the Choropleth
fig = go.Figure(data=go.Choropleth(
    locations = df_['country'],
    locationmode = 'country names',
    z = df_['# of Univs'],
    colorscale = 'Greens',
    marker_line_color = 'black',
    marker_line_width = 0.5,
))
fig.update_layout(
    title_text = 'Number of Universities across the world',
    title_x = 0.5,
    geo=dict(
        showframe = False,
        showcoastlines = False,
        projection_type = 'equirectangular'
    )
)
st.plotly_chart(fig, use_container_width=True)

st.write('Referencing：https://towardsdatascience.com/visualizing-the-coronavirus-pandemic-with-choropleth-maps-7f30fccaecf5')

U_name = st.text_input('Enter the University you are looking for', 'Harvard')
df_u = df3[df3["institution"].map(lambda g_list: U_name in g_list)]
st.write('World Ranking: ', df_u)

C_name = st.text_input('Enter the Country you are looking for', 'USA')
df_c = df3[df3["country"].map(lambda g_list: C_name in g_list)]
st.write('World Ranking: ', df_c)

st.write('Making prediction about the world rank of the university created on your own:')
col1,col2,col3,col4=st.columns(4)
col5,col6,col7,col8=st.columns(4)
quality_of_education= col1.number_input('Quality of education',1,367,1)
alumni_employment = col2.number_input('Alumni employment',1,367,1)
quality_of_faculty = col3.number_input('Quality of faculty',1,367,1)
publications = col4.number_input('Publications',1,367,1)
influence = col5.number_input('Influence',1,367,1)
citations = col6.number_input('Citations',1,367,1)
broad_impact = col7.number_input('Broad impact',1,367,1)
patents = col8.number_input('Patents',1,367,1)
# Making Predictions

# The world ranking for a given university which has the ranking of:
#  in quality_of_education
#  in alumni_employment
#  in quality_of_faculty
#  in publications
#  in influence
#  in citations
#  in broad_impact
#  in patents

prediction = int(model.predict(pd.DataFrame([[quality_of_education,
                             alumni_employment,
                             quality_of_faculty,
                             publications,
                             influence,
                             citations,
                             broad_impact,
                             patents]])))
st.write(f'Your World Ranking is {prediction}')
## Link to the dataset :  https://www.kaggle.com/mylesoneill/world-university-rankings
st.write('Link to the dataset :  https://www.kaggle.com/mylesoneill/world-university-rankings')

