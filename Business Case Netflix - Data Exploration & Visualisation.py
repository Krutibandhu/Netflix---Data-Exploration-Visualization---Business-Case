#!/usr/bin/env python
# coding: utf-8

# # Business Case: Netflix - Data Exploration and Visualisation

# About NETFLIX:
# 
#     1. NETFLIX is one of the most popular media and video streaming platforms.
#     2. They have over 10000 movies or tv shows available on their platform, as of mid-2021, they have over 222M Subscribers globally.
#     3. This tabular dataset consists of listings of all the movies and tv shows available on NETFLIX, along with details such as - cast, directors, ratings, release year, duration, etc.

# ![616QXs8yg0L.png](attachment:616QXs8yg0L.png)

# Business Problem:
# 
# Analyze the data and generate insights that could help NETFLIX decide which type of shows/movies to produce and how they can grow the business in different countries.

# The dataset provided to you consists of a list of all the TV shows/movies available on NETFLIX:
#     1. Show_id: Unique ID for every Movie / TV show
#     2. Type: Identifier - A Movie or TV Show
#     3. Title: Title of the Movie / TV Show
#     4. Director: Director of the Movie
#     5. Cast: Actors involved in the movie/show
#     6. Country: Country where the movie/show was produced
#     7. Date_added: Date it was added on Netflix
#     8. Release_year: Actual Release year of the movie/show
#     9. Rating: TV Rating of the movie/show
#     10. Duration: Total Duration - in minutes or number of seasons
#     11. Listed_in: Genre
#     12. Description: The Summary Description

# # Problem Statements:
# 
# 1. How has the number of movies released per year changed over the last 20-30 years?
# 2. Comparison of tv shows vs. movies.
# 3. What is the best time to lunch a TV show?
# 4. Analysis of actors/directors of different types of shows/movies.
# 5. Does Netflix has more focus on TV Shows than movies in recent years?
# 6. Understanding what content is available in different countries.

# # 1.  Analysing Basic Metics

# #Netflix is a popular service that people across the world use for entertainment. 
# #In this Exploratory Analysis and Visualization, 
# #I will explore the netflix dataset through visualizations and graphs using numpy, pandas,matplotlib and seaborn.

# The Aim of this Business case study is to explore and analyze the Netflix shows data after filtering some of the columns. This Netflix movies and TV shows data . We will alter and filter some columns and perform some feature engineering after this we prepare data for analysis.We will perform univariate analysis so we can get a better understanding of every column and then bivariate and multivariate analysis so we will understand the relations between columns. In the end, we will conclude the result of the analysis.

# IMPORTING THE LIBRARIES 

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
import warnings
warnings.filterwarnings('ignore')


# LOADING THE DATASET

# In[2]:


df_A = pd.read_csv("netflix.csv")
df_A


# After the loading dataset we can see that there is 8807 rows and 12 columns. Here also some "NaN" values.

# In[3]:


df_A.head(5) # Top 5


# In[4]:


df_A.tail(5) # Bottom 5


# # 2. Observations:
#     a. Shape of Data
#     b. Data type of all attributes
#     c. missing value detection
#     d. statistcal summary

# In[5]:


df_A.shape # shows like(rows, columns)


# there are 8807 rows and 12 columns.

# In[6]:


df_A.describe(include = 'all') #statistical summary


# In[7]:


df_A.describe() #statistical summary


# It give some statistical summary like Count, Mean, Minimum, Maximum, Standard Deviation, 25%tile, 50%tile, 75%tile.

# In[8]:


df_A.info()


# There are 8807 entries and 12 columns to work with for Exploratory Data Analysis and Visualization. Right off the bat, there are a few columns that contain null values ('director', 'cast', 'country', 'date_added', 'rating')

# In[9]:


df_A.columns # shows all the column names


# In[10]:


df_A.isnull().values.any()


# In[11]:


df_A.isnull().sum().sum()


# In[12]:


sns.heatmap(df_A.isnull(), cbar=False)
plt.title("I----Null Values----I")
plt.show()


# In[13]:


df_A.isnull().sum().sort_values(ascending = False)


# Above in the heatmap and table, we can see that there are many null values in the dataset. There are a total of 4,307 null values across the entire dataset with 2634 missing values in 'director', 825 in 'cast', 831 in 'country', 10 in 'date_added', 4 in 'rating', 3 in duration. We will have to handle all null data values before we can dive into Exploratory data and modeling.

# In[14]:


round(df_A.isnull().sum()/df_A.shape[0]*100,2).sort_values(ascending = False)


# It shows the percentage null values.There is 29.91% null value in director, 9.44% in country, 9.37% in cast, 0.11% in date_added
# 0.05% in rating, 0.03% in duration. 

# # 3.Non Graphical Analysis: Value counts and unique attributes

# In[15]:


df_A["director"].value_counts() # director value count


# In[16]:


df_A.nunique() #unique values


# We can see that for each of the columns, there are alot different unique values for some of them. It makes sense that show_id is large since it is a unique key used to identify a movies And TVshows. title, director, cast, country, date_added, release_year,rating, duration, listed_in, and description contain many unique values as well.

# In[17]:


df_A.country.value_counts() # country value count


# In[18]:


df_A["listed_in"].value_counts() # Genre value count


# In[19]:


df_A["date_added"].value_counts() # date_added value count


# In[20]:


df_A["release_year"].value_counts() # release year value count


# # Some Basic Visual Analysis:

# In[21]:


go.Figure(data = [go.Pie(labels = df_A.type.value_counts(normalize = True).index,
                        values = df_A.type.value_counts(normalize = True).values, hole =.5,
                        title = "Movies Vs Tv Shows")])


# This data set contain around 70% of movies And 30% of TV shows

# In[22]:


df_A["type"].value_counts()


# The total Movies is 6131 & Tv shows 2676.

# In[23]:


df_A["rating"].value_counts()


# In[24]:


sns.barplot(x=df_A.rating.value_counts(), y = df_A.rating.value_counts().index, data = df_A, orient = "h")


# Highest Count: TV-MA is the rating that shows that a program is intended for adults. "MA" stands for "Mature Aduiances". Child aged 17 and younger should not view these programs.
# 
# Followed by Highest(2nd) is the "TV-14". "TV-14" program is meant for children over 14 years of age. It is generally not recommended to let children watchthe program without parental attendance or atleast whithout them vetting it first. It can contain crude humor, the use of harmful substances, strong language, violence, and complex or upsetting themes.
# 
# Third largest TV-PG Parental Guidance Suggested This program contains material that parents may find unsuitable for younger children
# 
# Fourth largest is the very popular "R" ratings. R is the short for restricted, so any young person under 17 should not watch.

# In[25]:


# Top 10 countries of creation Movies & TV shows
df_A.country.value_counts().head(10)


# In[26]:


#year wise count
plt.figure(figsize = (8, 8))
ax = sns.countplot(y = "release_year", data=df_A, order= df_A.release_year.value_counts().index[0:15])


# Highest release in 2018 followed by 2017 and 2019

# In[27]:


# Top 10 directors
df_A.director.value_counts().head(10)


# In[28]:


# Top 15 director
plt.figure(figsize = (13, 8))
ax = sns.countplot(y = "director", data=df_A, order= df_A.director.value_counts().index[0:15])


# In[29]:


# Top 15 Genre
plt.figure(figsize = (13, 8))
bx = sns.countplot(y = "listed_in", data=df_A, order= df_A.listed_in.value_counts().index[0:15])


# # 4. Visual Analysis - Univariate, Bivariate after pre-processingof the Data
# 
# Note: Pre-processing involves unnesting of the data in columns like Actor, Director & Country.

# # Unnesting the Cast Column

# In[30]:


cast_constraint=df_A['cast'].apply(lambda x: str(x).split(', ')).tolist()
df_B = pd.DataFrame(cast_constraint, index = df_A['title']) 
df_B = df_B.stack()
df_B = pd.DataFrame(df_B.reset_index())
df_B.rename(columns={0:'Actors'},inplace=True)
df_B = df_B.drop(['level_1'],axis=1)
df_B.head(15)


# # Unnesting the Director Column

# In[31]:


dir_constraint=df_A['director'].apply(lambda x: str(x).split(', ')).tolist()
df_C = pd.DataFrame(dir_constraint, index = df_A['title']) 
df_C = df_C.stack()
df_C = pd.DataFrame(df_C.reset_index())
df_C.rename(columns={0:'Director'},inplace=True)
df_C = df_C.drop(['level_1'],axis=1)
df_C.head(15)


# # Unnesting the country column

# In[32]:


country_constraint=df_A['country'].apply(lambda x: str(x).split(', ')).tolist()
df_D = pd.DataFrame(country_constraint, index = df_A['title']) 
df_D = df_D.stack()
df_D = pd.DataFrame(df_D.reset_index())
df_D.rename(columns={0:'Country'},inplace=True)
df_D = df_D.drop(['level_1'],axis=1)
df_D.head(15)


# # Unnesting the listed_in column (Genre)

# In[33]:


listed_constraint=df_A['listed_in'].apply(lambda x: str(x).split(', ')).tolist()
df_E = pd.DataFrame(listed_constraint, index = df_A['title']) 
df_E = df_E.stack()
df_E = pd.DataFrame(df_E.reset_index())
df_E.rename(columns={0:'Genre'},inplace=True)
df_E = df_E.drop(['level_1'],axis=1)
df_E.head(10)


# # Collecting the all the unnested dataframes

# In[34]:


df_F = df_B.merge(df_C,on=['title'],how='inner')

df_G = df_F.merge(df_E,on=['title'],how='inner')

df_H = df_G.merge(df_D,on=['title'],how='inner')

df_H.head(10)


# In[35]:


df_H.shape # shows shape like(rows, columns)


# # Merging unnested data with the primary dataframe

# In[36]:


df_A = df_H.merge(df_A[['show_id', 'type', 'title', 'date_added',
       'release_year', 'rating', 'duration']],on=['title'],how='left')
df_A.head(20)


# In[37]:


df_A.shape # shows shape like(rows, columns) 


# Befor merging there is 201991 rows and 5 columns and after merging there are same rows but 11 columns are there. Now the dataset is fine to shows further visualization.

# # Handeling the Null values/ Missing values

# In[38]:


df_A.isnull().sum().sort_values(ascending = False)


# In[39]:


total_null_data = df_A.isnull().sum().sort_values(ascending = False)
percent = ((df_A.isnull().sum()/df_A.isnull().count())*100).sort_values(ascending = False)
print("Total records = ", df_A.shape[0])
Null_data = pd.concat([total_null_data,percent.round(2)],axis=1,keys=['Total Null Data','In % null value'])
Null_data.head(12)


# Above table gives missing values summary in absolute value and in Percentage, date added has the maximum missing values 158 then 67 is rating

# Missing Value Handeling

# In[40]:


df_A['Actors'].replace(['nan'],['Unknown Actor'],inplace=True)
df_A['Director'].replace(['nan'],['Unknown Director'],inplace=True)
df_A['Country'].replace(['nan'],[np.nan],inplace=True)
df_A.head(20)


# In[41]:


total_null = df_A.isnull().sum().sort_values(ascending = False)
percent = ((df_A.isnull().sum()/df_A.isnull().count())*100).sort_values(ascending = False)
print("Total records = ", df_A.shape[0])

missing_data = pd.concat([total_null,percent.round(2)],axis=1,keys=['Total Missing','In Percent'])
missing_data.head(10)


# After replacing string nan with np.nan, actual null values of country went upto 11897 or 5.89 %.

# In[42]:


df_A[df_A.duration.isnull()]


# In[43]:


df_A.loc[df_A['duration'].isnull(),'duration'] = df_A.loc[df_A['duration'].isnull(),'duration'].fillna(df_A['rating'])
df_A.loc[df_A['rating'].str.contains('min', na=False),'rating'] = 'NR'
df_A['rating'].fillna('NR',inplace=True)
df_A.isnull().sum()


# Filling missing values of date added column with mode value with respective release years

# In[44]:


for i in df_A[df_A['date_added'].isnull()]['release_year'].unique():
    date = df_A[df_A['release_year'] == i]['date_added'].mode().values[0]
    df_A.loc[df_A['release_year'] == i,'date_added'] = df_A.loc[df_A['release_year']==i,'date_added'].fillna(date)
df_A[df_A.Country.isna()]


# Filling missing values of country column with mode value with respective directors

# In[ ]:


for i in df_A[df_A['Country'].isnull()]['Director'].unique():
    if i in df_A[~df_A['Country'].isnull()]['Director'].unique():
        country = df_A[df_A['Director'] == i]['Country'].mode().values[0]
        df_A.loc[df_A['Director'] == i,'Country'] = df_A.loc[df_A['Director'] == i,'Country'].fillna(country)
df_A.isnull().sum()


# 

# In[ ]:


for i in df_A[df_A['Country'].isnull()]['Actors'].unique():
    if i in df_A[~df_A['Country'].isnull()]['Actors'].unique():
        imp = df_A[df_A['Actors'] == i]['Country'].mode().values[0]
        df_A.loc[df_A['Actors'] == i,'Country'] = df_A.loc[df_A['Actors']==i,'Country'].fillna(imp) 


# In[ ]:


df_A['Country'].fillna('Unknown Country',inplace=True) 
df_A.isnull().sum().sort_values(ascending = False)


# In[ ]:


df_A.head(10)


# In[ ]:


df_A["date_added"] = pd.to_datetime(df_A['date_added'])


# In[ ]:


df_A['duration'] = df_A['duration'].str.replace("min","")
df_A.head(10)


# In[ ]:


df_A['duration2'] = df_A.duration.copy()
df_ = df_A.copy()


# In[ ]:


df_.loc[df_['duration2'].str.contains('Season'),'duration2'] = 0
df_['duration2'] = df_.duration2.astype('int')
df_.head()


# In[ ]:


df_.duration2.describe()


# In[ ]:


df_.T.apply(lambda x: x.nunique(), axis=1)


# # Univariate analysis of duration column

# In[ ]:


plt.style.use('dark_background')
plt.figure(figsize=(10,2))
sns.displot(df_['duration2'])

plt.show()


# In[ ]:


bins = [-1,1,50,80,100,120,150,200,315]
labels = ['<1','1-50','50-80','80-100','100-120','120-150','150-200','200-315']
df_['duration2'] = pd.cut(df_['duration2'],bins = bins, labels = labels )
df_.head()


# In[ ]:


df_.loc[~df_['duration'].str.contains('Season'),'duration'] = df_.loc[~df_['duration'].str.contains('Season'),'duration2']
df_.drop(['duration2'],axis=1,inplace=True)
df_.head()


# In[ ]:


from datetime import datetime
from dateutil.parser import parse
df_["year_added"] = df_['date_added'].dt.year
df_["year_added"] = df_["year_added"].astype("Int64")
df_["month_added"] = df_['date_added'].dt.month
df_['month_name'] = df_['date_added'].dt.month_name()
df_["month_added"] = df_["month_added"].astype("Int64")
df_["day_added"] = df_['date_added'].dt.day
df_["day_added"] = df_["day_added"].astype("Int64")
df_['Weekday_added'] = df_['date_added'].apply(lambda x: parse(str(x)).strftime("%A"))
df_.head()


# In[ ]:


df_['title'] = df_['title'].str.replace(r"\(.*\)","")
df_.head()


# # Univariate Analysis

# In[ ]:


df_genre=df_.groupby(['Genre']).agg({"title":"nunique"}).reset_index().sort_values(by=['title'],ascending=False)[:15]
plt.figure(figsize=(20,6))
sns.barplot(x = "Genre",y = 'title', data = df_genre)
plt.xticks(rotation = 75)
plt.title('Top 20 Genres')
plt.show()


# In[ ]:


df_pie = df_.groupby(['type']).agg({'title':'nunique'}).reset_index()
df_pie


# In[ ]:


colors = sns.color_palette('dark')[0:9]
plt.figure(figsize=(20,8))

plt.pie(df_pie['title'], labels = df_pie['type'], colors = colors, autopct='%.0f%%')
plt.title('% of Movies VS TV shows')
plt.show()


# In[ ]:


df_['Country'] = df_['Country'].str.replace(',', '')
df_.head()


# In[ ]:


df_country = df_.groupby(['Country']).agg({'title':'nunique'}).reset_index().sort_values(by=['title'],ascending=False)[:10]
plt.figure(figsize=(15,6))
sns.barplot(y = "Country",x = 'title', data = df_country)
plt.xticks(rotation = 60)
plt.title('Top 10 Countries for content creation')
plt.show()


# In[ ]:


df_rating = df_.groupby(['rating']).agg({'title':'nunique'}).reset_index().sort_values(by=['title'],ascending=False)[:10]

plt.figure(figsize=(15,6))
sns.barplot(y = "rating",x = 'title', data = df_rating)
plt.xticks(rotation = 60)
plt.title('Top 10 rating types')
plt.show()


# In[ ]:


df_duration = df_.groupby(['duration']).agg({'title':'nunique'}).reset_index().sort_values(by=['title'],ascending=False)[:10]

plt.figure(figsize=(15,6))
sns.barplot(y = "duration",x = 'title', data = df_duration)
plt.xticks(rotation = 60)
plt.title('Top 10 duaration categories')
plt.show()


# In[ ]:


df_actors = df_.groupby(['Actors']).agg({'title':'nunique'}).reset_index().sort_values(by=['title'],ascending=False)[:15]
df_actors = df_actors[df_actors['Actors']!='Unknown Actor']
plt.figure(figsize=(15,6))
sns.barplot(y = "Actors",x = 'title', data = df_actors )
plt.xticks(rotation = 60)
plt.title('Top 15 most popular Actors')
plt.show()


# In[ ]:


df_directors = df_.groupby(['Director']).agg({'title':'nunique'}).reset_index().sort_values(by=['title'],ascending=False)[:15]
df_directors = df_directors[df_directors['Director']!='Unknown Director']
plt.figure(figsize=(15,6))
sns.barplot(y = "Director",x = 'title', data = df_directors )
plt.xticks(rotation = 60)
plt.title('Top 15 most popular Directors')
plt.show


# In[ ]:


df_.head(15)


# In[ ]:


df_year = df_.groupby(['year_added']).agg({'title':'nunique'}).reset_index()

plt.figure(figsize=(15,6))
sns.barplot(x = "year_added", y = 'title', data = df_year, color = 'red')
plt.xticks(rotation = 60)
plt.title('Trends of movies or TV shows across the years')
plt.show()


# In[ ]:


fig = plt.figure(figsize = (20,7))

plt.style.use('dark_background')
sns.countplot(data = df_,x = 'year_added',hue = 'type',palette ="Reds_r")
plt.title('Movies and TV Shows added added to Netflix by date ', fontsize=14)


# In[ ]:


df_month = df_.groupby(['month_name', 'type']).agg({'title':'nunique'}).reset_index()

plt.figure(figsize=(15,6))
sns.lineplot(x = "month_name",y = 'title', data = df_month, color = 'red', hue = df_month.type )
plt.xticks(rotation = 60)
plt.title('movies/ TV shows added across the months')
plt.show


# In[ ]:


df_month = df_.groupby(['month_name']).agg({'title':'nunique'}).reset_index()

plt.figure(figsize=(15,6))
sns.lineplot(x = "month_name",y = 'title', data = df_month, color = 'yellow' )
plt.xticks(rotation = 60)
plt.title('movies/ TV shows added across the months')
plt.show


# In[ ]:


df_day = df_.groupby(['day_added']).agg({'title':'nunique'}).reset_index()

plt.figure(figsize=(15,6))
sns.barplot(x = "day_added",y = 'title', data = df_day, color = 'white' )
plt.xticks(rotation = 60)
plt.title('movies/ TV shows added across each day')
plt.show


# In[ ]:


df_weekday = df_.groupby(['Weekday_added', 'type']).agg({'title':'nunique'}).reset_index()

plt.figure(figsize=(15,6))
sns.lineplot(x = "Weekday_added",y = 'title', data = df_weekday, color = 'blue' , hue = df_weekday.type)
plt.xticks(rotation = 78)
plt.title('movies/ TV shows added across weekdays')
plt.show


# In[ ]:


df_weekday = df_.groupby(['Weekday_added']).agg({'title':'nunique'}).reset_index()

plt.figure(figsize=(25,11))
sns.lineplot(x = "Weekday_added",y = 'title', data = df_weekday, color = 'white' )
plt.xticks(rotation = 60)
plt.title('movies/ TV shows added across weekdays')
plt.show


# In[ ]:


df_.columns


# In[ ]:


plt.figure(figsize=(15,6))
sns.boxplot(x='type', y='release_year', data=df_, )
sns.despine(left=True)
plt.title('Type of Show by Release Date')
plt.ylim(2000,2020)


# # Bivariate Analysis

# In[ ]:


month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 
               'October', 'November', 'December']
content = df_.groupby('year_added')['month_name'].value_counts().unstack().fillna(0)[month_order].T

plt.figure(figsize=(20,8))
plt.title("Number of months' content added per year")
sns.heatmap(content , cmap = 'Reds')
plt.show()


# In[ ]:


plt.figure(figsize = (25,8))
sns.scatterplot(y = df_.index , x = df_.release_year , hue = df_.type , palette='Set2')


# In[ ]:


df_.groupby(['day_added']).agg({"title":"nunique"})


# # Univariate Analysis separately for shows and movies

# In[ ]:


df_shows = df_[df_['type']=='TV Show']
df_movies = df_[df_['type']=='Movie']


# In[ ]:


df_genre = df_movies.groupby(['Genre']).agg({"title":"nunique"}).reset_index().sort_values(by=['title'],ascending=False)[:15]
plt.figure(figsize = (15,6))
sns.barplot(y = "Genre",x = 'title', data = df_genre)
plt.xticks(rotation = 60)
plt.title('Top 15 Genres Movies')
plt.show() # movies


# In[ ]:


df_genre = df_shows.groupby(['Genre']).agg({"title":"nunique"}).reset_index().sort_values(by=['title'],ascending=False)[:15]
plt.figure(figsize = (15,6))
sns.barplot(y = "Genre",x = 'title', data = df_genre)
plt.xticks(rotation = 60)
plt.title('Top 15 Genres Tv shows')
plt.show() # Tv shows


# In[ ]:


df_country = df_shows.groupby(['Country']).agg({'title':'nunique'}).reset_index().sort_values(by=['title'],ascending=False)[:10]
plt.figure(figsize=(25,13))
sns.barplot(y = "Country",x = 'title', data = df_country)
plt.xticks(rotation = 60)
plt.title('Top 10 Countries for content creation')
plt.show()


# In[ ]:


df_country = df_movies.groupby(['Country']).agg({'title':'nunique'}).reset_index().sort_values(by=['title'],ascending=False)[:10]
plt.figure(figsize=(25,13))
sns.barplot(y = "Country",x = 'title', data = df_country)
plt.xticks(rotation = 60)
plt.title('Top 10 Countries for content creation')
plt.show()


# United States is leading across both TV Shows and Movies, UK also provides great content across TV Shows and Movies. Surprisingly India is much more prevalent in Movies as compared TV Shows.
# 
# Moreover the number of Movies created in India outweigh the sum of TV Shows and Movies across UK since India was rated as second in net sum of whole content across Netflix.

# # Business Insights
# 
# 
# 1. Over the years both TV shows and movie contents addtion has increased till 2020, but after 2020 its started declining may be due to Covid relief, number of Movies added is more compare to TV shows over the years.
# 
# 2. Most of the content get added in december and july month, for day wise, Friday is the best day followed by Thursday.
# 
# 3. It was evident that 1st of every month was when the most content was added.
# 
# 4. Anupam Kher,SRK,Julie Tejwani, Naseeruddin Shah and Takahiro Sakurai occupy the top stop in Most Watched content.
# 
# 5. Rajiv Chilaka, Jan Suter and Raul Campos are the most popular directors across Netflix. Rajiv Chilaka director producing more movies.
# 
# 6. Netflix is more focusing on movies compare to TV shows.
# 
# 7. There is a 70% & 30% of Movies and TV Shows content in Netflix platform.
# 
# 8. International Movies, Dramas and Comedies are the most popular are most popular Genre.
# 
# 9. US,India,UK,Canada and France are leading countries in Content Creation on Netflix.
# 
# 10. Most of the highly rated content on Netflix is intended for TV - Mature Audiences(TV-MA)
# 
# 11. The duration of Most Watched content in our whole data is 80-120 mins. These must be movies and Shows having only 1 Season.
# 
# 12. United States is leading across both TV Shows and Movies, UK also provides great content across TV Shows and Movies. Surprisingly India is much more prevalent in Movies as compared TV Shows.
# 
# 13. India has second position in creating movies & UK has 2nd in Tv shows as well

# # Recommendations
# 
# 1. The most popular Genres across the countries and in both TV Shows and Movies are Drama, Comedy and International TV Shows/Movies, so recommended to generate more content on these genres.
# 
# 2. Add TV Shows/ movies in the month of July 1st / August 1st.
# 
# 3. Add movies for Indian Audience, it has been declining since 2018.
# 
# 4. While creating content, take into consideration the popular actors/directors for that country. Also take into account the director-actor combination which is highly recommended.
# 
# 5. 80-120 mins is avagrage watch by audiance so eye on this also towads creating the content.
# 
# 6. As per Most Rating TV-MA so we can create more content according to this followed by TV-14, TV-PG & R. 
