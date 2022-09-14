import streamlit as st
import pickle
import pandas as pd

# connecting python with Neo4j Graph DataBase
from neo4j import GraphDatabase

url = 'bolt://localhost:7687'
user = 'neo4j'
password = 'password'
driver = GraphDatabase.driver(url, auth=(user, password))


# function returns the DataFrame based on query and parameters passes
def fetch_data(query, params={}):
    with driver.session() as session:
        result = session.run(query, params)
        return pd.DataFrame([r.values() for r in result], columns=result.keys())


# importing necessary dataFiles using Pickle library

movies_dict = pickle.load(open('movies_dict.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)

users_dict = pickle.load(open('users_dict.pkl', 'rb'))
users = pd.DataFrame(users_dict)

ratings_dict = pickle.load(open('ratings_dict.pkl', 'rb'))
ratings = pd.DataFrame(ratings_dict)

# The title of the Website
st.title('Movie Recommender System')

# Select box for selecting the userId for whom we recommend movies
selected_user_id = st.selectbox('Select the UserId', users['userId'].values)

# Retrieve the top-rated and last seen movies by selected user
ratings_df = ratings[ratings['userId'] == selected_user_id]
ratinggs = ratings_df['rating']
n = ratinggs.quantile(0.90)
ratings_df = ratings_df[ratings_df['rating'] >= n]
if len(ratings_df) > 10:
    recent_ids = ratings_df.nlargest(10, 'timestamp')
    movie_ids = pd.Series.to_list(recent_ids['movieId'])
    m = movies[movies.id.isin(movie_ids)].title.values

else:
    movie_ids = pd.Series.to_list(ratings_df['movieId'])
    m = movies[movies.id.isin(movie_ids)].title.values
for i in m:
    st.write(i)

# select the movie from last seen and top-rated movies by selected user
selected_movie_name = st.selectbox('Select the movie', movies['title'].values)

# pick a number for No of recommendations we want
number = st.slider("Pick a number", 0, 20)

# Button for recommendation based on movie
if st.button('Movie Based'):
    query = '''MATCH (m:Movie {title: $i })-[:IN_GENRE|ACTED_IN|DIRECTED]-(t)-[:IN_GENRE|ACTED_IN|DIRECTED]-(other:Movie)
    WITH m, other, COUNT(t) AS intersection, COLLECT(t.name) AS i
    MATCH (m)-[:IN_GENRE|ACTED_IN|DIRECTED]-(mt)
    WITH m,other, intersection,i, COLLECT(mt.name) AS s1
    MATCH (other)-[:IN_GENRE|ACTED_IN|DIRECTED]-(ot)
    WITH m,other,intersection,i, s1, COLLECT(ot.name) AS s2

    WITH m,other,intersection,s1,s2

    WITH m,other,intersection,s1+[x IN s2 WHERE NOT x IN s1] AS union, s1, s2

    RETURN other.title AS recommendation, ((1.0*intersection)/SIZE(union)) AS jaccard ORDER BY jaccard DESC LIMIT $number'''
    result = fetch_data(query, {'i': selected_movie_name, 'number': number})
    a = result['recommendation'].values
    for i in a:
        st.write(i)

# Recommend movies based content filtering
if st.button('Content Based'):
    cont_df = ratings[ratings['userId'] == selected_user_id]
    cont_df = cont_df.sort_values(by=['rating'], ascending=False)
    top_movies = cont_df['movieId'][:5].values
    cont = pd.DataFrame(columns=['movieId', 'title', 'score'])
    query = '''MATCH (m:Movie {movieId: $i})-[:IN_GENRE|ACTED_IN|DIRECTED]-(t)-[:IN_GENRE|ACTED_IN|DIRECTED]-(other:Movie)
        WITH m, other, COUNT(t) AS intersection, COLLECT(t.name) AS i
        MATCH (m)-[:IN_GENRE|ACTED_IN|DIRECTED]-(mt)
        WITH m,other, intersection,i, COLLECT(mt.name) AS s1
        MATCH (other)-[:IN_GENRE|ACTED_IN|DIRECTED]-(ot)
        WITH m,other,intersection,i, s1, COLLECT(ot.name) AS s2

        WITH m,other,intersection,s1,s2

        WITH m,other,intersection,s1+[x IN s2 WHERE NOT x IN s1] AS union, s1, s2

        RETURN other.movieId AS movieId,other.title as title,((1.0*intersection)/SIZE(union)) AS score ORDER BY score DESC LIMIT 5'''

    for movie in top_movies:
        result = fetch_data(query, {'i': int(movie)})
        cont = pd.concat([cont, result], ignore_index=True)
    cont = cont.sort_values(by=['score'], ascending=False, ignore_index=True)[:number]
    l = cont['title'].values
    for movie in l:
        st.write(movie)

# Recommend movies based on collaborative filtering
if st.button('Collaborative'):
    query = '''MATCH (p1:User{userId:$i})-[x:RATED]->(m:Movie)<-[y:RATED]-(p2:User)
        WITH COUNT(m) AS numbermovies, SUM(x.rating * y.rating) AS xyDotProduct,
        SQRT(REDUCE(xDot = 0.0, a IN COLLECT(x.rating) | xDot + a^2)) AS xLength,
        SQRT(REDUCE(yDot = 0.0, b IN COLLECT(y.rating) | yDot + b^2)) AS yLength,
        p1, p2 WHERE numbermovies > 10
        WITH p1, p2, xyDotProduct / (xLength * yLength) AS sim
        ORDER BY sim DESC
        LIMIT 10

        MATCH (p2)-[r:RATED]->(m:Movie) WHERE NOT EXISTS( (p1)-[:RATED]->(m) )

        RETURN p1.userId as userId,m.movieId AS movieId,m.title AS title, SUM( sim * r.rating) AS score
        ORDER BY score DESC LIMIT $rec'''
    result = fetch_data(query, {'i': int(selected_user_id), 'rec': int(number)})
    l = result['title'].values
    for movie in l:
        st.write(movie)

# Recommend movies based on GNN
gnn_dict = pickle.load(open('gnn_dict.pkl', 'rb'))
gnn = pd.DataFrame(gnn_dict)
if st.button('GNN'):
    result = gnn[gnn['Userid'] == selected_user_id]['Title'][:number]
    for movie in result:
        st.write(movie)
