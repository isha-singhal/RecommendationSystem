import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# helper functions
def get_title_from_index(index):
	return df[df.index == index]["title"].values[0]

def get_index_from_title(title):
	return df[df.title == title]["index"].values[0]


##Step 1: Read CSV File
df= pd.read_csv("movie_dataset.csv") #dataframe
#print(df.columns)  #this dataset has these many features

#print(df.describe())
#plt.figure(figsize=(5,5))
#plt.bar(list(df['original_language'].value_counts()[0:5].keys()),list(df['original_language'].value_counts()[0:5]),color="r")
#plt.show()    

##Step 2: Select Features
features=['keywords','cast','genres','director']

##Step 3: Create a column in DF which combines all selected features
for feature in features:
	df[feature]=df[feature].fillna(' ')

def combine_features(row):
	try:
		return row['keywords']+" "+row['cast']+" "+row['genres']+" "+row['director']
	except:
		return "Error:",row
df["combined_features"]=df.apply(combine_features,axis=1)
#print("Combined Features:",df["combined_features"].head())

##Step 4: Create count matrix from this new combined column
cv=CountVectorizer()
count_matrix=cv.fit_transform(df["combined_features"])

##Step 5: Compute the Cosine Similarity based on the count_matrix
similarity_scores=cosine_similarity(count_matrix)
movie_user_likes = "Avatar"

## Step 6: Get index of this movie from its title
movie_index = get_index_from_title(movie_user_likes)
similar_movies =  list(enumerate(similarity_scores[movie_index]))

## Step 7: Get a list of similar movies in descending order of similarity score
sorted_similar_movies = sorted(similar_movies,key=lambda x:x[1],reverse=True)

## Step 8: Print titles of first 5 movies that are similar to given movie
i=0
print("Top 5 similar movies to "+movie_user_likes+" are:\n")
for element in sorted_similar_movies:
		print(get_title_from_index(element[0]))
		i=i+1
		if i>5:
			break

#recommending based on popularity or rating of the movies
sort_by_average_vote = sorted(sorted_similar_movies,key=lambda x:df["vote_average"][x[0]],reverse=True)

i=0
print("\nSuggesting top 5 movies in order of Average Votes:\n")
for element in sort_by_average_vote:
    print(get_title_from_index(element[0]))
    i=i+1
    if i>5:
        break