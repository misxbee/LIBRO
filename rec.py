#!/usr/bin/env python
# coding: utf-8

# ### Step 1: Import Dependencies
# 
# We are using [pandas.DataFrame](http://pandas.pydata.org/pandas-docs/version/0.19/generated/pandas.DataFrame.html) to represent our data. We will visualize our data with [matplotlib](https://matplotlib.org/) and [seaborn](https://seaborn.pydata.org/).
# 
# 

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ### Step 2: Load Data
# 
# #### Files used
# - goodreads_data.csv
# - ratings.csv

# In[2]:


ratings = pd.read_csv('ratings.csv')


# In[3]:


books = pd.read_csv('goodreads_data.csv')


# #### filter ratings that are only relevant

# In[4]:


ratings= ratings[ratings['bookId'].between(1, 9999)]


# ### Step 3: Exploratory Data Analysis

# In[5]:


n_ratings = len(ratings)
n_book = ratings['bookId'].nunique()
n_users = ratings['userId'].nunique()

print(f"Number of ratings: {n_ratings}")
print(f"Number of unique bookId's: {n_book}")
print(f"Number of unique users: {n_users}")
print(f"Average number of ratings per user: {round(n_ratings/n_users, 2)}")
print(f"Average number of ratings per book: {round(n_ratings/n_book, 2)}")


# ### What is the distribution of book ratings?

# In[6]:


sns.countplot(x="rating", data=ratings, palette="viridis")
plt.title("Distribution of book ratings", fontsize=14)
plt.show()


# In[7]:


print(f"Mean global rating: {round(ratings['rating'].mean(),2)}.")

mean_ratings = ratings.groupby('userId')['rating'].mean()
print(f"Mean rating per user: {round(mean_ratings.mean(),2)}.")


# ### Which book are most frequently rated?

# In[8]:


book_ratings = ratings.merge(books, on='bookId')
book_ratings['Book'].value_counts()[0:10]


# ### What are the lowest and highest rated books?
# 
# 

# In[9]:


mean_ratings = ratings.groupby('bookId')[['rating']].mean()
lowest_rated = mean_ratings['rating'].idxmin()
books[books['bookId']==lowest_rated]


# In[10]:


highest_rated = mean_ratings['rating'].idxmax()
books[books['bookId'] == highest_rated]


# In[11]:


ratings[ratings['bookId']==highest_rated]


# #### Bayesian Average
# 
# [Bayesian Average](https://en.wikipedia.org/wiki/Bayesian_average) is defined as:
# 
# $r_{i} = \frac{C \times m + \Sigma{\text{reviews}}}{C+N}$
# 
# where $C$ represents our confidence, $m$ represents our prior, and $N$ is the total number of reviews for books $i$. In this case, our prior $m$ will be the average mean rating across all books. By defintion, C represents "the typical data set size". Let's make $C$ be the average number of ratings for a given book.

# In[12]:


book_stats = ratings.groupby('bookId')['rating'].agg(['count', 'mean'])
book_stats.head()


# In[13]:


C = book_stats['count'].mean()
m = book_stats['mean'].mean()

print(f"Average number of ratings for a given book: {C:.2f}")
print(f"Average rating for a given book: {m:.2f}")

def bayesian_avg(ratings):
    bayesian_avg = (C*m+ratings.sum())/(C+ratings.count())
    return round(bayesian_avg, 3)


# In[14]:


bayesian_avg_ratings = ratings.groupby('bookId')['rating'].agg(bayesian_avg).reset_index()
bayesian_avg_ratings.columns = ['bookId', 'bayesian_avg']
book_stats = book_stats.merge(bayesian_avg_ratings, on='bookId')
book_stats.head()


# In[15]:


book_stats = book_stats.merge(books[['bookId', 'Book']])
book_stats.sort_values('bayesian_avg', ascending=False).head()


# In[16]:


book_stats.sort_values('bayesian_avg', ascending=True).head()


# The Goodreads dataset needs to be cleaned in two ways:
# 
# - `genres` is expressed as a string  separating each genre. We will manipulate this string into a list, which will make it much easier to analyze.
# 

# In[17]:


from collections import Counter

genre_frequency = Counter()
for book in books["Genres"]:
  for genre in eval(book):
    genre_frequency[genre] += 1

# Print each genre and its frequency
for genre, count in genre_frequency.items():
    print(f"{genre}: {count}")


# In[18]:


print("The 5 most common genres: \n", genre_frequency.most_common(5))


# ### Step 4: Data Pre-processing
# 
# We are going to use a technique called colaborative filtering to generate recommendations for users. This technique is based on the premise that similar people like similar things.
# 
# The first step is to transform our data into a user-item matrix, also known as a "utility" matrix. In this matrix, rows represent users and columns represent books. The beauty of collaborative filtering is that it doesn't require any information about the users or the books user to generate recommendations.
# 
# 

# The `create_X()` function outputs a sparse matrix $X$ with four mapper dictionaries:
# 
# - **user_mapper**: maps user id to user index
# - **b_mapper**: maps movie id to book index
# - **user_inv_mapper**: maps user index to user id
# - **b_inv_mapper**: maps movie index to book id
# 
# We need these dictionaries because they map which row/column of the utility matrix corresponds to which user/movie id.
# 
# 

# In[19]:


from scipy.sparse import csr_matrix

def create_X(df):

    M = df['userId'].nunique()
    N = df['bookId'].nunique()

    user_mapper = dict(zip(np.unique(df["userId"]), list(range(M))))
    b_mapper = dict(zip(np.unique(df["bookId"]), list(range(N))))

    user_inv_mapper = dict(zip(list(range(M)), np.unique(df["userId"])))
    b_inv_mapper = dict(zip(list(range(N)), np.unique(df["bookId"])))

    user_index = [user_mapper[i] for i in df['userId']]
    item_index = [b_mapper[i] for i in df['bookId']]

    X = csr_matrix((df["rating"], (user_index,item_index)), shape=(M,N))

    return X, user_mapper, b_mapper, user_inv_mapper, b_inv_mapper

X, user_mapper, b_mapper, user_inv_mapper, b_inv_mapper = create_X(ratings)


# In[20]:


X.shape
# Convert sparse matrix to dense array (output of pivot table)
X_dense = X.toarray()

# Create DataFrame from dense array
X_df = pd.DataFrame(X_dense, columns=[b_inv_mapper[i] for i in range(X_dense.shape[1])])

# Set index names
X_df.index = [user_inv_mapper[i] for i in range(X_dense.shape[0])]

# Print the DataFrame
print(X_df)


# Our `X` matrix contains 610 users and 9724 books

# ### Evaluating sparsity
# 
# Here, we calculate sparsity by dividing the number of stored elements by total number of elements. The number of stored (non-empty) elements in our matrix ([nnz](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.nnz.html)) is equivalent to the number of ratings in our dataset.

# In[21]:


n_total = X.shape[0]*X.shape[1]
n_ratings = X.nnz
sparsity = n_ratings/n_total
print(f"Matrix sparsity: {round(sparsity*100,2)}%")


# In[22]:


n_ratings_per_user = X.getnnz(axis=1)
len(n_ratings_per_user)


# In[23]:


print(f"Most active user rated {n_ratings_per_user.max()} book.")
print(f"Least active user rated {n_ratings_per_user.min()} book.")


# In[24]:


n_ratings_per_book = X.getnnz(axis=0)
len(n_ratings_per_book)


# ### Step 5: User-item Recommendations with k-Nearest Neighbors

# We are going to find the $k$ movies that have the most similar user engagement vectors for books $i$.

# In[25]:


import numpy as np
from sklearn.neighbors import NearestNeighbors

def find_similar_books(b_id, X, b_mapper, b_inv_mapper, k, metric='cosine'):
    # Transpose X if it's not already transposed
    X = X.T if X.shape[0] < X.shape[1] else X

    neighbour_ids = []

    # Get index of the given book ID
    b_ind = b_mapper[b_id]
    b_vec = X[b_ind]

    if isinstance(b_vec, (np.ndarray)):
        b_vec = b_vec.reshape(1, -1)

    # Use k+1 since kNN output includes the bookId of interest
    kNN = NearestNeighbors(n_neighbors=k+1, algorithm="brute", metric=metric)
    kNN.fit(X)
    neighbour = kNN.kneighbors(b_vec, return_distance=False)

    # Extract neighbor IDs
    for i in range(0, k):
        n = neighbour.item(i)
        neighbour_ids.append(b_inv_mapper[n])

    # Remove the input book ID itself
    neighbour_ids.pop(0)

    return neighbour_ids


# In[54]:


book_titles = dict(zip(books['bookId'], books['Book']))
print("Input book ID Except 33,35,37,51,56,59,67,84,90,91,96,98,109,114,115,117,120 since they are not in dataset")
book_id = int(input("Enter book ID between 1 and 120 for collaborative filtering recommendation."))

book_sim= find_similar_books(book_id, X,  b_mapper, b_inv_mapper, metric='euclidean', k=10)
book_title = book_titles[book_id]

print(f"Because you watched {book_title}:")
for i in book_sim:
    print(book_titles[i])


# ## RESULTS

# 
# The Mean Absolute Error (MAE) is a measure of the average absolute errors between predicted and actual values. In the context of your recommendation system, the MAE of approximately 3.43 means that, on average, your model's predictions for book ratings are off by around 3.43 rating points when compared to the actual ratings.
# 
# Here's what each part means:
# 
# Mean: This indicates that we're taking the average of the absolute errors.
# 
# Absolute Error: This is the absolute difference between the predicted rating and the actual rating for each book. The absolute difference is used to ensure that overestimates and underestimates contribute equally to the error.
# 
# MAE Value: The MAE value itself (3.43 in your case) represents the average absolute difference between predicted and actual ratings across all the test cases.
# 
# 
# In summary, a lower MAE value indicates better predictive performance. A MAE of 3.43 suggests that there's room for improvement in the accuracy of your recommendation system. You can further refine your model or try different algorithms to reduce this error.

# In[27]:


from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
def evaluate_recommendations(X, test_size=0.2, k=10, metric='cosine'):
    # Split data into train and test sets
    X_train, X_test = train_test_split(X, test_size=test_size)

    predictions = []
    actuals = []
    for i in range(1, X_test.shape[0] + 1):
        # Check if book ID exists in the mapper
        if i in b_inv_mapper:
            if i in [33,35,37,51,56,59,67,84,90,91,96,98,109,114,115,117,120]:#list of id that is not in dataset
                continue
            # For each test data point (book), find similar books
            similar_books = find_similar_books(i, X_train, b_mapper, b_inv_mapper, k, metric)
            
            # Assuming ratings are available, predict the ratings for similar books
            predicted_ratings = [X_train[i - 1, b_mapper[b_id]] for b_id in similar_books]

            # Consider actual ratings for evaluation
            actual_rating = X_test[i - 1, :].toarray().flatten()

            # Remove zero ratings (not rated)
            actual_rating = actual_rating[actual_rating != 0]

            # If actual ratings exist, calculate MAE
            if len(actual_rating) > 0:
                predictions.append(np.mean(predicted_ratings))
                actuals.append(np.mean(actual_rating))
        else:
            print(f"Book ID {i} not found in the mapper.")

    mae = mean_absolute_error(actuals, predictions)
    return mae


X, user_mapper, b_mapper, user_inv_mapper, b_inv_mapper = create_X(ratings)

# Evaluate recommendations
mae = evaluate_recommendations(X)
print("Mean Absolute Error (MAE):", mae)


# Note that these recommendations are based solely on user-item ratings. Movie features such as genres are not used in this approach.

# You can also play around with the kNN distance metric and see what results you would get if you use "manhattan" or "euclidean" instead of "cosine".

# ### Step 6: Handling the cold-start problem
# 
# Collaborative filtering relies solely on user-item interactions within the utility matrix. The issue with this approach is that brand new users or items with no iteractions get excluded from the recommendation system. This is called the **cold start problem**. Content-based filtering is a way to handle this problem by generating recommendations based on user and item features.
# 
# First, we need to convert the `genres` column into binary features. Each genre will have its own column in the dataframe, and will be populated with 0 or 1.

# In[28]:


n_book = books['bookId'].nunique()
print(f"There are {n_book} unique movies in our movies dataset.")


# In[29]:


import ast
# Function to convert string representation of list to actual list
def convert_to_list(string_list):
    return ast.literal_eval(string_list)

# Convert 'Genres' column to list
books['Genres'] = books['Genres'].apply(convert_to_list)


# In[30]:


books.head()


# In[42]:


import pandas as pd



# Get unique genres from the "Genres" column
genres = set(g for G in books['Genres'] for g in G)

# Iterate over each genre and add a binary column for it
for genre in genres:
    books[genre] = books['Genres'].apply(lambda x: int(genre in x))

# Drop unnecessary columns
book_genres = books.drop(columns=['bookId', 'Book', 'Genres', 'Author'])



# In[32]:


book_genres.head()
#incidence matrix


# In[33]:


from sklearn.metrics.pairwise import cosine_similarity

cosine_sim = cosine_similarity(book_genres, book_genres)
print(f"Dimensions of our genres cosine similarity matrix: {cosine_sim.shape}")


# In[34]:


from fuzzywuzzy import process

def book_finder(title):
    all_titles = books['Book'].tolist()
    closest_match = process.extractOne(title,all_titles)
    return closest_match[0]


# In[39]:


def get_content_based_recommendations(title_string, n_recommendations=10):
    title = book_finder(title_string)
    b_idx = dict(zip(books['Book'], list(books.index)))
    idx = b_idx[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:(n_recommendations+1)]
    similar_book= [i[0] for i in sim_scores]
    print(f"Because you watched {title}:")
    print(books['Book'].iloc[similar_book])


# In[44]:


bookname=input("Enter Book Name")
get_content_based_recommendations(bookname)


# ## Results

# In[41]:


def calculate_genre_similarity(genres1, genres2):
  
    # Count the number of matching genres
    num_matching_genres = len(set(genres1) & set(genres2))
    
    # Return the number of matching genres
    return num_matching_genres


# The accuracy of recommendations for 'Harry Potter' is 6.2, based on the genre similarity metric we used.
# 
# This means that, on average, the recommended books have a genre similarity score of 6.2 with the input book 'Harry Potter'. A higher similarity score indicates that the recommended books are more similar to 'Harry Potter' in terms of genre.

# You can interpret this score as an indication of how well the content-based recommendation algorithm is performing for the given input book. If the score is high, it suggests that the recommended books share similar genres with 'Harry Potter', which could indicate that the recommendations are relevant to fans of 'Harry Potter'. Conversely, a lower score may indicate that the recommendations are less relevant in terms of genre similarity.

# In[45]:


def evaluate_content_based_recommendations(title_string, n_recommendations=10):
    title = book_finder(title_string)
    b_idx = dict(zip(books['Book'], list(books.index)))
    idx = b_idx[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:(n_recommendations+1)]
    similar_books_indices = [i[0] for i in sim_scores]
    recommended_books = books['Book'].iloc[similar_books_indices].tolist()
    

    
    # Evaluate recommendations based on genre similarity
    input_book_genres = books.loc[books['Book'] == title, 'Genres'].iloc[0]
    recommended_books_genres = books.loc[similar_books_indices, 'Genres'].tolist()
    
    genre_similarity_scores = []
    for recommended_genre in recommended_books_genres:
        similarity_score = calculate_genre_similarity(input_book_genres, recommended_genre)
        genre_similarity_scores.append(similarity_score)
    
    # Calculate the mean genre similarity score
    mean_similarity_score = np.mean(genre_similarity_scores)
    
    return mean_similarity_score

# Example usage:
input_book_title = bookname
accuracy = evaluate_content_based_recommendations(input_book_title)
print(f"Accuracy of recommendations for '{input_book_title}': {accuracy}")



# ### Step 7: Dimensionality Reduction with Matrix Factorization (advanced)
# 
# Matrix factorization (MF) is a linear algebra technique that can help us discover latent features underlying the interactions between users and book. These latent features give a more compact representation of user tastes and item descriptions. MF is particularly useful for very sparse data and can enhance the quality of recommendations. The algorithm works by factorizing the original user-item matrix into two factor matrices:
# 
# - user-factor matrix (n_users, k)
# - item-factor matrix (k, n_items)
# 
# We are reducing the dimensions of our original matrix into "taste" dimensions. We cannot interpret what each latent feature $k$ represents. However, we could imagine that one latent feature may represent users who like romantic comedies from the 1990s, while another latent feature may represent movies which are independent foreign language films.
# 
# $$X_{mn}\approx P_{mk}\times Q_{nk}^T = \hat{X} $$
# 

# In[ ]:


from sklearn.decomposition import TruncatedSVD

svd = TruncatedSVD(n_components=20, n_iter=10)
Q = svd.fit_transform(X.T)
Q.shape


# In[ ]:


book_id = 1
similar_books = find_similar_books(book_id, Q.T, b_mapper, b_inv_mapper, metric='cosine', k=10)
b_title = book_titles[book_id]

print(f"Because you watched {b_title}:")
for i in similar_books:
    print(book_titles[i])


# The results above are the most similar movies to Toy Story using kNN on our “compressed” movie-factor matrix. We reduced the dimensions down to n_components=20. We can think of each component representing a latent feature such as movie genre.
