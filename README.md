# Song_Sentiment_Ver2.0
Using the foundation and understanding of my first approach to song sentiment analysis using lyrics, created a more robust and better approach to this problem.

#### Objective
Using song lyrics be able to tag songs based on the meaning of the lyrics without taking into consideration other variables such as genre, bps, or duration.

#### Resources:

- Data Cleaning:
  * NLTK
  * Pandas
  * Numpy
  * JSON
  * REGEX (re)
  
- Access to several APIs:
  * Spotify API and spotipy library in Python
  * Genius Lyrics API and lyricsgenius library in Python
  * Last.fm API
  * Wikipedia library in Python
  * AudioDB API and Beautiful Soup to extract the data

- Modeling:
  * LDA Topic Modeling
  * Text Rank
  * Word2Vec

- Testing Results:
  * Similarity Score
  * Confusion Matrix
        
#### Methods:

- Pull songs from 24 popular playlists from the Spotify API.
- Pull the lyrics of all the songs from the Genius Lyrics API
- Pull the Genre of all the songs from the AudioDB API using Beautiful Soup
- Use Wikipedia library to import genres and compare to LastFM tags
- Combine all Data into a DataFrame
- Use NLTK, REGEX and Spacy to clean up the lyrics
- Use LDA Topic Modeling and Text Rank to create a taxonomy of the topics of each lyric
- Use Word2Vec to vectorize the words in the lyrics
- Using the vectorized words, calculate cosine similarity of each lyric to a topic in the taxonamy, highest similarity wins
- Manually tag each lytic with one of the topics
- Calculate accuracy of the word2vec model vs the manual labels
