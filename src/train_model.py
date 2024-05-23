import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle


def train_model(animeData,model_path) :
    animeData = pd.read_csv("/home/adila/Projects/test/data/AnimeWorld.csv")
    animeData.reset_index(inplace=True)
    features = ['Genre', 'Description', 'Studio']
    for feature in features :
        animeData[feature] = animeData[feature].fillna('')
    combined_features = animeData['Genre']+' '+animeData['Description']+' '+animeData['Studio']
    #Converting the string values to vectors by feature extraction
    vectoriser = TfidfVectorizer()
    feature_vectors = vectoriser.fit_transform(combined_features)
    similarity = cosine_similarity(feature_vectors)

    model = {
        'vectoriser' : vectoriser,
        'similarity' : similarity,
        'animeData' : animeData
    }

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

if __name__ == "__main__" :
    train_model('/home/adila/Projects/test/data/AnimeWorld.csv', '/home/adila/Projects/test/models/anime_recommender_model.pkl')

