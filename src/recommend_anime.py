import pickle
import difflib

def load_model(model_path) :
    with open(model_path, 'rb') as f :
        model = pickle.load(f)
    return model


def recommend_anime(anime_name, model) :
    vectoriser = model['vectoriser']
    similarity = model['similarity']
    animeData = model['animeData']

    listOfAnimes = animeData['Anime'].tolist()
    findCloseMatch = difflib.get_close_matches(anime_name, listOfAnimes)

    if not findCloseMatch:
        return "No close match found."

    close_match = findCloseMatch[0]
    indexOfTheAnime = animeData[animeData.Anime == close_match]['index'].values[0]

    # Create similarity_score list for all animes
    similarity_score = []

    for i in range(len(animeData)):
        anime_title = animeData.loc[i, 'Anime']
        score = difflib.SequenceMatcher(None, anime_name, anime_title).ratio()
        similarity_score.append((animeData.loc[i, 'index'], score))

    # Sort similarity scores in descending order
    sortedSimilarAnimes = sorted(similarity_score, key=lambda x: x[1], reverse=True)

    recommended_animes = []
    for i, (index, score) in sortedSimilarAnimes[:10] :
        title = animeData.iloc[index]['Anime']
        recommended_animes.append((i, title))


if __name__ == "__main__":
    model = load_model('models/anime_recommender_model.pkl')
    anime_name = input("Enter your favourite anime (in Japanese): ")
    recommendations = recommend_anime(anime_name, model)

    if isinstance(recommendations, str):
        print(recommendations)
    else:
        print("Here are 10 anime recommendations based on your favourite anime:")
        for i, title in recommendations:
            print(f"{i}. {title}")