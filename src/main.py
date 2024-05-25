from recommend_anime import load_model, recommend_anime

if __name__ == "__main__":
    model = load_model('/home/adila/Projects/test/models/anime_recommender_model.pkl')
    anime_name = input("Enter your favourite anime (in Japanese): ")
    recommendations = recommend_anime(anime_name, model)
    
    if isinstance(recommendations, str):
        print(recommendations)
    else:
        print("Here are 10 anime recommendations based on your favourite anime:")
        for i, title in recommendations:
            print(f"{i}. {title}")
