import streamlit as st
import pandas as pd
import zipfile
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors


def main():
    st.title("Sistema de Recomendação de Filmes")
    st.title("Sistema de Recomendação usando técnica de conteúdo")
    st.markdown("Base de dados do [TMDB_Movies](https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies)")
    st.markdown("Base de dados [Ratings](https://www.kaggle.com/datasets/luisreimberg/ratingscsv)")
    st.markdown("Link do [Github](https://github.com/LucioFSMelo/recomender) do projeto.")
    st.markdown("""
        Este é um projeto da disciplina de **Sistemas de Recomendação**  
        **Instituição:** Serviço Nacional de Aprendizagem Comercial - SENAC  
        **Professor:** Welton Dionisio  
        **Alunos:**  
        Bruno Lundgren  
        João Pedro  
        José Victor  
        Lucio Flavio  
        Wellington França
        """)

    @st.cache_data
    def load_data(file):
        if file.name.endswith('.zip'):
            with zipfile.ZipFile(file) as z:
                for filename in z.namelist():
                    if filename.endswith('.csv'):
                        with z.open(filename) as f:
                            df = pd.read_csv(f)
                            return df
        elif file.name.endswith('.csv'):
            df = pd.read_csv(file)
            return df
        else:
            return None

    uploaded_file = st.file_uploader("Faça o upload de um arquivo CSV ou ZIP para dados de filmes", type=["csv", "zip"])
    uploaded_ratings = "C:/Users/luciu/Workspace/Senac_proj/Recomender_System/Recomender/recomender/data/ratings.csv"

    def content_based_recommendations(title, cosine_sim, indices, df):
        idx = indices[title]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]
        movie_indices = [i[0] for i in sim_scores]
        return df['title'].iloc[movie_indices]

    def collaborative_recommendations(user_id, user_movie_matrix, movie_ids, n_neighbors=10):
        model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
        model_knn.fit(user_movie_matrix)
        distances, indices = model_knn.kneighbors(user_movie_matrix.iloc[user_id].values.reshape(1, -1),
                                                  n_neighbors=n_neighbors)
        recommendations = [movie_ids[i] for i in indices.flatten()]
        return recommendations

    def hybrid_recommendations(user_id, title, cosine_sim, indices, df, user_movie_matrix, movie_ids, n=10):
        content_recs = content_based_recommendations(title, cosine_sim, indices, df)
        collab_recs = collaborative_recommendations(user_id, user_movie_matrix, movie_ids, n)
        hybrid_recs = pd.concat([content_recs, pd.Series(collab_recs)]).drop_duplicates().head(n)
        return hybrid_recs

    if uploaded_file is not None and uploaded_ratings is not None:
        df = load_data(uploaded_file)
        df_ratings = pd.read_csv(uploaded_ratings)

        if df is None:
            st.error(
                "Erro ao carregar o arquivo de filmes. Certifique-se de que é um arquivo CSV ou um arquivo ZIP contendo um CSV.")
        elif df_ratings.empty:
            st.error("Erro ao carregar o arquivo de avaliações. Certifique-se de que é um arquivo CSV.")
        else:
            st.write("Arquivos carregados com sucesso!")
            st.dataframe(df.head())

            # Usar um subconjunto menor de dados para evitar o MemoryError
            df_sample = df.sample(n=1000, random_state=42).reset_index(drop=True)

            # Pre-processamento e criação da matriz de similaridade para filtragem de conteúdo
            df_sample['title'] = df_sample['title'].fillna('Unknown')
            df_sample['genres'] = df_sample['genres'].fillna('Unknown')
            df_sample['keywords'] = df_sample['keywords'].fillna('Unknown')
            df_sample['features'] = df_sample['title'] + ' ' + df_sample['genres'] + ' ' + df_sample['keywords']

            # Vetorizar as características dos filmes usando TF-IDF
            tfidf = TfidfVectorizer(stop_words='english')
            tfidf_matrix = tfidf.fit_transform(df_sample['features'])

            # Calcular a similaridade do cosseno entre os filmes
            cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

            # Construir um mapeamento reverso de títulos de filmes para índices
            indices = pd.Series(df_sample.index, index=df_sample['title']).drop_duplicates()
            movie_ids = df_sample['title'].tolist()

            # Filtragem colaborativa
            user_movie_matrix = df_ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)

            # Seleção do método de recomendação e entrada de ID do usuário e título do filme
            selected_method = st.selectbox("Escolha o método de recomendação:", ["content", "collaborative", "hybrid"])
            user_id = st.number_input("Insira a ID do usuário:", min_value=1, max_value=user_movie_matrix.index.max())
            movie_title = st.selectbox("Selecione o título do filme:", df_sample['title'].unique())

            # Obter recomendações com base na seleção do usuário
            if st.button("Obter Recomendações"):
                if selected_method == 'content':
                    recommendations = content_based_recommendations(movie_title, cosine_sim, indices, df_sample)
                elif selected_method == 'collaborative':
                    recommendations = collaborative_recommendations(user_id, user_movie_matrix, movie_ids)
                elif selected_method == 'hybrid':
                    recommendations = hybrid_recommendations(user_id, movie_title, cosine_sim, indices, df_sample,
                                                             user_movie_matrix, movie_ids)

                # Exibindo recomendações
                st.write(f"Recomendações para o usuário '{user_id}' e filme '{movie_title}':")
                st.dataframe(recommendations)


if __name__ == "__main__":
    main()
