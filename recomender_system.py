import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import zipfile

data_path = "C:/Users/luciu/Workspace/Senac_proj/Recomender_System/app_recomender/recomender/dataset/TMDB_movie_dataset_v11.csv"

st.title("Sistema de Recomendação usando técnica de conteúdo")
st.markdown("Base de dados do [Kaggle](https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies)")
st.markdown("Link do [Github](https://github.com/LucioFSMelo/recomender) do projeto.")
st.markdown("""
Este é um projeto da disciplina de **Sistemas de Recomendação**  
**Instituição:** Serviço Nacional de Aprendizagem Comercial - SENAC  
**Professor:** Welton Dionisio  
**Alunos:**  
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

def get_recommendations(title, cosine_sim, indices, df):
    # Verificar se o título existe no índice
    if title not in indices:
        return "Filme não encontrado."

    # Obter o índice do filme que corresponde ao título
    idx = indices[title]

    # Obter as pontuações de similaridade de todos os filmes com aquele filme
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Classificar os filmes com base nas pontuações de similaridade
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Obter as pontuações dos 10 filmes mais similares
    sim_scores = sim_scores[1:11]

    # Obter os índices dos filmes
    movie_indices = [i[0] for i in sim_scores]

    # Retornar os 10 filmes mais similares
    return df['title'].iloc[movie_indices]

def main():
    uploaded_file = st.file_uploader("Faça o upload de um arquivo CSV ou ZIP", type=["csv", "zip"])

    if uploaded_file is not None:
        df = load_data(uploaded_file)

        if df is None:
            st.error("Erro ao carregar o arquivo. Certifique-se de que é um arquivo CSV ou um arquivo ZIP contendo um CSV.")
            return

        # Usar um subconjunto menor de dados para evitar o MemoryError
        df_sample = df.sample(n=1000, random_state=42).reset_index(drop=True)

        # Pre-processamento e criação da matriz de similaridade
        # Tratando valores nulos
        df_sample['title'] = df_sample['title'].fillna('Unknown')
        df_sample['release_date'] = df_sample['release_date'].fillna('Unknown')
        df_sample['original_title'] = df_sample['original_title'].fillna('Unknown')
        df_sample['genres'] = df_sample['genres'].fillna('Unknown')
        df_sample['keywords'] = df_sample['keywords'].fillna('Unknown')

        # Combinar 'title', 'genres', 'keywords', 'original_title' e 'release_date' em uma única coluna de características
        df_sample['features'] = df_sample['title'] + ' ' + df_sample['genres'] + ' ' + df_sample['keywords'] + ' ' + \
                                df_sample['original_title'] + ' ' + df_sample['release_date']

        # Vetorizar as características dos filmes usando TF-IDF
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(df_sample['features'])

        # Usar TruncatedSVD para reduzir a dimensionalidade
        svd = TruncatedSVD(n_components=100, random_state=42)
        tfidf_matrix_reduced = svd.fit_transform(tfidf_matrix)

        # Calcular a similaridade do cosseno entre os filmes
        cosine_sim = cosine_similarity(tfidf_matrix_reduced, tfidf_matrix_reduced)

        # Construir um mapeamento reverso de títulos de filmes para índices
        indices = pd.Series(df_sample.index, index=df_sample['title']).drop_duplicates()

        # Seleção do filme pelo usuário
        selected_movie = st.selectbox("Escolha um filme para obter recomendações:", df_sample['title'].tolist())

        # Obter recomendações com base na seleção do usuário
        if selected_movie:
            recommendations = get_recommendations(selected_movie, cosine_sim, indices, df_sample)

            # Exibindo recomendações
            if isinstance(recommendations, str):
                st.error(recommendations)
            else:
                st.write(f"Recomendações para o filme '{selected_movie}':")
                st.dataframe(recommendations)

if __name__ == "__main__":
    main()
