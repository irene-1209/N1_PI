from fastapi import FastAPI
import pandas as pd
import calendar
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Cargar el archivo CSV
df = pd.read_csv('PI_IRENE\ETL-EDA\data.csv')
df['fecha_estreno'] = pd.to_datetime(df['release_date'], errors='coerce')

# Diccionario de meses en español a inglés
meses_en = {
    'enero': 'January', 'febrero': 'February', 'marzo': 'March',
    'abril': 'April', 'mayo': 'May', 'junio': 'June',
    'julio': 'July', 'agosto': 'August', 'septiembre': 'September',
    'octubre': 'October', 'noviembre': 'November', 'diciembre': 'December'
}

# Diccionario de días en español a inglés
days_mapping = {
    'lunes': 'Monday', 'martes': 'Tuesday', 'miércoles': 'Wednesday', 'jueves': 'Thursday',
    'viernes': 'Friday', 'sábado': 'Saturday', 'domingo': 'Sunday'
}

# Definir la aplicación FastAPI
app = FastAPI()

# Definir la función para contar las filmaciones por mes
def cantidad_filmaciones_mes(Mes):
    try:
        mes_ingles = meses_en.get(Mes.lower())
        if mes_ingles:
            mes_numero = pd.to_datetime(mes_ingles, format='%B', errors='coerce').month
            films_mes = df[df['fecha_estreno'].dt.month == mes_numero]
            cantidad = len(films_mes)
            mensaje = f"{cantidad} cantidad de películas fueron estrenadas en el mes de {Mes}."
            return mensaje
        else:
            return "Error: Mes no reconocido. Por favor, ingrese un mes válido en español."
    except Exception as e:
        return f"Error: {e}"

# Ruta raíz con decorador
@app.get('/')
def root():
    return {"mensaje": "API para consultar cantidad de películas por mes y por día"}

# Ruta para obtener la cantidad de filmaciones por mes
@app.get('/cantidad_filmaciones_mes/{mes}')
def obtener_cantidad_filmaciones_mes(mes: str):
    return {"mensaje": cantidad_filmaciones_mes(mes)}

# Ruta para obtener la cantidad de filmaciones por día
@app.get('/cantidad_filmaciones_dia/{dia}')
def cantidad_filmaciones_dia(dia: str):
    # Convertir dia inglés a español
    dia_ingles = days_mapping.get(dia.lower())
    if not dia_ingles:
        return {"error": "Día no válido. Debe ser uno de: lunes, martes, miércoles, jueves, viernes, sábado, domingo"}
    
    # Filtrar solo el conjunto para incluir solo el número específico
    df['day_of_week'] = df['fecha_estreno'].dt.day_name()
    count = df[df['day_of_week'] == dia_ingles].shape[0]
    
    return {f"{count} cantidad de películas fueron estrenadas en los días {dia}"}



# Nueva ruta para obtener información de una película por título
@app.get('/score_titulo/{titulo}')
def score_titulo(titulo: str):
    try:
        film = df[df['title'].str.lower() == titulo.lower()]
        if not film.empty:
            titulo_film = film['title'].values[0]
            anio_estreno = film['fecha_estreno'].dt.year.values[0]
            puntuacion = film['popularity'].values[0]
            mensaje = f"La película {titulo_film} fue estrenada en el año {anio_estreno} con una puntuación/popularidad de {puntuacion}."
            return {"mensaje": mensaje}
        else:
            return {"error": "Película no encontrada. Por favor, ingrese un título válido."}
    except Exception as e:
        return {"error": f"Error: {e}"}


# Función para obtener información de votos de una película por título
@app.get('/votos_titulo/{titulo}')
def votos_titulo(titulo: str):
    try:
        film = df[df['title'].str.lower() == titulo.lower()]
        if not film.empty:
            votos = film['vote_count'].values[0]
            promedio_votos = film['vote_average'].values[0]
            if votos >= 2000:
                mensaje = f"La película {titulo} fue estrenada en el año {film['fecha_estreno'].dt.year.values[0]}. La misma cuenta con un total de {votos} valoraciones, con un promedio de {promedio_votos}."
                return {"mensaje": mensaje}
            else:
                return {"error": f"La película {titulo} no cumple con la condición de tener al menos 2000 valoraciones."}
        else:
            return {"error": "Película no encontrada. Por favor, ingrese un título válido."}
    except Exception as e:
        return {"error": f"Error: {e}"}
    
    # Función para obtener información de un actor por nombre
@app.get('/get_actor/{nombre_actor}')
def get_actor(nombre_actor: str):
    try:
        # Filtrar el DataFrame para obtener todas las películas en las que el actor ha participado
        actor_films = df[df['cast'].str.lower().str.contains(nombre_actor.lower())]
        
        if not actor_films.empty:
            cantidad_films = len(actor_films)
            retorno_total = actor_films['revenue'].sum()
            promedio_retorno = actor_films['revenue'].mean()
            
            mensaje = f"El actor {nombre_actor} ha participado en {cantidad_films} cantidad de filmaciones. El mismo ha conseguido un retorno total de {retorno_total} con un promedio de {promedio_retorno} por filmación."
            return {"mensaje": mensaje}
        else:
            return {"error": "Actor no encontrado en el dataset. Por favor, ingrese un nombre de actor válido."}
    
    except Exception as e:
        return {"error": f"Error: {e}"}
    
    # Función para obtener información de un director por nombre
@app.get('/get_director/{nombre_director}')
def get_director(nombre_director: str):
    try:
        # Filtrar el DataFrame para obtener todas las películas dirigidas por el director
        director_films = df[df['director'].str.lower().str.contains(nombre_director.lower())]
        
        if not director_films.empty:
            mensaje = f"El director {nombre_director} ha dirigido las siguientes películas:\n"
            
            for index, row in director_films.iterrows():
                titulo = row['title']
                fecha_lanzamiento = row['release_date']
                retorno = row['revenue']
                costo = row['budget']
                ganancia = retorno - costo
                
                mensaje += f"- {titulo} (Fecha de lanzamiento: {fecha_lanzamiento}, Retorno: {retorno}, Costo: {costo}, Ganancia: {ganancia})\n"
            
            return {"mensaje": mensaje}
        else:
            return {"error": "Director no encontrado en el dataset. Por favor, ingrese un nombre de director válido."}
    
    except Exception as e:
        return {"error": f"Error: {e}"}
    
    
# Función para recomendar películas similares
@app.get('/recomendacion/{titulo}')
def recomendacion(titulo: str):
    try:
        # Utilizar TF-IDF para vectorizar las descripciones y encontrar similitudes
        tfidf = TfidfVectorizer(stop_words='english')
        df['overview'] = df['overview'].fillna('')
        tfidf_matrix = tfidf.fit_transform(df['overview'])
        
        # Calcular la similitud del coseno entre películas
        cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
        
        # Obtener el índice de la película que coincide con el título
        idx = df[df['title'].str.lower() == titulo.lower()].index[0]
        
        #Filtrar las películas de los últimos 5 años
        df_recent_movies = df[df['fecha_estreno'].dt.year >= (pd.Timestamp.now().year - 5)]

        
        # Ordenar las películas según las puntuaciones de similitud
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Obtener las puntuaciones de similitud de las 5 películas más similares
        sim_scores = sim_scores[1:6]  # Excluye la propia película (índice 0) que es la más similar consigo misma
        
        # Obtener los índices de las películas más similares
        movie_indices = [i[0] for i in sim_scores]
        
        # Devolver información de las películas más similares
        similar_movies = df.iloc[movie_indices][['title', 'fecha_estreno', 'popularity']].reset_index(drop=True)
        
        return {"mensaje": f"Recomendaciones para la película '{titulo}':",
                "recomendaciones": similar_movies.to_dict(orient='records')}
    
    except Exception as e:
        return {"error": f"Error: {e}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

