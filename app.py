import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS
import folium
from folium.plugins import MarkerCluster
import plotly.graph_objects as go
import base64
import time
import unicodedata

# --- Configuración inicial de Streamlit ---
# st.set_page_config: Configura cómo se verá la página web, como el ancho y el título de la pestaña del navegador.
st.set_page_config(layout="wide", page_title="Análisis de Películas")
# st.title: Muestra el título principal en la aplicación web.
st.title("🎬 Dashboard Interactivo de Análisis de Películas")
# st.markdown: Agrega una línea divisoria para organizar la interfaz.
st.markdown("---")

# --- Rutas de archivos (ajustadas para el entorno de Streamlit) ---
# Se asume que la carpeta 'data' y 'imagenes_directores_actores' están en el mismo nivel que 'streamlit_app.py'
# Definimos las rutas a nuestras carpetas de datos e imágenes.
# RUTA_BASE_DATA: Donde están los archivos CSV y Excel.
RUTA_BASE_DATA = "data/"
# RUTA_IMAGENES_DIRECTORES_ACTORES: Donde se guardan las fotos de personas.
RUTA_IMAGENES_DIRECTORES_ACTORES = "imagenes_directores_actores/"
# RUTA_IMAGENES_DATA: Otra carpeta para imágenes dentro de los datos.
RUTA_IMAGENES_DATA = os.path.join(RUTA_BASE_DATA, 'imagenes') # Si tienes imágenes dentro de la carpeta 'data'

# --- Función para cargar datos (centralizada y optimizada para Streamlit) ---
# @st.cache_data: Esta línea es muy importante. Le dice a Streamlit que guarde en memoria
# los datos que se cargan con esta función. Así, si el usuario interactúa con la app,
# los datos no se vuelven a cargar desde el disco cada vez, haciendo la aplicación más rápida.
@st.cache_data # Decorador de caché para evitar recargar datos en cada interacción
def load_data(file_name, encoding='utf-8-sig'):
    # Construimos la ruta completa al archivo.
    path = os.path.join(RUTA_BASE_DATA, file_name)
    try:
        # Intentamos leer el archivo, ya sea CSV o Excel.
        if file_name.endswith('.csv'):
            df = pd.read_csv(path, encoding=encoding)
        elif file_name.endswith('.xlsx'):
            df = pd.read_excel(path)
        # st.success: Muestra un mensaje verde si la carga fue exitosa.
        st.success(f"✅ Cargado: {file_name} ({df.shape[0]} filas)")
        return df
    except FileNotFoundError:
        # st.error: Muestra un mensaje rojo si el archivo no se encuentra.
        st.error(f"❌ Error: El archivo '{file_name}' no se encontró en la ruta '{path}'. Asegúrate de haberlo subido correctamente.")
        return pd.DataFrame() # Devuelve un DataFrame vacío si hay error
    except Exception as e:
        # st.error: Muestra un mensaje rojo para cualquier otro tipo de error.
        st.error(f"❌ Error cargando {file_name}: {e}")
        return pd.DataFrame() # Devuelve un DataFrame vacío si hay error

# --- 1. Carga y Muestra de DataFrames (desde la sección 3 de tu código) ---
# st.header: Define un título para esta sección de la aplicación.
st.header("📊 Resumen de Datos Cargados")
# st.write: Agrega una descripción a la sección.
st.write("Aquí se cargan automáticamente todos los archivos CSV/XLSX de la carpeta `data/`.")

# st.columns: Divide el área en dos columnas para organizar el contenido.
col1, col2 = st.columns(2) # Usamos columnas para una mejor presentación

csv_files_info = {}
# os.listdir: Lista todos los archivos dentro de la carpeta de datos.
for file_name in os.listdir(RUTA_BASE_DATA):
    # Filtramos para asegurarnos de que solo procesamos archivos CSV o XLSX.
    if file_name.endswith('.csv') or file_name.endswith('.xlsx'):
        # Llamamos a nuestra función para cargar cada archivo.
        df = load_data(file_name)
        if not df.empty: # Si el DataFrame no está vacío (es decir, se cargó correctamente)
            csv_files_info[file_name.split('.')[0]] = df
            # st.checkbox: Muestra una casilla para que el usuario elija si quiere ver las primeras filas.
            if col1.checkbox(f"Mostrar primeras filas de **{file_name}**"):
                with col2: # El contenido dentro de 'with col2:' se mostrará en la segunda columna.
                    # st.dataframe: Muestra las primeras filas de la tabla de datos.
                    st.dataframe(df.head())
st.markdown("---") # Línea divisoria

# --- 2. Mostrar imágenes de directores/actores (desde la sección 4 y 5 de tu código) ---
st.header("🖼️ Imágenes de Directores y Actores")

# Esta función muestra una imagen dado su ruta.
def display_image_from_path(folder_path, image_name, caption=""):
    img_path = os.path.join(folder_path, image_name)
    if os.path.exists(img_path):
        # st.image: Muestra la imagen en la aplicación Streamlit.
        st.image(img_path, caption=caption, use_container_width=True) # Cambiado a use_container_width
    else:
        st.warning(f"⚠️ No se encontró la imagen: {image_name} en {folder_path}")

# Si quieres que se muestren imágenes de ejemplo aquí, puedes descomentar y modificar estas líneas:
# st.subheader("Imagen destacada:")
# display_image_from_path(RUTA_IMAGENES_DIRECTORES_ACTORES, 'Christopher_Nolan.jpg', 'Christopher Nolan')

st.markdown("---") # Línea divisoria

# --- 3. Películas más vistas y mejor puntuadas (desde tu segundo bloque de código) ---
st.header("📈 Top 10 Películas")

# Cargamos los DataFrames para las películas más vistas y mejor puntuadas.
df_vistas = load_data('top_10_mas_vistas.csv')
df_puntuadas = load_data('top_10_mejor_puntuadas.csv')

# Si los DataFrames no están vacíos, los mostramos.
if not df_vistas.empty:
    st.subheader("📊 Películas más vistas:")
    st.dataframe(df_vistas)

if not df_puntuadas.empty:
    st.subheader("🏆 Películas mejor puntuadas:")
    st.dataframe(df_puntuadas)
st.markdown("---") # Línea divisoria

# --- 4. Análisis de Sentimiento de Comentarios (desde tu tercer bloque de código) ---
st.header("💬 Análisis de Sentimiento de Comentarios")

# Cargamos los datos de comentarios de películas.
df_comentarios = load_data('comentarios_peliculas.csv')

if not df_comentarios.empty:
    # Definimos la función para clasificar el sentimiento (positivo, negativo, neutro) de un texto.
    def clasificar_sentimiento(texto):
        if pd.isnull(texto):
            return "neutral"
        polaridad = TextBlob(str(texto)).sentiment.polarity
        if polaridad > 0.1:
            return "positivo"
        elif polaridad < -0.1:
            return "negativo"
        else:
            return "neutral"

    # Aplicamos la función de sentimiento a cada comentario.
    df_comentarios["sentimiento"] = df_comentarios["contenido"].apply(clasificar_sentimiento)
    # Agrupamos los comentarios para contar cuántos hay de cada tipo por película.
    resumen_sentimiento = df_comentarios.groupby(["titulo", "sentimiento"]).size().reset_index(name="cantidad")

    # Creamos un gráfico de barras para visualizar los sentimientos.
    fig_sentimiento, ax_sentimiento = plt.subplots(figsize=(12, 7))
    barplot = sns.barplot(
        data=resumen_sentimiento,
        x="cantidad",
        y="titulo",
        hue="sentimiento",
        palette={"positivo": "green", "negativo": "red", "neutral": "gray"},
        ax=ax_sentimiento
    )
    # Añadimos las etiquetas de cantidad sobre cada barra.
    for container in barplot.containers:
        barplot.bar_label(container, fmt='%d', label_type='edge', fontsize=9, padding=3)

    # Configuramos el título y etiquetas del gráfico.
    ax_sentimiento.set_title("Cantidad de comentarios por película y tipo de sentimiento", fontsize=14)
    ax_sentimiento.set_xlabel("Cantidad de comentarios")
    ax_sentimiento.set_ylabel("Película")
    ax_sentimiento.legend(title="Sentimiento")
    plt.tight_layout()
    # st.pyplot: Muestra el gráfico de Matplotlib en la aplicación.
    st.pyplot(fig_sentimiento)
else:
    st.warning("No se pudo cargar el DataFrame de comentarios para el análisis de sentimiento.")
st.markdown("---") # Línea divisoria

# --- 5. Nube de Palabras de Comentarios (desde tu cuarto bloque de código) ---
st.header("☁️ Nube de Palabras de Comentarios")

if not df_comentarios.empty:
    # Juntamos todos los comentarios en un solo texto.
    texto_total = " ".join(str(comentario) for comentario in df_comentarios["contenido"].dropna())

    # Definimos una lista de palabras comunes a ignorar para que no saturen la nube.
    stopwords = set(STOPWORDS)
    stopwords.update([
        "film", "movie", "one", "can", "get", "like", "just", "also",
        "even", "story", "see", "make", "much", "well", "though",
        "really", "good", "bad", "movies", "will", "still", "show",
        "way", "back", "could", "makes", "first"
    ])

    # Creamos la nube de palabras. Las palabras más frecuentes aparecerán más grandes.
    wordcloud = WordCloud(
        stopwords=stopwords,
        background_color="white",
        max_words=200,
        width=1200,
        height=600
    ).generate(texto_total)

    # Mostramos la nube de palabras en la aplicación.
    fig_wordcloud, ax_wordcloud = plt.subplots(figsize=(15, 7))
    ax_wordcloud.imshow(wordcloud, interpolation="bilinear")
    ax_wordcloud.axis("off")
    ax_wordcloud.set_title("Palabras más frecuentes en los comentarios", fontsize=16)
    st.pyplot(fig_wordcloud)
else:
    st.warning("No hay datos de comentarios para generar la nube de palabras.")
st.markdown("---") # Línea divisoria

# --- 6. Mapa de Vistas por País (desde tu sexto y séptimo bloque de código) ---
st.header("🌍 Mapa de Películas Más Vistas por País")

# Cargamos los datos de coordenadas de países y vistas por país.
df_coords = load_data('vistas_por_pais_con_coords.csv')
df_vistas_pais = load_data('vistas_por_pais.csv')

if not df_coords.empty and not df_vistas_pais.empty:
    # Combinamos las dos tablas de datos para tener la información completa.
    df_merged_map = pd.merge(df_coords, df_vistas_pais, on='pais', how='inner')

    # Creamos el mapa base.
    m = folium.Map(location=[10, 0], zoom_start=2, tiles="CartoDB positron")
    # Agregamos un sistema para agrupar los marcadores si hay muchos en un área.
    marker_cluster = MarkerCluster().add_to(m)

    # Preparamos el texto que aparecerá al hacer clic en un marcador.
    df_merged_map['popup_text'] = df_merged_map.apply(
        lambda row: f"<b>{row['pais']}</b><br>🎬 Película: <i>{row['titulo']}</i><br>👁️‍🗨️ Vistas: {row['vistas']:,}",
        axis=1
    )
    # Preparamos el texto que aparecerá al pasar el ratón por encima.
    df_merged_map['tooltip_text'] = df_merged_map['pais']

    # Asignamos un color único a cada película para distinguirlas en el mapa.
    peliculas_map = df_merged_map['titulo'].unique()
    colores = [
        'red', 'green', 'blue', 'purple', 'orange', 'darkred', 'cadetblue',
        'darkgreen', 'darkpurple', 'pink', 'lightblue', 'lightgreen'
    ]
    color_dict = {pelicula: colores[i % len(colores)] for i, pelicula in enumerate(peliculas_map)}

    # Añadimos un marcador circular por cada país en el mapa.
    for _, row in df_merged_map.iterrows():
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=8,
            color=color_dict[row['titulo']],
            fill=True,
            fill_color=color_dict[row['titulo']],
            fill_opacity=0.7,
            popup=folium.Popup(row['popup_text'], max_width=300),
            tooltip=row['tooltip_text']
        ).add_to(marker_cluster)

    # Agregamos una leyenda flotante al mapa para explicar los colores.
    legend_html = '''
    <div style="position: fixed;
             bottom: 50px; left: 50px; width: 260px; height: auto;
             background-color: white; z-index:9999; font-size:14px;
             border:2px solid grey; border-radius:10px; padding: 10px;
             font-family: Arial;">
    <b>🎬 Mapa de películas más vistas por país</b><br><br>
    ''' + ''.join([
        f'<i style="background:{color_dict[p]}; width:12px; height:12px; display:inline-block; margin-right:6px; border-radius:50%;"></i>{p}<br>'
        for p in peliculas_map
    ]) + '</div>'

    m.get_root().html.add_child(folium.Element(legend_html))
    st.write("Mapa interactivo de películas más vistas por país:")
    # st.components.v1.html: Muestra el mapa Folium en Streamlit.
    st.components.v1.html(folium.Figure().add_child(m).render(), height=500)
else:
    st.warning("No se pudieron cargar los datos de vistas por país para mostrar el mapa.")
st.markdown("---") # Línea divisoria

# --- 7. Premios Ganados por Películas (desde tu penúltimo bloque de código) ---
st.header("🏆 Premios Ganados por Películas (2022–2024)")

# Cargamos los datos de premios de películas.
df_premios = load_data('peliculas_premios.csv')

if not df_premios.empty:
    # Limpiamos los nombres de las columnas y convertimos los premios a números.
    df_premios.columns = df_premios.columns.str.strip().str.lower()
    df_premios['premios_ganados'] = pd.to_numeric(df_premios['premios_ganados'], errors='coerce').fillna(0).astype(int)
    # Creamos una columna visual con trofeos (🏆) para cada premio.
    df_premios['🏆 PREMIOS'] = df_premios['premios_ganados'].apply(lambda x: "🏆 " * x if x < 6 else f"🏆 x{x}")

    # Definimos las columnas que queremos mostrar.
    cols_mostrar = ['titulo', 'año', '🏆 PREMIOS', 'detalle_premios']

    # Creamos una tabla interactiva usando Plotly.
    fig_premios = go.Figure(data=[go.Table(
        header=dict( # Configuramos el encabezado de la tabla.
            values=["<b style='font-size:16px'>" + col.upper() + "</b>" for col in cols_mostrar],
            fill_color='#4b0082',
            line_color='darkslategray',
            align='center',
            font=dict(color='white', size=14, family='DejaVu Sans Mono'),
            height=40
        ),
        cells=dict( # Configuramos las celdas de la tabla, incluyendo colores alternos.
            values=[df_premios[col] for col in cols_mostrar],
            fill_color=[['#ffffff', '#f3f4f6'] * (len(df_premios) // 2 + 1)][:len(df_premios)],
            line_color='lightgrey',
            align='left',
            font=dict(color='black', size=13, family='Verdana'),
            height=34
        )
    )])

    # Actualizamos el diseño de la tabla, añadiendo un título centrado.
    fig_premios.update_layout(
        title={
            'text': "<b style='color:#4b0082'>🎬 Premios Ganados por Películas (2022–2024)</b>",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        title_font=dict(size=22, family='Helvetica Neue'),
        margin=dict(l=5, r=5, t=80, b=10),
        paper_bgcolor='white'
    )
    # st.plotly_chart: Muestra el gráfico/tabla de Plotly en la aplicación.
    st.plotly_chart(fig_premios, use_container_width=True)
else:
    st.warning("No se pudo cargar el DataFrame de premios.")
st.markdown("---") # Línea divisoria

# --- 8. Visualización de Actores y Directores por Película (tu antepenúltimo bloque) ---
st.header("🎭 Actores y Directores por Película")

# Cargamos el DataFrame de actores y directores (asumiendo que ya está corregido).
df_actores_directores = load_data('actores_directores_corregido.csv')

if not df_actores_directores.empty:
    # Esta función convierte una imagen a un formato base64 para mostrarla directamente en HTML.
    def mostrar_imagen_b64(path, width=150):
        if os.path.exists(path):
            with open(path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode('utf-8')
            return f'<img src="data:image/jpeg;base64,{img_b64}" width="{width}" style="border-radius: 5px;"/>'
        return "🖼️ No disponible"

    # Iteramos por cada película para mostrar sus actores y directores.
    for pelicula in df_actores_directores['pelicula'].unique():
        st.subheader(f"🎞️ {pelicula}") # Título para cada película.
        # st.columns: Creamos varias columnas para organizar las "tarjetas" de personas.
        cols = st.columns(4) # Ajusta el número de columnas según tu preferencia

        idx = 0
        for _, row in df_actores_directores[df_actores_directores['pelicula'] == pelicula].iterrows():
            with cols[idx % 4]: # Asignamos cada tarjeta a una columna de forma rotatoria.
                ruta_img = os.path.join(RUTA_IMAGENES_DIRECTORES_ACTORES, row['imagen'])
                # Construimos el código HTML para cada tarjeta de actor/director.
                html_card = f"""
                <div style='text-align: center; background: #fff; padding: 10px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin-bottom: 15px;'>
                    {mostrar_imagen_b64(ruta_img)}
                    <h5 style='margin: 8px 0; color: #333;'>{row['nombre']}</h5>
                    <div style='color: #666; font-size: 0.9em;'>{row['rol']}</div>
                </div>
                """
                # st.markdown: Muestra el HTML de la tarjeta. unsafe_allow_html=True es necesario para mostrar HTML.
                st.markdown(html_card, unsafe_allow_html=True)
            idx += 1
else:
    st.warning("No se pudo cargar el DataFrame de actores y directores.")
st.markdown("---") # Línea divisoria

# --- 9. Recaudación de Películas (tu antepenúltimo bloque) ---
st.header("💰 Taquilla Mundial de Películas")

# Cargamos los datos de recaudación de películas.
df_recaudacion = load_data('recaudacion_peliculas.csv')

if not df_recaudacion.empty:
    # Limpiamos y preparamos los datos de recaudación.
    df_recaudacion.columns = df_recaudacion.columns.str.strip()
    df_recaudacion = df_recaudacion[["Película", "Recaudación (USD)", "Fecha de estreno"]]
    df_recaudacion["Recaudación (USD)"] = df_recaudacion["Recaudación (USD)"].astype(str)
    df_recaudacion = df_recaudacion.sort_values("Recaudación (USD)", ascending=False)

    # st.markdown: Inyectamos estilos CSS personalizados para la tabla de recaudación.
    st.markdown("""
    <style>
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.02); }
            100% { transform: scale(1); }
        }
        .movie-card-recaudacion {
            animation: fadeIn 0.6s ease-out;
            transition: all 0.3s ease;
            background: white;
            border-radius: 10px;
            padding: 20px;
            position: relative;
            margin-bottom: 20px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.08);
        }
        .movie-card-recaudacion:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 20px rgba(0,0,0,0.1);
        }
        .sticker {
            position: absolute;
            right: -10px;
            top: -10px;
            font-size: 11px;
            background: #ff4757;
            color: white;
            padding: 4px 8px;
            border-radius: 15px;
            transform: rotate(10deg);
            animation: pulse 2s infinite;
        }
        .total-box {
            background: linear-gradient(135deg, #d32f2f, #ff6b6b);
            color: white;
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
            text-align: center;
            font-size: 1.2em;
            box-shadow: 0 4px 15px rgba(210, 47, 47, 0.3);
        }
        .download-btn {
            background: #2ed573;
            color: white;
            padding: 12px 24px;
            border-radius: 30px;
            display: inline-block;
            margin: 10px 0;
            font-weight: bold;
            border: none;
            cursor: pointer;
            transition: all 0.3s;
            box-shadow: 0 4px 10px rgba(46, 213, 115, 0.3);
            text-decoration: none; /* Important for <a> tag */
        }
        .download-btn:hover {
            background: #25b562;
            transform: translateY(-2px);
            box-shadow: 0 6px 15px rgba(46, 213, 115, 0.4);
        }
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            padding: 10px 12px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }
        thead th {
            background-color: #d32f2f;
            color: white;
        }
        tbody tr:hover {
            background-color: #f5f5f5;
        }
    </style>
    """, unsafe_allow_html=True)

    # Comenzamos a construir la tabla HTML de recaudación.
    st.markdown(f"""
    <div class="movie-card-recaudacion">
        <div class="sticker">🔥 Actualizado</div>
        <table>
            <thead>
                <tr>
                    <th>Película</th>
                    <th style="text-align: right;">Recaudación</th>
                    <th style="text-align: center;">Estreno</th>
                </tr>
            </thead>
            <tbody>
    """, unsafe_allow_html=True)

    # Iteramos por cada película para añadirla a la tabla.
    for i, row in df_recaudacion.iterrows():
        emoji = "🎬" # Emoji por defecto.
        if i == 0: emoji = "👑" # Emoji especial para la película número 1.
        elif i == len(df_recaudacion) - 1: emoji = "🐢" # Emoji especial para la última película.

        st.markdown(f"""
                <tr>
                    <td>{emoji} {row['Película']}</td>
                    <td style="text-align: right; font-weight: bold;">{row['Recaudación (USD)']}</td>
                    <td style="text-align: center;">{row['Fecha de estreno']}</td>
                </tr>
        """, unsafe_allow_html=True)
    # Cerramos las etiquetas HTML de la tabla.
    st.markdown("""
            </tbody>
        </table>
    </div>
    """, unsafe_allow_html=True)

    # Función para limpiar el formato de los números de recaudación.
    def parse_number(x):
        try:
            return int(str(x).replace("$", "").replace(",", ""))
        except ValueError:
            return 0 # Devuelve 0 si la conversión falla

    # Calculamos la recaudación total.
    total_recaudacion = df_recaudacion["Recaudación (USD)"].apply(parse_number).sum()

    # Mostramos la recaudación total en un cuadro destacado.
    st.markdown(f"""
    <div class="total-box">
        💰 Recaudación Total: <strong>${total_recaudacion:,}</strong>
    </div>
    """, unsafe_allow_html=True)

    # Preparamos los datos para el botón de descarga.
    csv_to_download = df_recaudacion.to_csv(index=False, encoding='utf-8-sig')
    b64 = base64.b64encode(csv_to_download.encode()).decode()
    # st.markdown: Creamos el botón de descarga usando HTML.
    st.markdown(f'<a href="data:file/csv;base64,{b64}" download="recaudacion_peliculas.csv" class="download-btn">⬇️ Descargar Datos Completos</a>', unsafe_allow_html=True)

    # st.info: Muestra una pequeña nota informativa.
    st.info(f"🎥 Datos cargados desde archivo local | Visualizado: {time.strftime('%d/%m/%Y')}")
else:
    st.warning("No se pudo cargar el DataFrame de recaudación.")
st.markdown("---") # Línea divisoria

# --- 10. Recomendador de Películas (desde tu último bloque de código) ---
st.header("✨ Recomendador de Películas (2022-2024)")

# @st.cache_data: Al igual que antes, cacheamos esta función para que los datos solo se limpien una vez.
@st.cache_data
def cargar_y_limpiar_datos_recomendador():
    try:
        # Cargamos el archivo de películas completas.
        data = load_data("peliculas_completas.csv")
        if data.empty:
            return pd.DataFrame()

        # Filtramos las películas por año y eliminamos filas con datos faltantes.
        data = data[data['anio'].between(2022, 2024)].copy()
        data = data.dropna(subset=['titulo', 'genero', 'puntuacion', 'poster_url'])

        # Normalizamos los géneros para que las búsquedas sean más efectivas.
        data['genero_normalizado'] = data['genero'].apply(
            lambda x: ', '.join(sorted(set(unicodedata.normalize('NFKD', str(g)).encode('ASCII', 'ignore').decode('ASCII').lower().strip() for g in str(x).split(','))))
        )
        # Convertimos la duración a formato numérico.
        data['duracion'] = pd.to_numeric(data['duracion'], errors='coerce').fillna(0)
        return data
    except Exception as e:
        st.error(f"❌ Error al cargar y limpiar datos para el recomendador: {e}")
        return pd.DataFrame()

# Cargamos los datos para el recomendador.
df_recomendador = cargar_y_limpiar_datos_recomendador()

if not df_recomendador.empty:
    # Obtenemos las opciones únicas para géneros y plataformas.
    generos = sorted(set(g for sublist in df_recomendador['genero_normalizado'].str.split(', ') for g in sublist))
    plataformas = ["Todas"] + sorted(df_recomendador['plataforma'].dropna().unique())

    # with st.sidebar: Colocamos los controles de filtro en la barra lateral de la aplicación.
    with st.sidebar:
        st.subheader("Filtros de Búsqueda") # Título para los filtros.
        # st.selectbox: Crea un menú desplegable para el género y la plataforma.
        selected_genero = st.selectbox("Género:", options=generos)
        # st.slider: Crea un deslizador para seleccionar la puntuación mínima y el rango de duración.
        selected_puntuacion = st.slider("Puntuación mínima:", min_value=0.0, max_value=10.0, value=7.0, step=0.1)
        selected_duracion = st.slider("Duración (min):", min_value=0, max_value=240, value=(60, 180), step=15)
        selected_plataforma = st.selectbox("Plataforma:", options=plataformas)

        # st.button: Creamos los botones para buscar y reiniciar filtros.
        buscar_button = st.button("🔍 Buscar Películas", type="primary")
        reiniciar_button = st.button("🔄 Reiniciar Filtros")

    # Lógica de filtrado: esto se ejecuta cuando el botón "Buscar" es presionado.
    if buscar_button:
        # Aplicamos todos los filtros seleccionados por el usuario.
        filtered_data = df_recomendador[
            (df_recomendador['genero_normalizado'].str.contains(selected_genero, case=False)) &
            (df_recomendador['puntuacion'] >= selected_puntuacion) &
            (df_recomendador['duracion'].between(selected_duracion[0], selected_duracion[1])) &
            ((df_recomendador['plataforma'] == selected_plataforma) if selected_plataforma != "Todas" else True)
        ].sort_values(by='puntuacion', ascending=False)

        if filtered_data.empty: # Si no se encuentran resultados, mostramos una advertencia.
            st.warning("⚠️ No hay resultados con esos filtros. Intenta ajustar los parámetros.")
        else: # Si hay resultados, los mostramos.
            st.write(f"Se encontraron **{len(filtered_data)}** películas con los filtros seleccionados.")
            # st.columns: Organizamos las películas en varias columnas para una visualización tipo grilla.
            cols_per_row = 4 # Puedes ajustar esto
            num_rows = (len(filtered_data) + cols_per_row - 1) // cols_per_row # Calcular número de filas

            # Iteramos para mostrar cada película en su "tarjeta".
            for i in range(num_rows):
                row_cols = st.columns(cols_per_row)
                for j in range(cols_per_row):
                    idx = i * cols_per_row + j
                    if idx < len(filtered_data):
                        row = filtered_data.iloc[idx]
                        with row_cols[j]: # Colocamos la tarjeta en la columna correspondiente.
                            # st.markdown: Mostramos el HTML de la tarjeta de película.
                            st.markdown(f"""
                            <div style='margin: 10px; padding: 10px; text-align: center; background: #f0f2f6; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); height: 380px; display: flex; flex-direction: column; justify-content: space-between;'>
                                <img src='{row['poster_url']}' width='150' style='border-radius: 5px; margin: 0 auto; object-fit: cover; height: 225px;' onerror="this.onerror=null;this.src='https://via.placeholder.com/150x225.png?text=No+Poster';"/>
                                <div style="flex-grow: 1; display: flex; flex-direction: column; justify-content: flex-start; margin-top: 10px;">
                                    <h5 style='margin: 0 0 5px 0; color: #1e90ff;'>{row['titulo']} ({row['anio']})</h5>
                                    <p style='font-size: 0.9em; margin: 0;'>⭐ {row['puntuacion']:.1f} | 🕒 {int(row['duracion'])} min</p>
                                    <p style='font-size: 0.8em; margin: 5px 0 0 0; color: #555;'><i>{row['genero']}</i></p>
                                    <p style='font-size: 0.8em; margin: 0; color: #777;'>Plataforma: {row['plataforma']}</p>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
    elif reiniciar_button:
        # st.experimental_rerun(): Si se presiona "Reiniciar", se recarga toda la página
        # para que los filtros vuelvan a sus valores iniciales.
        st.experimental_rerun()
    else:
        # Mensaje inicial para guiar al usuario.
        st.info("Ajusta los filtros en la barra lateral izquierda y haz clic en 'Buscar Películas'.")
else:
    st.error("No se pudieron cargar los datos de películas para el recomendador. Asegúrate de que 'peliculas_completas.csv' esté en la carpeta 'data'.")

st.markdown("---") # Línea divisoria
# Mensaje final para el usuario.
st.success("¡Tu aplicación Streamlit está lista! Sube este archivo (`app.py`) junto con tus carpetas `data/` y `imagenes_directores_actores/` a un repositorio de GitHub para desplegarla en Streamlit Cloud.")
