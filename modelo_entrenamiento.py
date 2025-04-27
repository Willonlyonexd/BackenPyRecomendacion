# modelo_entrenamiento.py

import pandas as pd
import numpy as np
import pickle
import certifi
from pymongo import MongoClient
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

# 1. Conexión a MongoDB
ca = certifi.where()
client = MongoClient(
    "mongodb+srv://will:will@cluster0.rmkpe.mongodb.net/EcommerML?retryWrites=true&w=majority",
    tlsCAFile=ca
)
db = client['EcommerML']

# 2. Extracción de datos
ventas_detalles = list(db.ventadetalles.find({}, {'cliente': 1, 'producto': 1, 'cantidad': 1, 'precio': 1, 'createdAT': 1}))
productos = list(db.productos.find({}, {'_id': 1, 'titulo': 1, 'slug': 1}))  # ⚡ Agregado 'slug'
clientes = list(db.clientes.find({}, {'_id': 1}))

df_ventas = pd.DataFrame(ventas_detalles)
df_productos = pd.DataFrame(productos)
df_clientes = pd.DataFrame(clientes)

# 3. Preprocesamiento
extract_oid = lambda x: str(x['$oid']) if isinstance(x, dict) and '$oid' in x else str(x)
extract_date = lambda x: x['$date'] if isinstance(x, dict) and '$date' in x else x

df_ventas['cliente_id'] = df_ventas['cliente'].apply(extract_oid)
df_ventas['producto_id'] = df_ventas['producto'].apply(extract_oid)
df_ventas['fecha'] = pd.to_datetime(df_ventas['createdAT'].apply(extract_date), errors='coerce')
df_ventas = df_ventas[['cliente_id', 'producto_id', 'cantidad', 'precio', 'fecha']]

df_productos['producto_id'] = df_productos['_id'].apply(extract_oid)

# 4. Matriz cliente-producto
compras_agregadas = df_ventas.groupby(['cliente_id', 'producto_id'])['cantidad'].sum().reset_index()
matriz_compras = compras_agregadas.pivot(index='cliente_id', columns='producto_id', values='cantidad').fillna(0)
matriz_dispersa = csr_matrix(matriz_compras.values)

# 5. Similitudes
similitud_usuarios = cosine_similarity(matriz_dispersa)
similitud_productos = cosine_similarity(matriz_dispersa.T)

# 6. Modelo de Regresión para popularidad
df_ventas['mes'] = df_ventas['fecha'].dt.month
df_modelo = df_ventas.groupby(['producto_id', 'mes']).agg({'cantidad': 'sum', 'precio': 'mean'}).reset_index()

X = df_modelo[['precio', 'mes']]
y = df_modelo['cantidad']

modelo_regresion = LinearRegression()
modelo_regresion.fit(X, y)

# 7. Guardar modelo y datos
modelo_data = {
    'matriz_dispersa': matriz_dispersa,
    'similitud_productos': similitud_productos,
    'similitud_usuarios': similitud_usuarios,
    'indices_clientes': matriz_compras.index,
    'indices_productos': matriz_compras.columns,
    'df_productos': df_productos,
    'df_ventas': df_ventas,
    'modelo_regresion': modelo_regresion
}

with open('modelo_recomendacion.pkl', 'wb') as f:
    pickle.dump(modelo_data, f)

print("✅ Modelo de recomendación guardado como modelo_recomendacion.pkl")
#hola