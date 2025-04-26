# app.py

from fastapi import FastAPI
import uvicorn
import pickle
import numpy as np
import pandas as pd

# 1. FastAPI App
app = FastAPI()

# 2. Función para cargar modelo dinámicamente
def cargar_modelo():
    with open('modelo_recomendacion.pkl', 'rb') as f:
        modelo_data = pickle.load(f)
    return modelo_data

# 3. Funciones de recomendación
def recomendar_productos(cliente_id, n=5):
    modelo = cargar_modelo()

    matriz_dispersa = modelo['matriz_dispersa']
    similitud_usuarios = modelo['similitud_usuarios']
    indices_clientes = modelo['indices_clientes']
    indices_productos = modelo['indices_productos']
    df_productos = modelo['df_productos']

    if cliente_id not in indices_clientes:
        return []

    idx = np.where(indices_clientes == cliente_id)[0][0]
    scores_cliente = matriz_dispersa[idx].toarray().flatten()
    scores_recomendacion = np.zeros_like(scores_cliente)

    for i, similar_user_score in enumerate(similitud_usuarios[idx]):
        if i != idx:
            scores_i = matriz_dispersa[i].toarray().flatten()
            for j in range(len(scores_i)):
                if scores_cliente[j] == 0 and scores_i[j] > 0:
                    scores_recomendacion[j] += similar_user_score * scores_i[j]

    productos_recomendados_idx = np.argsort(scores_recomendacion)[::-1][:n]
    recomendaciones = []
    for idx in productos_recomendados_idx:
        if scores_recomendacion[idx] > 0:
            producto_id = indices_productos[idx]
            producto_info = df_productos[df_productos['producto_id'] == producto_id]
            if not producto_info.empty:
                recomendaciones.append({
                    'id': producto_id,
                    'titulo': producto_info.iloc[0]['titulo'],
                    'puntuacion': round(float(scores_recomendacion[idx]), 3)
                })
    return recomendaciones

def obtener_productos_populares(n=10):
    modelo = cargar_modelo()

    df_productos = modelo['df_productos']
    df_ventas = modelo['df_ventas']
    modelo_regresion = modelo['modelo_regresion']

    mes_futuro = 7
    productos_unicos = df_productos.copy()
    precios_promedio = df_ventas.groupby('producto_id')['precio'].mean().to_dict()
    productos_unicos['precio'] = productos_unicos['producto_id'].map(precios_promedio).fillna(100)

    X_pred = productos_unicos[['precio']].copy()
    X_pred['mes'] = mes_futuro

    predicciones = modelo_regresion.predict(X_pred)
    productos_unicos['prediccion_cantidad'] = predicciones

    productos_top = productos_unicos.sort_values(by='prediccion_cantidad', ascending=False).head(n)
    resultado = productos_top[['producto_id', 'titulo']].to_dict(orient='records')
    return resultado

# 4. Endpoints API
@app.get("/api/productos/populares")
async def get_productos_populares():
    productos = obtener_productos_populares(n=10)
    return {"success": True, "productos": productos}

@app.get("/api/productos/recomendados/{cliente_id}")
async def get_recomendaciones_cliente(cliente_id: str):
    recomendaciones = recomendar_productos(cliente_id, n=5)
    return {"success": True, "recomendaciones": recomendaciones}

# 5. Ejecutar servidor
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
