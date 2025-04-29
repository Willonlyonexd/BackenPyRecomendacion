from fastapi import FastAPI, Body, Query
import uvicorn
import pickle
import numpy as np
import pandas as pd
from pydantic import BaseModel
from typing import Optional
from datetime import datetime, timedelta
import threading
import time
import schedule
import re

# 1. FastAPI App
app = FastAPI()

# 2. Función para cargar modelo dinámicamente
def cargar_modelo():
    with open('modelo_recomendacion.pkl', 'rb') as f:
        modelo_data = pickle.load(f)
    return modelo_data

# Nuevas estructuras de datos para almacenar las recomendaciones por rango de fechas
ultima_prediccion_por_rango = None
fecha_ultimo_calculo = None
fecha_proxima_actualizacion = None

# Nuevos modelos de datos para la API
class RangoFechas(BaseModel):
    fecha_inicio: str  # formato: DD/MM/YYYY
    fecha_fin: str     # formato: DD/MM/YYYY

# Nueva función para extraer categorías de los títulos de productos
def extraer_categoria(titulo):
    """Extrae una categoría del título del producto."""
    titulo_lower = titulo.lower()
    
    # Patrones de categorías comunes
    if "mochila" in titulo_lower:
        return "Mochilas"
    elif "zapatilla" in titulo_lower or "zapato" in titulo_lower:
        return "Calzado"
    elif "pantalon" in titulo_lower or "jean" in titulo_lower:
        return "Pantalones"
    elif "camisa" in titulo_lower or "camiseta" in titulo_lower or "polo" in titulo_lower:
        return "Camisetas"
    elif "chaqueta" in titulo_lower or "abrigo" in titulo_lower:
        return "Abrigos"
    elif "bolso" in titulo_lower or "cartera" in titulo_lower:
        return "Bolsos"
    elif "reloj" in titulo_lower:
        return "Relojes"
    elif "gorra" in titulo_lower or "sombrero" in titulo_lower:
        return "Gorras"
    else:
        return "Otros"

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
                    'slug': producto_info.iloc[0].get('slug', ''),
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
    resultado = productos_top[['producto_id', 'titulo', 'slug']].to_dict(orient='records')
    return resultado

# Nueva función para calcular productos populares por rango de fechas
def obtener_productos_populares_por_rango(fecha_inicio, fecha_fin, n=10, diversidad=False):
    """
    Calcula los productos más populares basado en un rango de fechas específico
    
    Args:
        fecha_inicio: Fecha inicial en formato DD/MM/YYYY
        fecha_fin: Fecha final en formato DD/MM/YYYY
        n: Número de productos a recomendar
        diversidad: Si es True, intenta diversificar los resultados entre categorías
    """
    modelo = cargar_modelo()
    df_productos = modelo['df_productos']
    df_ventas = modelo['df_ventas']
    modelo_regresion = modelo['modelo_regresion']
    
    # Convertir fechas de formato DD/MM/YYYY a datetime
    try:
        fecha_inicio = datetime.strptime(fecha_inicio, "%d/%m/%Y")
        fecha_fin = datetime.strptime(fecha_fin, "%d/%m/%Y")
    except ValueError:
        return {"error": "Formato de fecha incorrecto. Use DD/MM/YYYY."}
    
    # Filtrar ventas en el rango especificado
    df_ventas_rango = df_ventas[
        (df_ventas['fecha'] >= fecha_inicio) & 
        (df_ventas['fecha'] <= fecha_fin)
    ]
    
    if df_ventas_rango.empty:
        return obtener_productos_populares(n)  # Usar método original si no hay datos
    
    # Calcular promedios de precios basados en el rango seleccionado
    precios_promedio = df_ventas_rango.groupby('producto_id')['precio'].mean().to_dict()
    
    # Crear datos para la predicción
    productos_unicos = df_productos.copy()
    productos_unicos['precio'] = productos_unicos['producto_id'].map(precios_promedio).fillna(100)
    
    # Usar el mes promedio del rango para la predicción
    mes_promedio = (fecha_inicio.month + fecha_fin.month) / 2
    mes_futuro = int(mes_promedio) if mes_promedio.is_integer() else mes_promedio
    
    X_pred = productos_unicos[['precio']].copy()
    X_pred['mes'] = mes_futuro
    
    # Realizar predicciones
    predicciones = modelo_regresion.predict(X_pred)
    productos_unicos['prediccion_cantidad'] = predicciones
    
    # Si queremos diversidad, extraemos categorías
    if diversidad:
        # Extraer categorías de los títulos de los productos
        productos_unicos['categoria'] = productos_unicos['titulo'].apply(extraer_categoria)
        
        # Lista para almacenar resultados diversificados
        resultado_diversificado = []
        categorias = productos_unicos['categoria'].unique()
        
        # Número máximo de productos por categoría
        max_por_categoria = max(1, n // len(categorias))
        
        # Para cada categoría, seleccionar los mejores productos
        for categoria in categorias:
            productos_cat = productos_unicos[productos_unicos['categoria'] == categoria]
            
            if not productos_cat.empty:
                productos_cat = productos_cat.sort_values(by='prediccion_cantidad', ascending=False)
                top_cat = productos_cat.head(max_por_categoria)
                
                # Añadir a resultados
                for _, producto in top_cat.iterrows():
                    if len(resultado_diversificado) < n:
                        resultado_diversificado.append({
                            'producto_id': producto['producto_id'],
                            'titulo': producto['titulo'],
                            'slug': producto.get('slug', ''),
                            'categoria': categoria
                        })
        
        # Si no tenemos suficientes productos diversificados, completar con los mejores
        if len(resultado_diversificado) < n:
            productos_faltantes = n - len(resultado_diversificado)
            ids_actuales = {p['producto_id'] for p in resultado_diversificado}
            
            productos_restantes = productos_unicos[~productos_unicos['producto_id'].isin(ids_actuales)]
            productos_restantes = productos_restantes.sort_values(by='prediccion_cantidad', ascending=False)
            
            for _, producto in productos_restantes.head(productos_faltantes).iterrows():
                resultado_diversificado.append({
                    'producto_id': producto['producto_id'],
                    'titulo': producto['titulo'],
                    'slug': producto.get('slug', ''),
                    'categoria': producto['categoria']
                })
        
        return resultado_diversificado
    else:
        # Método original: ordenar y seleccionar los N mejores
        productos_top = productos_unicos.sort_values(by='prediccion_cantidad', ascending=False).head(n)
        resultado = productos_top[['producto_id', 'titulo', 'slug']].to_dict(orient='records')
        return resultado

# Función para actualización automática programada
def actualizar_predicciones_automaticamente():
    """Actualiza las predicciones cada domingo a las 23:50"""
    global ultima_prediccion_por_rango, fecha_ultimo_calculo, fecha_proxima_actualizacion
    
    # Si no hay una predicción previa, usar últimos 10 días
    hoy = datetime.now()
    fecha_fin = hoy.strftime("%d/%m/%Y")
    fecha_inicio = (hoy - timedelta(days=10)).strftime("%d/%m/%Y")
    
    # Calcular nuevas predicciones con diversidad
    nueva_prediccion = obtener_productos_populares_por_rango(fecha_inicio, fecha_fin, diversidad=True)
    
    # Actualizar variables globales
    ultima_prediccion_por_rango = nueva_prediccion
    fecha_ultimo_calculo = hoy
    fecha_proxima_actualizacion = hoy + timedelta(days=(6 - hoy.weekday()) % 7 + 1)  # Próximo domingo
    
    print(f"Actualización automática realizada el {hoy}. Próxima actualización: {fecha_proxima_actualizacion}")

# Función para iniciar el programador en segundo plano
def iniciar_programador():
    # Programar la tarea para los domingos a las 23:50
    schedule.every().sunday.at("23:50").do(actualizar_predicciones_automaticamente)
    
    # Ejecutar la primera actualización inmediatamente
    actualizar_predicciones_automaticamente()
    
    # Mantener el programador ejecutándose
    while True:
        schedule.run_pending()
        time.sleep(60)  # Verificar cada minuto

# 4. Endpoints API
@app.get("/api/productos/populares")
async def get_productos_populares():
    """Endpoint original para productos populares"""
    global ultima_prediccion_por_rango, fecha_ultimo_calculo
    
    # Si tenemos una predicción por rango válida, la usamos
    if ultima_prediccion_por_rango and fecha_ultimo_calculo:
        return {
            "success": True, 
            "productos": ultima_prediccion_por_rango,
            "calculado_en": fecha_ultimo_calculo.isoformat(),
            "proxima_actualizacion": fecha_proxima_actualizacion.isoformat() if fecha_proxima_actualizacion else None
        }
    
    # De lo contrario, usamos el método original
    productos = obtener_productos_populares(n=10)
    return {"success": True, "productos": productos}

@app.post("/api/productos/populares/por-rango")
async def calcular_productos_populares_por_rango(
    rango: RangoFechas,
    diversidad: bool = Query(False, description="Activar diversificación entre categorías de productos")
):
    """Nuevo endpoint para calcular productos populares por rango de fechas"""
    global ultima_prediccion_por_rango, fecha_ultimo_calculo, fecha_proxima_actualizacion
    
    productos = obtener_productos_populares_por_rango(
        fecha_inicio=rango.fecha_inicio, 
        fecha_fin=rango.fecha_fin,
        n=10,
        diversidad=diversidad
    )
    
    # Actualizar las variables globales
    ultima_prediccion_por_rango = productos
    fecha_ultimo_calculo = datetime.now()
    
    # Calcular próxima actualización (próximo domingo)
    hoy = datetime.now()
    fecha_proxima_actualizacion = hoy + timedelta(days=(6 - hoy.weekday()) % 7 + 1)
    
    return {
        "success": True, 
        "productos": productos,
        "rango_analizado": {
            "inicio": rango.fecha_inicio,
            "fin": rango.fecha_fin
        },
        "calculado_en": fecha_ultimo_calculo.isoformat(),
        "proxima_actualizacion": fecha_proxima_actualizacion.isoformat(),
        "diversidad_aplicada": diversidad
    }

@app.get("/api/productos/recomendados/{cliente_id}")
async def get_recomendaciones_cliente(cliente_id: str):
    recomendaciones = recomendar_productos(cliente_id, n=5)
    return {"success": True, "recomendaciones": recomendaciones}



# 4. Endpoints API
@app.get("/health")
async def health_check():
    """
    Endpoint de health check para mantener la aplicación activa en Render
    """
    global contador_health_checks
    contador_health_checks += 1
    
    tiempo_activo = datetime.now() - inicio_servidor
    dias = tiempo_activo.days
    horas, remainder = divmod(tiempo_activo.seconds, 3600)
    minutos, segundos = divmod(remainder, 60)
    
    return {
        "status": "OK",
        "timestamp": datetime.now().isoformat(),
        "uptime": f"{dias} días, {horas} horas, {minutos} minutos, {segundos} segundos",
        "health_checks_realizados": contador_health_checks,
        "ultima_actualizacion_predicciones": fecha_ultimo_calculo.isoformat() if fecha_ultimo_calculo else None,
        "proxima_actualizacion_predicciones": fecha_proxima_actualizacion.isoformat() if fecha_proxima_actualizacion else None
    }

# 5. Ejecutar servidor
if __name__ == "__main__":
    # Iniciar el hilo para la actualización programada
    threading_programador = threading.Thread(target=iniciar_programador, daemon=True)
    threading_programador.start()
    
    # Iniciar servidor FastAPI
    uvicorn.run(app, host="0.0.0.0", port=8000)