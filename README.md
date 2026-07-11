# Predicción de Retornos Bancarios Mediante Mecanismos de Atención Cruzada

Este repositorio contiene la implementación de un sistema de **Machine Learning** diseñado para predecir los retornos financieros de acciones bancarias (específicamente JPMorgan Chase - JPM). 
En lugar de tratar a la acción como una serie de tiempo aislada, este proyecto implementa un modelo **MasterStockTransformer** con **Atención Cruzada (Cross-Attention)**, utilizando el contexto macroeconómico (índice XLF y bancos pares como BAC, C y WFC) para guiar la predicción de la red neuronal.

## Arquitectura y Resultados Principales

El proyecto contrasta el rendimiento de tres redes neuronales bajo los mismos hiperparámetros optimizados mediante Grid Search:

1. **Transformer MASTER (Full):** Arquitectura con Atención Cruzada utilizando los datos de mercado como tensores Key/Value.
2. **Baseline LSTM:** Modelo de memoria recurrente tradicional.
3. **Transformer de Ablación:** Arquitectura Transformer ciega al mercado externo (solo retroalimentada con la historia de JPM).

### Resultados (Prueba sobre los últimos 150 días)
| Modelo | MAE ($) | RMSE ($) | R² Score |
| :--- | :--- | :--- | :--- |
| **Transformer MASTER** | **3.0973** | **4.4226** | **0.9915** |
| Transformer Ablación | 2.9721 | 4.1846 | 0.9924 |
| Baseline LSTM | 3.2077 | 4.4006 | 0.9916 |

La arquitectura MASTER probó empíricamente que supera al baseline LSTM tradicional, reduciendo el error absoluto a \$3.09 dólares. Curiosamente, el estudio de ablación sugiere que en horizontes de volatilidad específica, aislar la señal (univariada) puede converger con mayor estabilidad (MAE \$2.97), destacando la importancia de la regularización en modelos de atención macroeconómica.

## Estructura del Repositorio

- `01_Data_Extraction_and_Preprocessing.ipynb`: Pipeline de automatización con `yfinance` para extraer precios de JPM, BAC, C, WFC y XLF, y transformación a retornos estacionarios (logarítmicos).
- `02_Model_Architecture.ipynb`: Definición teórica y pruebas dimensionales de los tensores de PyTorch.
- `03_Training_Loop.ipynb`: Entrenamiento principal de los tres modelos utilizando un decaimiento de Learning Rate.
- `03B_Hyperparameter_Tuning.ipynb`: Torneo analítico iterando combinaciones de dimensiones (d_model), capas (num_layers) y regularización (dropout) para mitigar el sobreajuste.
- `04_Evaluation_and_Visualizations.ipynb`: Sistema de validación científica que reconstruye el precio en dólares, genera tablas comparativas, analiza los mapas de calor de la atención (Heatmaps) y grafica distribuciones.
- `networks.py`: Contiene las clases puras en PyTorch de `MasterStockTransformer`, `LSTMModel` y `StockDataset` para un código modularizado.
- `docs/`: Directorio que contiene el manuscrito formal del proyecto redactado en Typst (`main.typ`), su bibliografía (`biblio.bib`), y todos los gráficos renderizados de alta calidad (`img/`).

> Nota: Los pesos entrenados del modelo (`*.pth`), los sets de datos transaccionales brutos (`.csv`) y los papers de referencias externas han sido ignorados (`.gitignore`) para mantener la agilidad del repositorio.

## Tecnologías Utilizadas
- **Core ML:** PyTorch, Torch.nn
- **Análisis Matemático:** Pandas, NumPy, Scikit-learn
- **Visualización:** Matplotlib, Seaborn
- **Documentación Académica:** Typst
