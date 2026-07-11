# Predicción de Retornos Bancarios Mediante Mecanismos de Atención Cruzada

Este repositorio contiene la implementación matemática y de software de un sistema avanzado de **Machine Learning** diseñado para predecir los retornos financieros de acciones bancarias (específicamente JPMorgan Chase - JPM). En lugar de tratar a la acción como una serie de tiempo aislada, este proyecto implementa un modelo **MasterStockTransformer** con **Atención Cruzada (Cross-Attention)**, utilizando el contexto macroeconómico (índice XLF y bancos pares como BAC, C y WFC) para guiar direccionalmente la predicción de la red neuronal.

## Arquitectura y Resultados Principales

El proyecto contrasta el rendimiento de tres redes neuronales bajo los mismos hiperparámetros optimizados mediante Grid Search:
1. **Transformer MASTER (Full):** Arquitectura con Atención Cruzada utilizando los datos de mercado como tensores Key/Value.
2. **Baseline LSTM:** Modelo de memoria recurrente tradicional.
3. **Transformer de Ablación:** Arquitectura Transformer ciega al mercado externo (solo retroalimentada con la historia de JPM).

### Resultados (Prueba sobre los últimos 150 días)
| Modelo | MAE ($) | RMSE ($) | R² Score |
| :--- | :--- | :--- | :--- |
| **Transformer MASTER** | **2.8100** | **4.0188** | **0.9930** |
| Transformer Ablación | 2.8981 | 4.1068 | 0.9927 |
| Baseline LSTM | 3.2124 | 4.4012 | 0.9916 |

La arquitectura MASTER probó empíricamente que la inclusión de contexto macroeconómico mejora sustancialmente la precisión predictiva, reduciendo el error absoluto a tan solo \$2.81 dólares.

## Estructura del Repositorio

- `01_Data_Extraction_and_Preprocessing.ipynb`: Pipeline de automatización con `yfinance` para extraer precios de JPM, BAC, C, WFC y XLF, y transformación a retornos estacionarios (logarítmicos).
- `02_Model_Architecture.ipynb`: Definición teórica y pruebas dimensionales de los tensores de PyTorch.
- `03_Training_Loop.ipynb`: Entrenamiento principal de los tres modelos utilizando un decaimiento de Learning Rate.
- `03B_Hyperparameter_Tuning.ipynb`: Torneo analítico iterando combinaciones de dimensiones (d_model), capas (num_layers) y regularización (dropout) para mitigar el sobreajuste.
- `04_Evaluation_and_Visualizations.ipynb`: Sistema de validación científica que reconstruye el precio en dólares, genera tablas comparativas, analiza los mapas de calor de la atención (Heatmaps) y grafica distribuciones.
- `model.py`: Contiene las clases puras en PyTorch de `MasterStockTransformer`, `LSTMModel` y `StockDataset` para un código modularizado.
- `docs/`: Directorio que contiene el manuscrito formal del proyecto redactado en Typst (`main.typ`), su bibliografía (`biblio.bib`), y todos los gráficos renderizados de alta calidad (`img/`).

> Nota: Los pesos entrenados del modelo (`*.pth`), los sets de datos transaccionales brutos (`.csv`) y los papers de referencias externas han sido ignorados (`.gitignore`) para mantener la agilidad del repositorio.

## ⚙️ Tecnologías Utilizadas
- **Core ML:** PyTorch, Torch.nn
- **Análisis Matemático:** Pandas, NumPy, Scikit-learn
- **Visualización:** Matplotlib, Seaborn
- **Documentación Académica:** Typst
