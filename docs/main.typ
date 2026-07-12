#import "@preview/charged-ieee:0.1.4": ieee
#set text(lang: "es")
#show link: underline

#show: ieee.with(
  title: [Predicción de Retornos Bancarios Mediante Mecanismos de Atención Cruzada],
  abstract: [
    El pronóstico financiero es un desafío clásico en Machine Learning debido a la alta volatilidad del mercado. 
    
    Tradicionalmente, se usan modelos que analizan las acciones de forma aislada. Como aplicación empírica, este proyecto busca predecir los retornos de JPMorgan Chase utilizando redes neuronales basadas en Transformers. En lugar de utilizar arquitecturas complejas de la literatura reciente, se implementará una adaptación simplificada inspirada en el modelo MASTER. El enfoque central es explorar el mecanismo de atención cruzada (Cross-Attention), utilizando el comportamiento histórico de bancos competidores (Bank of America, Citigroup y Wells Fargo) como contexto para guiar las predicciones sobre el banco objetivo. Este diseño permite reducir el costo computacional del entrenamiento y analizar si la red logra capturar cómo las instituciones financieras se influyen mutuamente. Finalmente, el rendimiento de esta implementación se comparará de forma práctica contra modelos de línea base tradicionales vistos en el estado del arte, evaluando métricas como el Error Absoluto Medio (MAE) para determinar si la integración multivariada de bancos competidores justifica el uso de algoritmos de aprendizaje profundo en la gestión de riesgos.
  ],
  authors: (
   
    (
      name: "Borjas Bernaola, Gerald Marcelo Fernando",
      organization: [Universidad de Ingenieria y Tecnologia],
      location: [Lima, Peru],
      email: "gerald.borjas@utec.edu.pe"
    ),
  ),
  index-terms: ("Aprendizaje profundo", "Atención cruzada", "Pronóstico financiero", "Redes neuronales Transformer", "Series temporales multivariadas"),
  bibliography: bibliography("biblio.bib"),
  figure-supplement: [Fig.],
)

= Introducción

La predicción precisa de activos financieros es altamente compleja debido a la volatilidad del mercado y la influencia de factores macroeconómicos externos. La mayoría de los enfoques de Machine Learning modelan el precio de una acción basándose únicamente en su propio historial (autorregresión), fallando en capturar correlaciones temporales cruzadas ("cross-time") generadas por el comportamiento global del mercado. En el sector bancario, comprender cómo las fluctuaciones del mercado impactan a una entidad específica es crítico para la gestión del riesgo.

== Objetivos del proyecto

	1.	Desarrollar e implementar una arquitectura basada en Transformers con un mecanismo de atención cruzada (Cross-Attention) adaptado para datos financieros de alta frecuencia.
	2.	Reducir la complejidad dimensional modelando el "estado del mercado" a través de un panel selecto de los principales bancos pares (Bank of America, Citigroup, Wells Fargo) y el índice XLF.
	3.	Evaluar la precisión predictiva del modelo propuesto utilizando métricas de error (MAE, RMSE) en la predicción del comportamiento de JPMorgan Chase (JPM), comparándolo con modelos convencionales.
 
== Dataset utilizado

Se utilizó un dataset creado para el proyecto mediante la extracción de datos históricos diarios a través de la API `yfinance`. El conjunto comprende las series temporales del precio de cierre ajustado y volumen de transacciones del banco objetivo (JPM) y un panel de características de bancos guía (BAC, C, WFC) junto con el ETF financiero XLF durante la última década. Los datos crudos fueron preprocesados en retornos logarítmicos y métricas de volatilidad rodante, garantizando la estacionariedad y la coherencia matemática para el entrenamiento de la red.

= Revisión bibliográfica

El modelamiento de series temporales financieras ha evolucionado hacia arquitecturas de aprendizaje profundo capaces de capturar dependencias no lineales, superando a los modelos recurrentes como LSTM @hochreiter1997long. La literatura identifica los mecanismos de atención, introducidos por @vaswani2017attention, como herramientas robustas para abordar la volatilidad @lim2021temporal.

En la base de esta investigación se encuentra el trabajo de @li2024master, quienes introducen el modelo MASTER, argumentando que las correlaciones entre activos ocurren de manera momentánea y cruzada. Su propuesta de guiar la predicción con información global fundamenta el uso de atención cruzada en este proyecto. Asimismo, @li2025transformer proponen "Stockformer", reforzando que el pronóstico debe tratarse como un problema multivariado y no como una autorregresión aislada, idea soportada también por @zhang2025emat y @chen2024cmtf.

Pese a la potencia de los Transformers, @wang2023stock señala debilidades en la captura de patrones locales, sugiriendo arquitecturas híbridas. No obstante, @mozaffari2024predictive demuestran que la arquitectura de atención multi-cabezal supera consistentemente a modelos recurrentes como LSTM en la caracterización de dinámicas de mercado. Finalmente, @muhammad2022transformer validan la robustez de estos modelos en entornos de alta volatilidad.

Para contextualizar el avance de estas arquitecturas, es vital considerar la evolución detallada en estudios comprehensivos como el de @wen2023transformers, quienes exponen los retos de adaptar Transformers a series de tiempo. Para mitigar estos problemas, propuestas recientes como Autoformer @wu2021autoformer y FEDformer @zhou2022fedformer introducen mecanismos de descomposición en frecuencia para predicciones a largo plazo. Más aún, la integración de análisis de sentimiento mediante Modelos Fundacionales (LLMs) ha comenzado a fusionarse con Transformers @chen2024sentiment, demostrando que el futuro de la predicción radica en el análisis de eventos complejos @ding2020deep.

= Metodología

La metodología de esta investigación se divide en tres fases principales: la ingeniería de características para el modelo de atención, la optimización de hiperparámetros (Grid Search) y el entrenamiento de tres arquitecturas de red neuronal para el estudio comparativo.

== Ingeniería de Características
Para evitar la trampa académica del "Look-ahead Bias", los datos fueron separados cronológicamente en conjuntos de Entrenamiento (70%), Validación (15%) y Prueba (15%). Se calcularon indicadores técnicos como MACD y RSI sobre el precio de JPM. La entrada de la red neuronal se estructuró en dos tensores diferenciados:
- *Tensor Query (Objetivo):* Secuencia histórica de JPM.
- *Tensor Key/Value (Contexto):* Comportamiento histórico simultáneo de los bancos rivales y el índice macroeconómico (XLF).

== Optimización de Hiperparámetros (Grid Search)

Para asegurar el rigor científico, se programó una búsqueda de cuadrícula (Grid Search) automatizada. Se evaluaron 16 configuraciones combinando el número de capas de atención (1 vs 2), la dimensión del modelo (64 vs 128), el dropout (0.1 vs 0.2) y el número de épocas (30 vs 50). La configuración óptima que minimizó el Error Cuadrático Medio (MSE) en el conjunto de validación fue: `d_model=64`, `num_layers=2`, `dropout=0.1` en `50 épocas`.

== Arquitecturas Entrenadas

Se entrenaron tres arquitecturas bajo las mismas condiciones óptimas:
1. *MasterStockTransformer:* El modelo propuesto con atención cruzada hacia el mercado completo.
2. *Baseline LSTM:* Red neuronal recurrente tradicional para establecer una línea base comparativa.
3. *Estudio de Ablación:* Un Transformer ciego al mercado, entrenado únicamente con los datos aislados de JPM, para probar empíricamente el valor del contexto macroeconómico.

= Resultados y Discusión

Los modelos fueron evaluados reconstruyendo el precio absoluto (en dólares) a partir de los retornos logarítmicos predichos sobre un horizonte temporal de 150 días de prueba.

== Comparativa de Precisión (Leaderboard)

El *MasterStockTransformer* superó significativamente al modelo LSTM clásico. Al medir el Error Absoluto Medio (MAE), el modelo Transformer logró predecir el precio con un margen de error de tan solo \$3.31 dólares, frente a los \$4.49 dólares del LSTM. Asimismo, el RMSE del Transformer fue de \$4.59 frente a los \$5.99 del LSTM, con un R² sobresaliente de 0.9908. Esta reducción del error en más de 1 dólar por acción justifica contundentemente la superioridad de la arquitectura basada en atención para series de tiempo financieras de alta volatilidad. Adicionalmente, el análisis de los residuales mostró una distribución normal centrada en cero (ruido blanco), demostrando ausencia de sesgo predictivo.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    align: (left, right, right, right, right, right),
    [*Modelo*], [*MAE (\$)*], [*MSE*], [*RMSE (\$)*], [*MAPE (%)*], [*R² Score*],
    [Transformer MASTER (Full)], [3.3145], [21.0919], [4.5926], [1.3061], [0.9908],
    [Baseline LSTM], [4.4927], [35.9272], [5.9939], [1.8070], [0.9844],
    [Transformer Ablación (Solo JPM)], [4.1339], [32.8585], [5.7322], [1.6379], [0.9857],
  ),
  caption: [Leaderboard de Precisión Predictiva]
)

#figure(
  image("img/graph1.png", width: 95%),
  caption: [Comparativa de Predicciones Reconstruidas (Últimos 150 días)]
)

#figure(
  image("img/graph2.png", width: 95%),
  caption: [Comparativa de Errores: RMSE y MAE]
)

#figure(
  image("img/graph3.png", width: 95%),
  caption: [Precisión del Modelo MASTER (Scatter Plot)]
)

#figure(
  image("img/graph4.png", width: 95%),
  caption: [Distribución de Errores (Predicción - Real)]
)

== El Impacto del Contexto Macro (Ablación)

El estudio de ablación confirmó de nuevo la hipótesis principal del proyecto: el Transformer entrenado *sin* el contexto del mercado (XLF, BAC, C, WFC) obtuvo peores métricas (MAE de \$4.13 y RMSE de \$5.73) que el modelo entrenado con el panel completo. Esto demuestra empíricamente, y bajo control estocástico riguroso, que la integración multivariada mediante atención cruzada logra capturar exitosamente la influencia sistémica del mercado sobre el precio individual de JPMorgan.

== Interpretabilidad: Mapa de Calor de Atención

A diferencia de los modelos de "caja negra" convencionales (como el LSTM), el mecanismo de atención cruzada permitió extraer un Mapa de Calor (Heatmap) de los pesos probabilísticos de la última inferencia. La matriz resultante reveló un patrón de fuertes bandas verticales (especialmente en índices específicos más recientes como el t-1 y el t-4, correspondientes a las columnas 19 y 16). Esto demuestra matemáticamente que la red neuronal no pondera el pasado de forma lineal decreciente, sino que el estado actual de la acción (Query) presta picos de atención intensa a choques o eventos sistémicos específicos del mercado (Key) ocurridos en el pasado reciente, validando la capacidad del modelo para extraer "memoria" asimétrica de factores exógenos.

#figure(
  image("img/graph5.png", width: 95%),
  caption: [Mapa de Calor de Atención Cruzada (JPM Query vs Macro Key)]
)

= Conclusiones

Esta investigación empírica demuestra que el uso de mecanismos de Atención Cruzada (Cross-Attention) en arquitecturas Transformer mejora significativamente la predicción direccional de activos financieros individuales al integrarlos con el contexto macroeconómico. Al lograr un R² de 0.9908 y reducir el error absoluto a \$3.31 frente al baseline LSTM (\$4.49) y al modelo de ablación (\$4.13), el *MasterStockTransformer* evidencia que el comportamiento del mercado dicta, en gran medida, el destino de una acción individual. La visualización de los pesos de atención cruzada proporciona una herramienta robusta, precisa e interpretable para la gestión cuantitativa de riesgos e inversiones.

= Trabajo Futuro

Si bien el modelo *MasterStockTransformer* demostró un rendimiento predictivo sobresaliente, existen múltiples vías de mejora para investigaciones futuras:
1. *Nuevos Datasets y Modalidades (Multimodalidad):* Incorporar análisis de sentimiento de noticias financieras, reportes de ganancias y redes sociales mediante el uso de Modelos de Lenguaje Grande (LLMs). Como sugiere la literatura reciente, fusionar datos numéricos con "Alpha verbal" podría capturar la irracionalidad del mercado que los precios por sí solos no muestran.
2. *Mejoras a la Arquitectura:* Explorar arquitecturas más complejas como *Graph Transformers* para modelar la red completa de relaciones entre cientos de acciones simultáneamente, en lugar de un panel estático. Asimismo, implementar mecanismos de descomposición de frecuencias (como en *Autoformer*) podría mejorar la predicción en horizontes de tiempo de largo plazo (multi-step forecasting).
3. *Simulación de Trading:* Traducir las métricas de error puro (RMSE/MAE) en una estrategia de trading simulada (Backtesting) considerando costos de transacción y comisiones reales (slippage), para medir la rentabilidad económica neta del algoritmo.

= Anexos

== Código Implementado

La totalidad de los algoritmos desarrollados para esta investigación, incluyendo los canales de extracción de datos automatizados vía API (`yfinance`), la arquitectura matemática neuronal (Transformer y LSTM en PyTorch), los scripts del torneo de hiperparámetros (Grid Search) y el entorno visual analítico (Matplotlib/Seaborn), se encuentran versionados y disponibles para la comunidad. El código fuente completo está entregado como un anexo interactivo del proyecto final en el siguiente repositorio de GitHub:

#align(center)[#link("https://github.com/gmborjasb/ML_MasterStockTransformer")[https://github.com/gmborjasb/ML_MasterStockTransformer]]

Este repositorio incluye todo el entorno necesario para garantizar la total reproducibilidad científica de los resultados obtenidos (utilizando semillas) e incluye instrucciones de ejecución paso a paso.