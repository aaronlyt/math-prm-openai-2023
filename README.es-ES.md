

# math-prm-openai-2023
reproducir el proceso de construcción de datos de Let’s Verify Step by Step y MATH-SHEPHERD con fines de aprendizaje 

Artículo reproducido: Let’s Verify Step by Step
Otros artículos de referencia:
Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters
MATH-SHEPHERD: VERIFY AND REINFORCE LLMS STEP-BY-STEP WITHOUT HUMAN ANNOTATIONS

notas: es posible que algunas rutas necesiten configurarse manualmente; además, está destinado principalmente al aprendizaje, por lo que la legibilidad es promedio. Si estás interesado, puedes consultarlo como referencia.

Contenido principal de los experimentos:
1. Entrenamiento de un modelo de verificación (verify model) basado en el conjunto de datos Grade School Math.
2. Entrenamiento de un modelo de recompensa (reward model) siguiendo Let’s Verify Step by Step:
- Utilización del conjunto de datos original.
- Construcción de un conjunto de datos de recompensa PRM siguiendo MATH-SHEPHERD y entrenamiento del modelo de recompensa.


I. Experimentos con el modelo de verificación (verify model)

1. Marcos de trabajo: trl, transformers 
2. Modelo base: qwen2.5 1.5b BASE
3. Datos de entrenamiento: conjunto de datos de entrenamiento de Grade School Math
4. Flujo de ejecución
- Ajuste fino (SFT) del modelo base durante 2 épocas: matho1/scripts/sft_gsm.py
- Construcción de datos de muestreo para entrenar el verify model a partir del modelo SFT (inferencia con múltiples instancias de vllm): matho1/inference/grade_math/sample_4verifymodel_.py
- Entrenamiento del verify model: matho1/scripts/verify_gsm_trainer.py o matho1/scripts/verify_gsm.py
- Clasificación de los datos de muestreo del modelo SFT utilizando el verify model: matho1/inference/grade_math/verify_rank.py
- Script de evaluación: matho1/inference/grade_math/eval_ranker.py
5. Resultados de evaluación: conjunto de prueba de Grade School Math, precisión (accuracy): 0.4147

II.  Entrenamiento del modelo de recompensa (reward model) siguiendo Let’s Verify Step by Step

1. Marcos de trabajo: trl, transformers 
2. Modelo base: qwen2.5 1.5b BASE
3. Datos de entrenamiento: conjunto de datos prm800k
4. Flujo de ejecución con prm800k
- Preprocesamiento de datos prm800k: matho1/inference/prm800k/prm_dataset.py： prod Defaults to False. Si hay múltiples rutas, solo se selecciona una; si es True, se combinan múltiples opciones.
- Generación del conjunto de datos de entrenamiento SFT para el generador mediante few-shot: matho1/inference/prm800k/fewshot_4gensft.py
- Ajuste fino (SFT) del generador: matho1/scripts/sft_prm.py
- Muestreo y generación de datos de entrenamiento para el reward model PRM utilizando el generador entrenado: matho1/inference/prm800k/gen_mathslu.py, también puede utilizarse para muestrear el conjunto de datos de prueba.
- Entrenamiento del modelo de recompensa PRM: matho1/scripts/nexttok_prm_trainer.py, Los PRMs se entrenan durante 2 épocas. En conjuntos de datos más pequeños (como en la fase 1 y las primeras generaciones de la fase 2), esto mejora el rendimiento final en comparación con entrenar solo durante 1 época.
- Uso del modelo de recompensa de proceso para clasificar el muestreo del generador en el conjunto de prueba: matho1/scripts/prm_rank.py
- Evaluación estadística: matho1/scripts/gpstate_ranker_score
Resultados de evaluación: no supera el rendimiento de la votación por mayoría (majority voting).
- Mejor resultado experimental: precisión: 0.46 (el resultado óptimo se obtuvo con prod=True)
- Votación por mayoría con generador SFT: precisión: 0.508

Análisis: Posible problema de inconsistencia en la distribución de los datos. Según Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters: Encontramos que era fácil explotar un PRM entrenado en este conjunto de datos incluso con estrategias ingenuas como el muestreo best-of-N. Hipotetizamos que esto probablemente sea resultado del desplazamiento de distribución entre las muestras generadas por GPT-4 en su conjunto de datos y nuestros modelos PaLM 2.

Siguiente: Por lo tanto, se construirá el conjunto de datos MATH-SHEPHERD siguiendo el método de Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters.

III  Flujo de construcción de datos de MATH-SHEPHERD
- Preprocesamiento de datos prm800k: matho1/inference/prm800k/prm_dataset.py
- Generación del conjunto de datos de entrenamiento SFT para el generador mediante few-shot: matho1/inference/prm800k/fewshot_4gensft.py
- Ajuste fino (SFT) del generador: matho1/scripts/sft_prm.py------Aquí no es obligatorio realizar el ajuste fino; no hubo tiempo para realizar experimentos comparativos.
- Muestreo y generación de datos de entrenamiento para el reward model PRM utilizando el generador entrenado: matho1/inference/prm800k/gen_mathslu.py, también puede utilizarse para muestrear el conjunto de datos de prueba.
- Organización de los datos en formato de pasos (steps) para una inferencia eficiente: matho1/inference/prm800k/shepsherd_steps.py
- Código central de MATH-SHEPHERD - muestreo para construir señales de recompensa: matho1/inference/prm800k/math_shepherd_data.py
- Postprocesamiento de datos: matho1/inference/prm800k/math_shepherd_agg.py, agregar el conjunto de datos de muestreo de math-shepherd, ejecutar después de math_shepherd_data.
- Entrenamiento del modelo de recompensa PRM: matho1/scripts/nexttok_prm_trainer.py
- Uso del modelo de recompensa de proceso para clasificar el muestreo del generador en el conjunto de prueba: matho1/scripts/prm_rank.py
- Evaluación estadística: matho1/scripts/gpstate_ranker_score
Resultados de evaluación: precisión: 0.418

Problemas existentes: durante el proceso de construcción de la señal de recompensa, no es posible muestrear un conjunto de datos suficiente; se sospecha que podría deberse a sobreajuste (overfitting) en el SFT.

Acciones futuras posibles: 
- Generar los datos utilizando el enfoque de modelo base + few-shot descrito en el artículo Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters.
- O seguir el enfoque presentado en el artículo MATH-SHEPHERD.
