# Amazon Reviews Spanish — Clasificación de Sentimiento y Recuperación

Proyecto de NLP que clasifica el sentimiento de 208.899 reseñas de Amazon en español en negativo, neutro y positivo, comparando un baseline de bolsa de palabras contra un transformer español ajustado, y construyendo un sistema de recuperación sobre el mismo corpus.

- **Problema:** Recuperar el sentimiento del cliente únicamente a partir del texto de la reseña, y hacer consultables en lenguaje natural los motivos que hay detrás
- **Resultado:** F1 macro de 0,765 con BETO frente a 0,725 del baseline TF-IDF — y F1 de 0,85 a 0,88 en los dos polos, invirtiendo el signo en solo el 1 % de los casos
- **Valor:** El transformer gana por cuatro puntos y cuesta **1.009 veces más por predicción**, lo que convierte la comparación en una decisión de despliegue en lugar de un ranking

> [View this project in English](README.md)

---

## Tabla de contenidos

1. [Definición del problema](#definición-del-problema)
2. [Valor de negocio](#valor-de-negocio)
3. [Dataset](#dataset)
4. [Retos y transformaciones de los datos](#retos-y-transformaciones-de-los-datos)
5. [Análisis exploratorio de datos](#análisis-exploratorio-de-datos)
6. [Análisis estadístico](#análisis-estadístico)
7. [Metodología](#metodología)
8. [Baseline — TF-IDF y regresión logística](#baseline--tf-idf-y-regresión-logística)
9. [Ajuste fino de BETO](#ajuste-fino-de-beto)
10. [Comparación de modelos](#comparación-de-modelos)
11. [Sistema de recuperación (RAG)](#sistema-de-recuperación-rag)
12. [Limitaciones](#limitaciones)
13. [Conclusiones](#conclusiones)
14. [Posibles mejoras](#posibles-mejoras)
15. [Requisitos](#requisitos)

---

## Definición del problema

Los marketplaces acumulan opinión de clientes más rápido de lo que nadie puede leerla. La puntuación en estrellas es fácil de agregar pero solo dice *cuánto* de satisfecho estaba el cliente; el texto dice *por qué*, y es la parte que no escala.

Este proyecto aborda dos preguntas:

> **¿Se puede recuperar el sentimiento únicamente a partir del texto de la reseña?**
>
> **¿Se puede después consultar el corpus en lenguaje natural, con respuestas fundamentadas en reseñas reales?**

La primera es un problema de **clasificación supervisada de tres clases**. La puntuación proporciona la etiqueta, agrupada en negativo (1-2 estrellas), neutro (3) y positivo (4-5). Predecir el nivel exacto de estrellas no es el objetivo deliberadamente: la diferencia entre cuatro y cinco estrellas refleja más la escala personal de cada cliente que nada de lo que dice la reseña.

---

## Valor de negocio

**El modelo hace una cosa: distinguir satisfacción de insatisfacción, y lo hace bien.** De las reseñas realmente negativas, el 83 % se clasifican correctamente y el 1 % se etiquetan como positivas. En las positivas, 85 % y 1 %.

La forma de ese perfil de error es lo que lo hace utilizable. **El modelo prácticamente nunca invierte el signo.** Cuando falla en una reseña polar no afirma lo contrario, se repliega al centro. Para cualquier proceso que enrute reseñas por sentimiento, un cliente descontento como mucho se queda sin enrutar, nunca se archiva como satisfecho.

Tres aplicaciones se derivan directamente:

- **Puntuar texto que no lleva valoración asociada.** Tickets de soporte, respuestas de encuesta y menciones en redes contienen opinión de clientes sin ninguna estrella. El mismo modelo se aplica sin reentrenar.
- **Detectar deterioro antes de que se mueva la media.** La media de estrellas de un producto se calcula sobre todo su histórico y reacciona despacio. Clasificar las reseñas según entran hace visible el cambio de inmediato.
- **Hacer consultables los motivos.** El sistema de recuperación convierte *¿de qué se quejan los clientes en esta categoría?* en una consulta en lugar de una semana de lectura manual.

---

## Dataset

- **Fuente:** [Amazon Reviews Multi — Kaggle](https://www.kaggle.com/datasets/mexwell/amazon-reviews-multi)
- **Registros:** 210.000 reseñas en español antes de la limpieza, 208.899 después
- **Distribución:** se entrega en tres ficheros (`train` 200.000 / `validation` 5.000 / `test` 5.000) que cubren seis idiomas; solo se usa el español

| Columna | Contenido |
|---|---|
| `review_body` | Texto libre de la reseña |
| `review_title` | Título escrito por el cliente |
| `stars` | Puntuación de 1 a 5, de la que se deriva la etiqueta objetivo |
| `product_category` | Categoría de producto, 30 valores |
| `product_id`, `reviewer_id` | Identificadores anonimizados |
| `review_id`, `language` | Identificador de reseña y código de idioma |

**Variable objetivo:**

| Sentimiento | Estrellas | Reseñas |
|---|---|---|
| Negativo | 1, 2 | 83.545 |
| Neutro | 3 | 41.828 |
| Positivo | 4, 5 | 83.526 |

---

## Retos y transformaciones de los datos

Los tres ficheros de origen se fusionaron en un único DataFrame con una columna `split` que registra el fichero de procedencia de cada fila. Esa columna es lo que hace la fusión reversible: la partición de los autores, que existe para que los resultados sean comparables entre estudios, queda preservada dentro de los datos.

### Medir antes de limpiar

Cada artefacto se contó antes de escribir ninguna regla. **Ningún artefacto alcanza el 0,4 % del corpus:**

| Artefacto | `review_body` | `review_title` | Decisión |
|---|---|---|---|
| Emojis | 786 (0,37 %) | 351 (0,17 %) | Filas eliminadas |
| URLs | 1 | 0 | Fila eliminada |
| Etiquetas HTML | 2 | 0 | Filas eliminadas |
| Entidades HTML | 0 | 0 | Sin tratamiento |
| Espacios anómalos | 0 | 0 | Normalizados igualmente |
| Valores nulos | 0 | 0 | — |

El marcado HTML es habitual en corpus de reseñas extraídos de la web, así que su práctica ausencia merece señalarse: el dataset venía saneado de origen, y **la fase de eliminación de HTML que un pipeline así incluiría normalmente no llegó a escribirse**, porque la medición demostró que no había nada que eliminar.

### Por qué eliminar en lugar de transformar

Cada artefacto encontrado se resolvió eliminando las filas afectadas, por tres motivos: los recuentos son lo bastante bajos como para que la eliminación no pueda desplazar ninguna distribución; reescribir un artefacto inserta cadenas que ningún cliente escribió, que es la forma de una característica espuria a la que un modelo puede agarrarse; y sobreviven más de 208.000 reseñas, así que nada aquí está limitado por el tamaño de muestra.

**Total eliminado: 1.101 filas, el 0,52 % del corpus.**

### Sin filtro de longitud

Descartar reseñas muy cortas es un paso rutinario en pipelines de texto. Leer los extremos lo zanjó: los cuerpos más cortos son dos palabras del tipo *muy bien* o *muy mal*, y los títulos más cortos una sola palabra como *Excelente* o *Devuelto*. **Una reseña corta aquí es un cliente que dijo lo que pensaba en el mínimo de palabras posible**, no una reseña poco informativa. Un filtro de longitud mínima habría eliminado algunos de los ejemplos más claros del corpus.

### Columnas resultantes

| Columna | Contenido |
|---|---|
| `review_body`, `review_title` | Texto original, nunca modificado |
| `body_clean`, `title_clean` | Texto con espacios normalizados, usado para análisis y modelado |
| `sentiment` | Etiqueta de tres clases derivada de `stars` |

Conservar los originales es lo que permite que el sistema de recuperación muestre las reseñas tal como las escribió el cliente mientras busca sobre texto normalizado.

---

## Análisis exploratorio de datos

**Los cinco niveles de puntuación están perfectamente equilibrados por diseño**, 42.000 cada uno, con media exactamente 3,000. Esto es cómodo para el modelado y descalificante para cualquier afirmación de negocio: el dataset no puede decir cómo de común es la insatisfacción, porque su distribución fue construida. Las puntuaciones reales de Amazon se concentran fuertemente en cinco estrellas.

**La dimensión de producto no existe.** 156.458 productos para 208.899 reseñas, con mediana de una reseña cada uno y máximo de ocho. Solo 427 productos alcanzan cinco reseñas y ninguno llega a diez. El análisis obvio que construir con un dataset de reseñas, el ranking de mejores y peores productos, se habría calculado sobre medias de una observación. Lo mismo ocurre con los reseñadores: 187.140, con mediana de una reseña.

**Detectarlo y descartar ese análisis fue lo más útil que hizo la fase exploratoria.** Determinó además qué se le podría preguntar después al sistema de recuperación.

**Son textos cortos.**

| | `body_clean` | `title_clean` |
|---|---|---|
| Mediana de caracteres | 120 | 16 |
| Mediana de palabras | 22 | 3 |
| Percentil 99 de caracteres | 665 | 68 |
| Máximo de caracteres | 3.086 | 128 |

Los clientes escriben veredictos breves y directos en lugar de ensayos razonados. Hay poco espacio en 120 caracteres para el matiz, así que la señal que exista está concentrada en un puñado de palabras.

---

## Análisis estadístico

Siete preguntas, cada una con un contraste formal y un tamaño del efecto. **Con 200.000 observaciones todos los p-valores rechazan**, así que el tamaño del efecto es lo que lleva la información.

| Pregunta | Test | Resultado |
|---|---|---|
| ¿Cómo se reparten las reseñas entre categorías? | Descriptivo | 28.164 en `home` frente a 1.102 en `grocery`; rating medio de 2,76 a 3,40 |
| ¿Escriben más los clientes insatisfechos? | Kruskal-Wallis | Sí, poco: ε² = 0,017, y **no de forma monótona** |
| ¿Varía la longitud entre categorías? | Kruskal-Wallis | Apenas: ε² = 0,007 |
| ¿Están las clases repartidas por igual entre particiones? | Chi cuadrado | Perfectamente estratificadas: V = 0,0003, p = 1,000 |
| ¿Se asocia la categoría con la polaridad? | Chi cuadrado | Real pero débil: V = 0,063 |
| ¿Qué palabras distinguen cada nivel? | Frecuencias | `no` encabeza todos los grupos; neutro no tiene vocabulario propio |
| ¿Las reseñas en mayúsculas se concentran en ratings bajos? | Chi cuadrado | Sí, de forma despreciable: V = 0,012, sobre el 0,96 % del corpus |

**La longitud no es monótona respecto a la puntuación.** Las reseñas más largas no son las más airadas: el máximo está en dos estrellas (24 palabras de mediana), mientras que las de una estrella se quedan en 22. Un cliente completamente insatisfecho devuelve un veredicto corto y seco; uno decepcionado se explica. El post-hoc de Dunn separa todos los pares de niveles **salvo el uno y el tres** (p = 0,19), lo que significa que la longitud refleja cuánto se implicó el cliente, no si quedó satisfecho.

**El hallazgo que condicionó todo lo posterior:** neutro y negativo comparten la misma mediana de longitud (23 palabras); la proporción de neutras se mantiene cerca del 20 % en las treinta categorías; y las palabras más frecuentes de la clase neutra son idénticas a las de las reseñas de dos y tres estrellas. **Tres mediciones independientes, todas apuntando a la misma frontera, tres etapas antes de entrenar ningún modelo.**

---

## Metodología

1. **Carga y limpieza** — fusión con la partición preservada, filtrado a español, diagnóstico y eliminación de artefactos
2. **Análisis estadístico** — tests no paramétricos con tamaños del efecto sobre longitud, categoría y estilo de escritura
3. **Baseline** — TF-IDF con regresión logística, 144 configuraciones comparadas en validation
4. **Ajuste fino** — BETO con pérdida ponderada por clase en una GPU L4
5. **Comparación formal** — McNemar, intervalos de confianza por bootstrap, tiempo de inferencia y análisis de acuerdo
6. **Visualización de embeddings** — PCA y t-SNE sobre la representación `[CLS]`
7. **Sistema de recuperación** — ChromaDB con filtrado por metadatos y un generador local

**Protocolo, fijado antes de entrenar ningún modelo:** el F1 macro es la métrica principal, porque con clases en proporción 2:1:2 un modelo que nunca predijera neutro seguiría acertando el 80 % de las veces. Validation selecciona la configuración; el conjunto de test se mira una sola vez, al final.

---

## Baseline — TF-IDF y regresión logística

**Primero el suelo.** Un `DummyClassifier` que predice la clase mayoritaria alcanza accuracy 0,399 y **F1 macro 0,190**. La distancia entre esos dos números es la cuestión: un modelo puede parecer correcto en un 40 % siendo completamente inútil.

Se compararon 144 configuraciones, variando modelo, rango de n-gramas, `min_df`, `C`, `class_weight` y `sublinear_tf`.

| | Modelo | n-gramas | min_df | C | class_weight | F1 macro | Vocabulario |
|---|---|---|---|---|---|---|---|
| Mejor absoluto | LogisticRegression | (1,3) | 3 | 0,5 | balanced | **0,7247** | 466.829 |
| Mejor LinearSVC | LinearSVC | (1,3) | 5 | 0,1 | balanced | 0,7208 | 253.940 |
| Mejor compacto | LogisticRegression | (1,2) | 5 | 0,5 | balanced | 0,7213 | 136.349 |

**Solo un parámetro importa de verdad, y es `class_weight`.** La mejor configuración ponderada alcanza 0,7247 frente a 0,7096 sin ponderar, y las quince primeras posiciones usan `balanced`. Todo lo demás es una meseta: los quince mejores caben en 0,0050 de F1 macro.

**Los n-gramas ayudan con rendimientos decrecientes acusados.** Los unigramas llegan a 0,7069, los bigramas a 0,7213 y los trigramas a 0,7247. El salto a bigramas vale 0,014; añadir trigramas vale 0,003 y multiplica el vocabulario por tres.

**Resultados en test:**

| | Precisión | Recall | F1 |
|---|---|---|---|
| Negativo | 0,845 | 0,806 | 0,825 |
| Neutro | 0,457 | 0,565 | **0,505** |
| Positivo | 0,878 | 0,815 | 0,845 |
| **Macro** | **0,727** | **0,729** | **0,725** |

El F1 macro en validation es 0,7247 y en test 0,725, una diferencia de una diezmilésima. Elegir entre 144 configuraciones conlleva el riesgo de dar con una que encaje en esa partición concreta; estas dos cifras dicen que no ocurrió.

---

## Ajuste fino de BETO

`dccuchile/bert-base-spanish-wwm-cased`, entrenado en **23 minutos** en una GPU L4 (12.434 pasos a 8,9 pasos/s).

| Parámetro | Valor | Motivo |
|---|---|---|
| `max_length` | 192 | El percentil 99 de longitud en tokens es 162; un valor menor truncaría específicamente las reseñas largas, y esas se inclinan al lado negativo |
| Tasa de aprendizaje | 2e-5 | Rango estándar para ajuste fino de BERT |
| Tamaño de lote | 32 | Con 10 % de calentamiento, para que la cabeza de clasificación inicializada al azar no destroce los pesos preentrenados |
| Épocas | 2 | Mejor época seleccionada por F1 macro en validation |
| Pérdida | Entropía cruzada ponderada | El equivalente de `class_weight='balanced'`, el único parámetro que movía el baseline |

Título y cuerpo se pasan al tokenizador **como par**, de modo que inserta el separador y marca qué tokens pertenecen a cada campo, en lugar de concatenarse en una única cadena.

**Una desviación deliberada respecto al baseline:** allí se compararon 144 configuraciones porque un ajuste tarda 53 segundos. Aquí una sola ejecución son 23 minutos, así que una búsqueda equivalente llevaría días. Se aceptó la receta estándar y se dejó constancia del motivo.

**Resultados en test:**

| | Precisión | Recall | F1 |
|---|---|---|---|
| Negativo | 0,881 | 0,827 | 0,853 |
| Neutro | 0,507 | 0,633 | **0,563** |
| Positivo | 0,909 | 0,852 | 0,879 |
| **Macro** | **0,765** | **0,770** | **0,765** |

---

## Comparación de modelos

| | Baseline | BETO | Δ |
|---|---|---|---|
| F1 macro | 0,725 | **0,765** | +0,040 |
| Accuracy | 0,761 | **0,798** | +0,037 |
| F1 de neutro | 0,505 | **0,563** | +0,058 |
| Inferencia | **0,048 ms/reseña** | 48,05 ms/reseña | **1.009×** |
| Entrenamiento | 53 s, CPU local | 23 min, GPU de pago | |
| Tamaño del modelo | vectorizador + coeficientes | 440 MB | |
| Interpretable | **un peso por palabra** | no | |

**La diferencia es real.** McNemar devuelve chi cuadrado 50,40 con p = 1,3e-12: BETO recupera 416 reseñas que el baseline falla, frente a 234 en sentido contrario. El intervalo bootstrap de la diferencia en F1 macro, **[0,029, 0,051]**, excluye el cero con holgura.

**La mejora se concentra donde hacía falta.** Neutro gana 0,058 frente a 0,028 y 0,034 de los dos polos. El transformer aporta en el caso ambiguo, que es precisamente donde un modelo lineal no puede, y aporta menos donde el baseline ya era competente.

**Y cuesta tres órdenes de magnitud más por predicción**, medido con los dos modelos en la misma CPU. Clasificar un millón de reseñas son 48 segundos con el baseline y unas trece horas con BETO.

**Ninguno de los dos domina.**

| | Reseñas | Porcentaje |
|---|---|---|
| Aciertan los dos | 3.558 | 71,4 % |
| Solo BETO | 416 | 8,4 % |
| Solo el baseline | 234 | 4,7 % |
| Fallan los dos | 772 | 15,5 % |

El kappa de Cohen entre los dos conjuntos de predicciones es 0,782 — alto, pero lejos de la casi duplicación que se daría si un modelo fuera simplemente un refinamiento del otro. **Un oráculo que eligiera siempre el modelo acertado alcanzaría un 84,5 % de accuracy frente al 79,8 % de BETO**, y esos 4,7 puntos de margen son mayores que la diferencia entre los dos modelos.

**Dónde viven los errores.** La proyección de embeddings muestra que BETO aprendió **un único eje** de negativo a positivo, no tres conceptos separados, con neutro ocupando el centro. Los errores se concentran en una banda densa a lo largo de esa zona de transición y se adelgazan hacia los dos extremos. El modelo es seguro donde el sentimiento es inequívoco y se confunde donde el propio texto lo es.

---

## Sistema de recuperación (RAG)

Reseñas indexadas en ChromaDB con embeddings de frase, recuperadas por similitud semántica con filtrado por metadatos, y resumidas por un modelo de lenguaje restringido al texto recuperado.

**Decisiones de diseño:**

- **Una reseña es un fragmento.** Con una mediana de 22 palabras, la reseña ya es la unidad correcta. El corpus elimina un problema de ajuste entero.
- **La recuperación funciona sobre texto limpio; lo que se devuelve es el original.** Solo posible porque las columnas originales se conservaron durante la limpieza.
- **El sentimiento es un campo indexado y filtrable**, que es lo que conecta las dos mitades del proyecto. Sobre texto sin etiquetar la etiqueta vendría del clasificador — y el análisis de coste dice cuál: 10 segundos con el baseline frente a 2,8 horas con BETO.
- **Distancia coseno, no la de Chroma por defecto**, para que la magnitud del vector no deje que la longitud de la reseña interfiera en el orden.
- **La deduplicación no es opcional.** *Buena relación calidad precio* aparece 112 veces de forma literal; sin un filtro de diversidad, una consulta devuelve quince fragmentos que dicen todos lo mismo.
- **Tres reglas en el prompt:** responder solo con las reseñas proporcionadas, citar el número de cada reseña que respalda una afirmación, y decir explícitamente cuándo la respuesta no está.

**Lo que reveló la recuperación.** Preguntar *¿de qué se quejan los clientes?* en la categoría `wireless` devolvió quince reseñas de las que casi ninguna hablaba del producto: vendedores que no contestan, pedidos que nunca llegaron, devoluciones que tardaron semanas, garantías que nadie atendió. **En las reseñas negativas de esa categoría, la insatisfacción es mayoritariamente logística y no de producto** — lo que sitúa la palanca en la gestión del vendedor y la entrega, no en la fabricación.

Preguntar específicamente por la batería devolvió quince reseñas todas sobre el tema, agrupadas en autonomía insuficiente, fallos de carga y discrepancia con la especificación anunciada. **El índice funcionaba en ambos casos; la primera pregunta era simplemente demasiado genérica.**

---

## Limitaciones

- **La clase neutra no es fiable para decisiones automáticas.** Con una precisión de 0,507, la mitad de lo que el modelo etiqueta como neutro no lo es.
- **El corpus no admite afirmaciones absolutas de negocio.** Sus cinco niveles de puntuación fueron equilibrados por diseño, así que no dice nada sobre cómo de común es cada puntuación en la realidad. Las comparaciones entre categorías son válidas; los niveles absolutos no.
- **No es posible ningún análisis a nivel de producto.** Una mediana de una reseña por producto lo descarta, tanto para el análisis estadístico como para el sistema de recuperación.
- **El 15,5 % del conjunto de test está fuera del alcance de ambos modelos.** Una reseña de tres estrellas que dice *"está bien pero se rompió a los dos meses"* lleva señal de dos clases a la vez. Es el techo práctico del problema, no un defecto del enfoque.
- **Los hiperparámetros de BETO no se ajustaron**, solo se aceptaron de la receta estándar, porque cada ejecución cuesta 23 minutos.
- **El sistema de recuperación no se ha evaluado cuantitativamente.** No hay ninguna medición de si las reseñas recuperadas son las correctas.
- **El RAG recupera y resume evidencia; no cuenta.** *¿Qué porcentaje de clientes se queja del precio?* queda fuera de su alcance y corresponde al análisis estadístico.

---

## Conclusiones

**El corpus decide más que los modelos.** Tres hallazgos de las fases exploratorias condicionaron todo lo posterior: el equilibrio artificial de clases, que descalifica las afirmaciones absolutas de negocio; la ausencia de dimensión de producto, que canceló un análisis planificado entero; y el hecho de que ninguna característica medida separa neutro de negativo.

**Los modelos se comportaron como predijeron los datos.** Ambos clasificadores fallan exactamente donde la exploración dijo que lo harían. La longitud no separaba neutro de negativo, la categoría tampoco, el vocabulario tampoco — y el F1 de neutro es 0,505 en el baseline y 0,563 en BETO, frente al 0,83-0,88 de los dos polos.

**En lo que los modelos son buenos es en la distinción que importa comercialmente.** El signo prácticamente nunca se invierte, y cuando el modelo falla en una reseña polar se repliega al centro en lugar de afirmar lo contrario.

**La comparación es una decisión, no un ranking.** BETO gana por cuatro puntos y el resultado es estadísticamente sólido. También cuesta 1.009 veces más por predicción. Lo que convierte la pregunta de *cuál es mejor* en *cuánto valen aquí cuatro puntos*, y la respuesta depende del volumen, del presupuesto de latencia y de si la decisión tiene que ser auditable. Para la mayoría de usos, el baseline.

**El método es la parte reutilizable.** Nada se limpió antes de medirlo, que es la razón por la que la fase de eliminación de HTML nunca se escribió. Ningún test se reportó sin tamaño del efecto, porque con este tamaño muestral el p-valor no lleva información. Y el rigor se escaló al coste: 144 configuraciones donde un ajuste cuesta 53 segundos, una receta estándar donde cuesta 23 minutos.

---

## Posibles mejoras

**Baseline**

- Vectorizar título y cuerpo por separado con un `ColumnTransformer` en lugar de concatenarlos, para que una palabra en el título sea una característica distinta de la misma palabra en el cuerpo
- Probar `ComplementNB` y `SGDClassifier`, ninguno incluido en la búsqueda
- Reentrenar sobre train más validation una vez fijada la configuración

**BETO**

- **Ajustar los umbrales de decisión** en validation en lugar de tomar el `argmax`. La mejora más barata disponible, no requiere reentrenar, y va dirigida directamente a la clase neutra
- **Un modelo mayor.** `roberta-large-bne` no pudo cargarse por un problema de dependencias del tokenizador; `xlm-roberta-large` sí funciona pero necesita hora y media de entrenamiento para una ganancia esperada de uno o dos puntos
- **Búsqueda de hiperparámetros**, limitada por el coste de 23 minutos por ejecución

**La propia tarea**

- **Colapsar el problema a dos clases.** Llevaría el F1 macro a los noventa y pocos, pero como redefinición de la tarea y no como mejora del modelo: la ganancia viene de eliminar la parte difícil, no de resolverla

**Ambos modelos**

- **Un ensamblado.** El 8,4 % de las reseñas las recupera solo BETO y el 4,7 % solo el baseline; los 4,7 puntos de margen hasta el oráculo son la mayor oportunidad identificada en este trabajo
- **Cuantificar el techo de las etiquetas.** El corpus contiene textos de reseña idénticos con puntuaciones distintas; medir qué parte del 15,5 % residual es irreducible establecería cuánto del error restante merece la pena perseguir

**Recuperación**

- Un modelo de embeddings más potente, reescritura de consultas, recuperación híbrida combinando vectores con búsqueda por palabras clave, reordenación con cross-encoder, y un conjunto etiquetado de preguntas para evaluar la calidad de la recuperación

---

## Requisitos

```bash
pip install kagglehub pandas numpy matplotlib seaborn scikit-learn scikit-posthocs statsmodels nltk emoji
pip install torch transformers datasets accelerate
pip install chromadb sentence-transformers
```

El ajuste fino requiere GPU. Los entrenamientos se ejecutaron en Google Colab con una L4; todo lo demás corre en CPU.

---

*Fuente de datos: https://www.kaggle.com/datasets/mexwell/amazon-reviews-multi*
