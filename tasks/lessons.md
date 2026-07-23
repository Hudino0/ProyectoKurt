# Lecciones

Patrones a aplicar en el futuro, extraídos de los errores cometidos en este proyecto.

---

## L1 — Verificar el preprocesamiento contra una muestra publicada antes de modelar

**Qué pasó.** El artículo no describe cómo pasó del Excel del EEDB a su matriz de datos.
En vez de suponerlo, se usó la Table 6 del paper ("muestras seleccionadas al azar") como
test de aceptación: el motor `1AS001` fijó sin ambigüedad que las variables ambientales
son el punto medio de las columnas Min/Max y que TF→1, MTF→2.

**Regla.** Antes de entrenar nada, buscar en el artículo alguna fila de datos concreta y
reproducirla exactamente. Si el paper publica estadística descriptiva (media, desviación
típica, N), reproducirla también: es el único modo de saber que se está replicando el
mismo conjunto de datos y no uno parecido. Aquí eso permitió alcanzar coincidencia exacta
en las 8 estadísticas de la Table 5.

---

## L2 — Cuando el desplazamiento de una salta a la vista, reconstruirlo con aritmética

**Qué pasó.** La Table 4 del paper reportaba presión barométrica máxima de 1 019 394.5 kPa
y temperatura mínima de 144 K. La tentación era descartarlo como "errata" y seguir.

En vez de eso se reconstruyó el defecto: partiendo de la media y la desviación típica
publicadas y de N=565, la suma implícita y la suma de cuadrados implícita determinan
cuántos valores corruptos hay y de qué magnitud (1 celda de temperatura, ~4-5 de presión).
La predicción se verificó: media esperada 286.97 vs 286.9116 publicada, desviación típica
esperada 10.33 vs 10.30618 publicada.

**Regla.** Un estadístico publicado que parezca imposible es información, no ruido.
Media, desviación típica y N determinan la suma y la suma de cuadrados; con eso se puede
inferir cuántos valores anómalos hay y de qué tamaño, sin acceso a los datos del autor.

---

## L3 — Los hiperparámetros por defecto de la herramienta original son una pista verificable

**Qué pasó.** La Table 13 del paper reporta Box constraint = 1.2076 y Epsilon = 0.1208.
La razón exacta 10 entre ambos delata las heurísticas por defecto de `fitrsvm` de MATLAB:
`iqr(y)/1.349` y `iqr(y)/13.49`. Aplicadas al bloque de entrenamiento reconstruido dan
1.2092 y 0.1209 — coincidencia a la tercera cifra.

**Regla.** Cuando un paper reporta hiperparámetros sin explicarlos, comprobar si son los
valores por defecto de la herramienta que usó. Si al recalcularlos sobre los datos propios
salen los mismos números, es una confirmación independiente y fuerte de que el conjunto de
datos reconstruido es el correcto.

---

## L4 — En optimización con restricciones, derivar la condición de parada, no intuirla

**Qué pasó (bug).** En el SMO del SVR, la selección del par de trabajo usaba los
subgradientes intercambiados: se elegía el índice de subida con el subgradiente de bajada
y viceversa. No dio error ni resultados absurdos — simplemente el algoritmo paraba a las
85 iteraciones en vez de a las ~1700, y el modelo quedaba mal ajustado. El síntoma sólo
fue visible al comparar con las cifras del paper.

**Regla.** En un optimizador con restricciones, escribir explícitamente la derivada
direccional de cada movimiento admisible antes de codificar la selección del par. Y
desconfiar de una convergencia sospechosamente rápida: pocas iteraciones para un problema
de cientos de variables casi siempre significa criterio de parada mal planteado, no
eficiencia.

---

## L5 — Al trocear un fichero por índices, comprobar quién ya consumió qué

**Qué pasó (bug).** `_read_sheet` devolvía `rows[1:]` (ya había consumido la cabecera) y
`load_2019` volvía a saltar 4 filas, descartando la primera fila de datos real. N daba 562
en vez de 563 — una diferencia de una fila, invisible salvo comparándola con un conteo
independiente.

**Regla.** Cuando dos funciones recortan la misma estructura, dejar por escrito en el
comentario cuántas filas ha quitado cada una. Y validar el conteo final contra una
referencia externa: aquí, contar las filas directamente sobre el Excel reveló la
discrepancia de una unidad que un test de "¿corre sin errores?" nunca habría detectado.

---

## L6 — Un modelo desde cero puede converger a la solución degenerada

**Qué pasó.** El GPR con kernel exponencial llevaba σ_ruido → 0, es decir, a interpolar
exactamente los datos de entrenamiento: la verosimilitud marginal crece sin límite y la
matriz de covarianza queda casi singular. Técnicamente óptimo, numéricamente frágil.

**Regla.** Al implementar máxima verosimilitud desde cero, preguntarse siempre si el
objetivo es acotado. Si no lo es, poner una cota explícita al parámetro que se dispara y
documentar por qué — en vez de dejar que el resultado dependa de dónde pare el optimizador.

---

## L7 — Los datasets históricos siguen existiendo aunque la web oficial ya no los sirva

**Qué pasó.** El paper usa las ediciones 09/2019 y 07/2021 del EEDB, pero EASA sólo publica
la edición vigente. La API CDX del Internet Archive, consultada por patrón de URL sobre el
directorio de ficheros de EASA, devolvió todas las ediciones históricas archivadas; la hoja
*Record of Changes* de cada fichero confirmó las fechas exactas (Issue 26B = 2019-09-20,
Issue 28C = 2021-07-20).

**Regla.** Ante un dataset versionado del que sólo se publica la versión actual, consultar
`web.archive.org/cdx/search/cdx` filtrando por el directorio de descargas antes de dar la
replicación por imposible. Y verificar siempre la identidad de lo descargado con metadatos
internos del propio fichero, no con el nombre.
