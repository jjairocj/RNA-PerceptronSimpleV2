# RNA · Laberinto

Reimplementación moderna de un sistema de detección de obstáculos con redes neuronales: una esfera aprende, por
ensayo y error, a salir de un laberinto 3D generado a partir de un BMP (blanco = libre, negro = pared) y devuelve la
ruta más corta de salida.

- **React + Vite**, **Three.js** (escena 3D) y **GSAP** (movimiento de la esfera, recorrido de la ruta, animación de la red).
- Red neuronal escrita a mano, sin librerías de ML: [`src/lib/nn.js`](src/lib/nn.js).
- Agente de aprendizaje por refuerzo: [`src/lib/agent.js`](src/lib/agent.js).
- Laberintos (BMP, generador, BFS): [`src/lib/maze.js`](src/lib/maze.js).

```bash
npm install
npm run dev
```

`public/laberinto-ejemplo.bmp` es un laberinto de ejemplo para probar la carga.

---

## Índice

1. [Del perceptrón simple al Q-learning profundo](#1-del-perceptrón-simple-al-q-learning-profundo)
2. [El laberinto como problema de decisión (MDP)](#2-el-laberinto-como-problema-de-decisión-mdp)
3. [Representación del estado: entradas de la red](#3-representación-del-estado-entradas-de-la-red)
4. [La red neuronal: propagación hacia adelante](#4-la-red-neuronal-propagación-hacia-adelante)
5. [Q-learning: la ecuación de Bellman](#5-q-learning-la-ecuación-de-bellman)
6. [Deep Q-Network: función de pérdida y estabilización](#6-deep-q-network-función-de-pérdida-y-estabilización)
7. [Retropropagación](#7-retropropagación)
8. [Optimizador Adam (y Adam disperso)](#8-optimizador-adam-y-adam-disperso)
9. [Exploración: política ε-greedy](#9-exploración-política-ε-greedy)
10. [Currículo inverso](#10-currículo-inverso)
11. [Evaluación, checkpoint y criterio de parada](#11-evaluación-checkpoint-y-criterio-de-parada)
12. [¿Cómo "detecta" un camino equivocado?](#12-cómo-detecta-un-camino-equivocado)
13. [El laberinto: BMP, generador y BFS](#13-el-laberinto-bmp-generador-y-bfs)
14. [El algoritmo completo](#14-el-algoritmo-completo)
15. [Hiperparámetros](#15-hiperparámetros)
16. [Rendimiento y complejidad](#16-rendimiento-y-complejidad)
17. [Visualización](#17-visualización)
18. [Mapa del código](#18-mapa-del-código)

---

## 1. Del perceptrón simple al Q-learning profundo

Un perceptrón simple calcula

```math
y = f\left(\sum_i w_i x_i + b\right)
```

y separa el espacio de entradas con un hiperplano. Eso basta para reaccionar a obstáculos ("si hay muro al norte,
no vayas al norte"), pero **no basta para encontrar la salida**: decidir en un cruce depende de lo que hay muchos
pasos más adelante, y esa información no está en los sensores locales.

Esta versión separa el problema en dos partes:

| Pregunta | Quién la responde |
| --- | --- |
| ¿Qué tan buena es cada acción en esta celda? | Una red neuronal multicapa que aproxima la función $Q(s, a)$ |
| ¿Cómo se aprende esa función sin saber la respuesta? | Aprendizaje por refuerzo (Q-learning) con recompensas |

No hay ejemplos etiquetados del tipo "en esta celda la respuesta correcta es Este". La red se entrena con su propia
experiencia: se mueve, recibe recompensas o castigos, y corrige sus estimaciones.

---

## 2. El laberinto como problema de decisión (MDP)

El problema se formaliza como un **proceso de decisión de Markov** $(S, A, P, R, \gamma)$:

- **Estados** $S$: las celdas libres del laberinto.
- **Acciones** $A = \{N, E, S, O\}$: moverse una celda en cada dirección.
- **Transición** $P$: determinista. Si la celda destino es pared (o está fuera del mapa), la esfera **se queda donde
  está**.
- **Recompensa** $R$:

```math
r(s, a) =
\begin{cases}
+1 & \text{si } s' \text{ es la salida (fin del episodio)} \\
-0{,}3 & \text{si choca con una pared} \\
-0{,}02 & \text{en cualquier otro paso}
\end{cases}
```

- **Descuento** $\gamma = 0{,}99$.

El objetivo es encontrar una política $\pi(s)$ que maximice la recompensa acumulada descontada:

```math
G_t = \sum_{k=0}^{\infty} \gamma^k \, r_{t+k+1}
```

**Por qué estas recompensas producen la ruta más corta.** Si el camino óptimo desde una celda tiene $n$ pasos, el
retorno es

```math
G = -0{,}02 \sum_{k=0}^{n-2} \gamma^k + \gamma^{n-1} \cdot 1
```

que **decrece estrictamente con $n$**: cada paso de más suma un castigo y además descuenta el premio final. Por lo
tanto, la política que maximiza $G$ es la que llega en el menor número de pasos. Los choques ($-0{,}3$) son mucho
más caros que un paso, así que la red aprende muy temprano a no chocar: esa es la parte de **detección de
obstáculos**.

Implementación: `Agent.move()` en [`agent.js`](src/lib/agent.js).

---

## 3. Representación del estado: entradas de la red

Para la celda $(x, y)$ el vector de entrada $\mathbf{x} \in \mathbb{R}^{6 + |S|}$ es:

| Índice | Entrada | Valor |
| --- | --- | --- |
| 0–3 | Sensores de obstáculo N, E, S, O | 1 si la celda vecina es pared, 0 si está libre |
| 4 | Posición $x$ normalizada | $x / (W - 1)$ |
| 5 | Posición $y$ normalizada | $y / (H - 1)$ |
| 6 … 6+\|S\|−1 | Codificación *one-hot* de la celda | 1 en la posición de la celda actual, 0 en el resto |

- Los **sensores** permiten aprender reglas generales de evitación de obstáculos.
- La **posición** $(x, y)$ da una noción continua de "dónde estoy".
- El **one-hot** le da a cada celda su propia fila de pesos en la primera capa, que funciona como un *embedding*
  aprendido de la celda. Sin él, dos celdas con los mismos sensores y posiciones parecidas serían casi
  indistinguibles, y en un laberinto dos celdas vecinas pueden necesitar acciones opuestas.

Como el vector es casi todo ceros (como mucho $4 + 2 + 1 = 7$ valores distintos de cero), se guarda en formato
**disperso** `{ idx, val }` y se calcula una sola vez por celda (caché). Implementación: `Agent.features()`.

---

## 4. La red neuronal: propagación hacia adelante

Es un perceptrón multicapa con arquitectura por defecto

```math
(6 + |S|) \;\rightarrow\; 48 \;\rightarrow\; 24 \;\rightarrow\; 4
```

Para cada capa $l$:

```math
\mathbf{z}^{(l)} = \mathbf{a}^{(l-1)} W^{(l)} + \mathbf{b}^{(l)}, \qquad
\mathbf{a}^{(l)} =
\begin{cases}
\tanh\left(\mathbf{z}^{(l)}\right) & \text{capas ocultas} \\
\mathbf{z}^{(l)} & \text{capa de salida (lineal)}
\end{cases}
```

con $\mathbf{a}^{(0)} = \mathbf{x}$. La salida $\mathbf{a}^{(L)} = \big(Q(s,N), Q(s,E), Q(s,S), Q(s,O)\big)$ es el
valor estimado de cada acción.

- **tanh** en las capas ocultas: está acotada en $(-1, 1)$, lo que facilita la visualización y mantiene estables
  las activaciones.
- **Salida lineal**: los valores Q pueden ser negativos o positivos y no deben quedar acotados.

**Primera capa dispersa.** En vez de recorrer las $6 + |S|$ entradas, solo se recorren las activas:

```math
z^{(1)}_j = b^{(1)}_j + \sum_{i \,:\, x_i \neq 0} x_i \, W^{(1)}_{ij}
```

**Inicialización.** $W_{ij} \sim \mathcal{N}(0, 1) \cdot \sqrt{1/n}$, con $n$ igual al número de entradas de la
capa. En la primera capa se usa $n = 8$ (el número aproximado de entradas *activas*) en lugar del tamaño total,
porque de lo contrario los pesos de una entrada one-hot serían diminutos. Los sesgos empiezan en 0.

Los pesos se guardan en `Float32Array` contiguos con el índice `W[l][i * nOut + j]`.

Implementación: `MLP.forward()` en [`nn.js`](src/lib/nn.js).

---

## 5. Q-learning: la ecuación de Bellman

La función de valor-acción óptima $Q^*(s, a)$ es el retorno esperado al tomar $a$ en $s$ y actuar de forma óptima
después. Cumple la **ecuación de optimalidad de Bellman**:

```math
Q^*(s, a) = r(s, a) + \gamma \max_{a'} Q^*(s', a')
```

(en un estado terminal, $Q^*(s, a) = r$). Q-learning convierte esa igualdad en una regla de aprendizaje: si la
estimación actual no la cumple, se corrige en la dirección del **objetivo TD** (*temporal difference*):

```math
y = r + \gamma \max_{a'} Q(s', a')
```

**Intuición: el valor fluye desde la salida hacia atrás.**

1. La celda pegada a la salida aprende rápido que su mejor acción vale $\approx +1$.
2. La celda anterior aprende $\approx -0{,}02 + 0{,}99 \cdot 1$.
3. Y así sucesivamente: cada celda hereda el valor de su mejor vecina, un poco descontado y castigado.

El resultado es un campo de valores que crece hacia la salida. Elegir en cada celda la acción de mayor Q equivale a
"subir la pendiente" de ese campo, siguiendo el camino más corto. Eso es lo que se ve en el **Mapa de valores Q**.

---

## 6. Deep Q-Network: función de pérdida y estabilización

Al aproximar $Q$ con una red $Q_\theta$, el aprendizaje se convierte en minimizar el error entre la predicción y el
objetivo TD. Con una red neuronal, Q-learning "ingenuo" es inestable; se usan cuatro técnicas estándar.

### 6.1 Replay de experiencias

Cada paso genera una transición $(s, a, r, s', \text{done})$ que se guarda en un **buffer circular** de 30 000
elementos. Para entrenar se toma un **lote aleatorio de 32** transiciones del buffer. Esto:

- rompe la correlación entre pasos consecutivos (que son casi idénticos), y
- reutiliza cada experiencia muchas veces.

El entrenamiento empieza cuando hay al menos $4 \times 32 = 128$ transiciones. Implementación: `Agent.store()`,
`Agent.learn()`.

### 6.2 Red objetivo

El objetivo $y$ se calcula con una **copia congelada** de la red, $Q_{\theta^-}$, que se sincroniza con la red
principal cada 250 actualizaciones:

```math
\theta^- \leftarrow \theta \quad \text{cada 250 actualizaciones}
```

Sin ella, la red estaría persiguiendo un objetivo que se mueve con cada actualización.

### 6.3 Double DQN

El $\max$ del objetivo clásico tiende a **sobreestimar** los valores (el máximo de estimaciones con ruido está
sesgado hacia arriba). Double DQN separa la **elección** de la acción de su **evaluación**:

```math
a^* = \arg\max_{a'} Q_\theta(s', a'), \qquad
y = r + \gamma \, Q_{\theta^-}(s', a^*) \cdot (1 - \text{done})
```

La red principal elige y la red objetivo evalúa.

### 6.4 Pérdida de Huber

Solo se entrena la salida de la acción que realmente se tomó. Con el error $\delta = Q_\theta(s, a) - y$:

```math
\mathcal{L}_\delta =
\begin{cases}
\tfrac{1}{2}\delta^2 & |\delta| < 1 \\
|\delta| - \tfrac{1}{2} & |\delta| \geq 1
\end{cases}
\qquad
\frac{\partial \mathcal{L}}{\partial Q_\theta(s,a)} = \operatorname{clip}(\delta, -1, 1)
```

Es cuadrática cerca de cero (precisa) y lineal para errores grandes (el gradiente nunca pasa de 1, lo que evita
saltos bruscos en los pesos). La pérdida del lote es el promedio sobre las 32 muestras.

Implementación: `Agent.learn()` y `MLP.trainBatch()`.

---

## 7. Retropropagación

Para cada muestra, el error se inyecta **solo** en la salida de la acción tomada:

```math
\boldsymbol{\delta}^{(L)}_j =
\begin{cases}
\operatorname{clip}(\delta, -1, 1) & j = a \\
0 & j \neq a
\end{cases}
```

y se propaga hacia atrás capa por capa:

```math
\frac{\partial \mathcal{L}}{\partial W^{(l)}_{ij}} = a^{(l-1)}_i \, \delta^{(l)}_j, \qquad
\frac{\partial \mathcal{L}}{\partial b^{(l)}_j} = \delta^{(l)}_j
```

```math
\delta^{(l-1)}_i = \left(\sum_j W^{(l)}_{ij} \, \delta^{(l)}_j\right) \left(1 - \left(a^{(l-1)}_i\right)^2\right)
```

donde $1 - \tanh^2(z)$ es la derivada de tanh, expresada con la propia activación. Los gradientes de las 32
muestras se acumulan y se escalan por $1/32$ antes de actualizar.

En la **primera capa** solo se calculan gradientes para las filas de entradas activas, porque
$\partial \mathcal{L} / \partial W^{(1)}_{ij} = x_i \, \delta^{(1)}_j = 0$ cuando $x_i = 0$.

Implementación: `MLP.trainBatch()`.

---

## 8. Optimizador Adam (y Adam disperso)

Cada parámetro $\theta$ con gradiente $g_t$ se actualiza con Adam ($\beta_1 = 0{,}9$, $\beta_2 = 0{,}999$,
$\epsilon = 10^{-8}$, $\eta = 0{,}0015$):

```math
m_t = \beta_1 m_{t-1} + (1-\beta_1)\, g_t \qquad
v_t = \beta_2 v_{t-1} + (1-\beta_2)\, g_t^2
```

```math
\hat m_t = \frac{m_t}{1-\beta_1^t} \qquad
\hat v_t = \frac{v_t}{1-\beta_2^t} \qquad
\theta_t = \theta_{t-1} - \eta \, \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon}
```

- $m$ (momento) suaviza la dirección del gradiente.
- $v$ adapta el tamaño del paso para cada peso por separado.
- La corrección $1 - \beta^t$ compensa que $m$ y $v$ empiezan en cero.

**Adam disperso ("lazy Adam").** En la primera capa solo se actualizan las filas de las entradas que aparecieron en
el lote (como mucho unas $32 + 6$ filas en vez de $|S| + 6$). Las capas densas se actualizan completas. Es la misma
aproximación que `LazyAdam` de TensorFlow, y es la que hace viable el one-hot en laberintos grandes.

Implementación: `MLP.adam()`.

---

## 9. Exploración: política ε-greedy

En cada paso:

```math
a =
\begin{cases}
\text{acción aleatoria} & \text{con probabilidad } \varepsilon \\
\arg\max_a Q_\theta(s, a) & \text{con probabilidad } 1 - \varepsilon
\end{cases}
```

$\varepsilon$ decae linealmente de 1 a 0,05 en $T = \max(3000,\; 15 \cdot |S|)$ pasos:

```math
\varepsilon_t = \max\!\left(\varepsilon_{\min},\; 1 + (0{,}05 - 1)\,\min\!\left(1, \tfrac{t}{T}\right)\right),
\qquad
\varepsilon_{\min} =
\begin{cases}
0{,}15 & \text{mientras el currículo no termina} \\
0{,}05 & \text{después}
\end{cases}
```

El piso de 0,15 mantiene la exploración mientras sigan apareciendo zonas nuevas del laberinto. En la visualización,
la acción elegida se marca en **ámbar** si fue exploración y en **verde** si la decidió la red.

Implementación: getter `Agent.epsilon`.

---

## 10. Currículo inverso

**El problema.** En un laberinto de 400 celdas, una exploración aleatoria casi nunca llega a una salida concreta.
Sin llegar nunca, la red solo ve recompensas de $-0{,}02$ y $-0{,}3$, y no hay ninguna señal que le diga dónde está
la salida.

**La solución.** Los episodios empiezan **cerca de la salida** y el punto de partida se aleja a medida que la red
aprende:

1. Se calcula con BFS la distancia $d(c)$ en pasos desde cada celda $c$ hasta la salida.
2. Se mantiene un radio $\rho$, que empieza en $\min(4, d_{\max})$.
3. Mientras $\rho < d(\text{inicio})$, **todos** los episodios empiezan en una celda con $0 < d(c) \leq \rho$.
   Después, el 50 % empieza desde el inicio elegido por el usuario y el otro 50 % desde una celda del radio.
4. Al elegir la celda, con probabilidad 0,6 se toma de la **frontera** ($d(c) > 0{,}6\,\rho$), donde la red todavía
   tiene que aprender.
5. Un episodio aleatorio cuenta como **éxito** si llega a la salida en $\leq 2\rho + 4$ pasos. Con al menos 8
   episodios registrados (ventana de los últimos 12) y una tasa de éxito $\geq 75\,\%$, el radio crece:

```math
\rho \leftarrow \min\!\left(d_{\max},\; \lceil 1{,}25\,\rho + 2 \rceil\right)
```

6. Los episodios aleatorios tienen un límite de $\min(3|S|,\; 6\rho + 20)$ pasos; los que empiezan en el inicio,
   $\max(100,\; 3|S|)$.

Es el mismo principio que la propagación del valor de la sección 5: primero se aprende bien el final del camino y
luego se extiende hacia atrás. Por eso, durante el entrenamiento, la esfera a veces **aparece** en un punto más
adelante: es el inicio de un episodio nuevo.

> **Nota honesta:** el currículo usa las distancias reales del laberinto (BFS) para elegir *dónde empieza* cada
> episodio. La red no ve esas distancias ni sabe nada de ellas; solo recibe sensores, posición y recompensas. Aun
> así, es una ayuda al entrenamiento que un agente completamente "a ciegas" no tendría.

Implementación: constructor, `resetEpisode()` y el final de `step()` en [`agent.js`](src/lib/agent.js).

---

## 11. Evaluación, checkpoint y criterio de parada

Al final de cada episodio se evalúa la **política greedy** (sin exploración) desde el inicio del usuario:

```math
s_{t+1} = \operatorname{mover}\!\left(s_t,\; \arg\max_a Q_\theta(s_t, a)\right)
```

La ruta se da por **fallida** si choca con una pared, si vuelve a una celda ya visitada (un ciclo: la red se
quedaría dando vueltas para siempre) o si supera $\max(100, 3|S|)$ pasos.

**Checkpoint de la mejor red.** El DQN es inestable: una red que encontró la ruta puede "olvidarla" unas cuantas
actualizaciones después. Por eso, cada vez que la ruta greedy llega a la salida con **menos pasos** que la mejor
anterior, se copian los pesos a `agent.best.net`. El botón *Recorrer ruta aprendida* siempre usa esa red.

**Criterio de parada.** Si la ruta greedy tiene la misma longitud que la de BFS, se marca como resuelto
(`solvedAt`) y, con la opción *Parar al hallar la ruta óptima*, el entrenamiento se detiene y la esfera recorre la
ruta. BFS solo sirve de **referencia** para comparar y decidir cuándo parar; no participa en las decisiones de la
red.

Implementación: `Agent.greedyPath()` y el final de `Agent.step()`.

---

## 12. ¿Cómo "detecta" un camino equivocado?

No hay ninguna regla del tipo "si es un callejón, vuelve". Un camino equivocado es simplemente un camino cuyo
**valor Q es menor**:

- Al fondo de un callejón, la ruta más corta a la salida es larga (hay que deshacer lo andado), así que su retorno
  $G$ es menor (sección 2) y su valor $Q$ también.
- En un cruce, la red compara los 4 valores Q. La dirección hacia el callejón lleva a celdas de menor valor, así
  que pierde frente a la dirección correcta.
- Las paredes se evitan porque chocar cuesta $-0{,}3$ y los sensores dicen dónde están.

Se puede ver con velocidad *Lento* y el botón *Paso*: en los cruces cambian los 4 valores Q del panel de la red, y
en el minimapa las flechas apuntan hacia la mejor dirección de cada celda.

---

## 13. El laberinto: BMP, generador y BFS

### Lectura de BMP

El navegador decodifica la imagen (BMP, PNG, etc.) y se dibuja en un `canvas` sin suavizado. Si el lado mayor pasa
de 61 píxeles, se reescala. Cada píxel se clasifica por su luminancia:

```math
Y = 0{,}299\,R + 0{,}587\,G + 0{,}114\,B, \qquad \text{celda} =
\begin{cases} \text{pared} & Y < 128 \\ \text{libre} & Y \geq 128 \end{cases}
```

### Exportación a BMP

Se escribe a mano un BMP de 24 bits: cabecera de 54 bytes (`BM`, tamaño, `BITMAPINFOHEADER`), filas guardadas **de
abajo hacia arriba** y relleno de cada fila a un múltiplo de 4 bytes.

### Generador aleatorio

1. **Backtracker recursivo** (DFS con pila) sobre una rejilla de $n \times n$ celdas lógicas, en una imagen de
   $(2n+1) \times (2n+1)$ píxeles. Produce un laberinto *perfecto*: exactamente un camino entre dos celdas
   cualesquiera.
2. Se abre aproximadamente el **6 %** de las paredes internas que separan dos pasillos en línea recta. Así aparecen
   **ciclos** y varias rutas posibles, y "la ruta más corta" deja de ser trivial.
3. Se abre una salida en el borde derecho, abajo.

### Inicio, salida y BFS

- Inicio por defecto: la celda libre más cercana a la esquina superior izquierda.
- Salida por defecto: la celda libre **del borde** más lejana (en pasos) desde el inicio; si no hay ninguna en el
  borde, la más lejana de todas.
- **BFS** (búsqueda en anchura) da la ruta óptima de referencia en $O(|S|)$ y también las distancias $d(c)$ que usa
  el currículo.

Implementación: [`maze.js`](src/lib/maze.js).

---

## 14. El algoritmo completo

```text
Entrada: laberinto, inicio, salida
Inicializar red Q_θ y red objetivo Q_θ⁻ ← Q_θ
Inicializar buffer D (capacidad 30 000), radio ρ ← min(4, d_max)

repetir por cada episodio:
    s ← inicio  o  celda aleatoria con 0 < d(c) ≤ ρ          (currículo, §10)
    repetir hasta llegar a la salida o agotar el límite de pasos:
        a ← aleatoria con prob. ε, si no argmax_a Q_θ(s, a)    (§9)
        ejecutar a → recompensa r, estado s', done              (§2)
        guardar (s, a, r, s', done) en D
        si |D| ≥ 128:
            tomar lote aleatorio B de 32 transiciones de D       (§6.1)
            para cada (s, a, r, s', done) en B:
                a* ← argmax_a' Q_θ(s', a')                     (Double DQN, §6.3)
                y  ← r + γ · Q_θ⁻(s', a*) · (1 − done)
            minimizar Huber(Q_θ(s, a) − y) con retropropagación + Adam   (§6.4, §7, §8)
            cada 250 actualizaciones: θ⁻ ← θ                    (§6.2)
        s ← s'
    actualizar el currículo (ρ)                                  (§10)
    evaluar la ruta greedy desde el inicio                       (§11)
    si es la más corta hasta ahora: guardar checkpoint
    si su longitud = longitud BFS: resuelto
```

En la interfaz, cada "paso" de la esfera ejecuta exactamente una iteración del bucle interno: un movimiento y una
actualización de la red.

---

## 15. Hiperparámetros

| Parámetro | Valor | Dónde |
| --- | --- | --- |
| Capas ocultas | 48 · 24 (configurable: 16, 32·16, 48·24, 64·32) | `Agent` opts `hidden` |
| Activación | tanh (ocultas), lineal (salida) | `MLP.forward` |
| Tasa de aprendizaje $\eta$ | 0,0015 (configurable) | `lr` |
| Descuento $\gamma$ | 0,99 (configurable) | `gamma` |
| Tamaño de lote | 32 | `batch` |
| Buffer de replay | 30 000 | `bufferSize` |
| Sincronización de la red objetivo | cada 250 actualizaciones | `targetSync` |
| $\varepsilon$ | 1 → 0,05 en $\max(3000, 15\lvert S\rvert)$ pasos; piso 0,15 durante el currículo | `epsilon` |
| Inicios aleatorios (tras cubrir el inicio) | 50 % | `randomStarts` |
| Recompensas | +1 salida, −0,02 paso, −0,3 choque | `move` |
| Adam | $\beta_1 = 0{,}9$, $\beta_2 = 0{,}999$, $\epsilon = 10^{-8}$ | `MLP.adam` |

Los valores salieron de pruebas sobre laberintos de 13×13 a 29×29. Con $\eta = 0{,}003$ la red llegaba antes a la
ruta, pero después la "olvidaba" con más frecuencia; con $\eta = 0{,}001$ era más estable pero más lenta.
$\gamma = 0{,}99$ mantiene visible el premio final incluso en rutas de más de 100 pasos ($0{,}99^{100} \approx 0{,}37$).

---

## 16. Rendimiento y complejidad

Cada paso del agente hace, por cada una de las 32 muestras del lote, tres pasadas hacia adelante (red principal en
$s$ y en $s'$, red objetivo en $s'$) y una hacia atrás. Con la entrada dispersa, el costo por pasada **no depende
del tamaño del laberinto**:

```math
O\big(k \cdot h_1 + h_1 h_2 + h_2 \cdot 4\big), \qquad k \leq 7 \text{ entradas activas}
```

| Laberinto | Celdas libres | Pasos hasta la ruta óptima (aprox.) | Tiempo en modo Turbo |
| --- | --- | --- | --- |
| 13×13 | ~80 | 5 000 – 18 000 | 2 – 8 s |
| 21×21 | ~210 | 15 000 – 47 000 | 6 – 20 s |
| 29×29 | ~400 | 40 000 – 170 000 | 20 – 70 s |

(≈ 400 µs por paso en un portátil moderno; los números varían con cada laberinto aleatorio.)

En el navegador el entrenamiento corre en el hilo principal con un presupuesto de ~13 ms por cuadro, y las
visualizaciones se refrescan con límites de frecuencia (red cada 60 ms, mapa de calor cada 350 ms, estadísticas
cada 150 ms) para no frenar el entrenamiento.

---

## 17. Visualización

- **Escena 3D** (`Maze3D.jsx`): paredes y baldosas con `InstancedMesh` (un solo *draw call* para miles de cubos).
  Las baldosas se colorean con $\max_a Q(s, a)$ normalizado. La esfera se mueve con `gsap.quickTo` y rueda según
  su desplazamiento ($\Delta\theta = \lVert\Delta\mathbf{p}\rVert / r$). La ruta final se dibuja con un
  `TubeGeometry` que crece mediante `setDrawRange` en una `gsap.timeline`.
- **Red** (`NetworkViz.jsx`): el grosor y la opacidad de cada conexión son la **señal que fluye**,
  $\lvert w_{ij} \, a_i \rvert$; cian si es positiva y coral si es negativa. Las activaciones se interpolan con
  GSAP (`endArray`). El nodo "celda" representa la entrada one-hot activa y sus conexiones son su fila de pesos.
- **Mapa de valores Q** (`MiniMap.jsx`): mapa de calor, flechas de $\arg\max_a Q$, ruta de la red (verde) y ruta
  BFS (punteada).
- **Gráfica** (`EpisodeChart.jsx`): pasos por episodio (verde intenso = desde el inicio, verde tenue = inicio
  aleatorio, coral = tiempo agotado) y longitud de la ruta greedy (ámbar) frente al óptimo.

---

## 18. Mapa del código

```text
src/
├── lib/
│   ├── nn.js          MLP: forward disperso, retropropagación, Huber, Adam disperso
│   ├── agent.js       Double DQN, replay, red objetivo, ε-greedy, currículo, checkpoint
│   ├── maze.js        BMP ↔ rejilla, generador, BFS, distancias
│   └── colors.js      escalas de color
├── components/
│   ├── Maze3D.jsx     escena Three.js + animaciones GSAP
│   ├── NetworkViz.jsx dibujo de la red
│   ├── MiniMap.jsx    editor 2D + mapa de valores Q
│   └── EpisodeChart.jsx
└── App.jsx            bucle de entrenamiento (requestAnimationFrame) e interfaz
```
