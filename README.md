# RNA · Laberinto

Reimplementación moderna de un sistema de detección de obstáculos con redes neuronales:
una esfera aprende a salir de un laberinto 3D generado desde un BMP (blanco = libre, negro = pared).

- **React + Vite**, **Three.js** (escena 3D) y **GSAP** (movimiento de la esfera, recorrido de la ruta, animación de la red).
- Red neuronal escrita a mano (`src/lib/nn.js`): perceptrón multicapa, tanh, Adam.
- Agente (`src/lib/agent.js`): Double DQN con replay de experiencias, red objetivo y currículo inverso.
  - Entradas: 4 sensores de obstáculo (N, E, S, O), posición (x, y) y one-hot de la celda.
  - Salidas: valor Q para cada dirección.
  - Recompensas: +1 salida, −0,02 por paso, −0,3 por choque.
- Laberintos (`src/lib/maze.js`): carga de BMP/PNG, generador aleatorio, exportación a BMP y BFS como referencia de la ruta óptima.

```bash
npm install
npm run dev
```

`public/laberinto-ejemplo.bmp` es un laberinto de ejemplo para probar la carga.
