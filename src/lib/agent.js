import { MLP, sparse, toDense } from './nn.js'
import { DIRS, isWall, bfs, allFree, distances } from './maze.js'

export const ACTIONS = ['N', 'E', 'S', 'O']
export const SENSOR_COUNT = 4

// Agente de Q-learning profundo (Double DQN con replay de experiencias).
// Entradas de la red:
//   [4 sensores de obstáculo (N,E,S,O), x normalizada, y normalizada, one-hot de la celda]
// Salidas: valor Q para cada una de las 4 acciones.
export class Agent {
  constructor(maze, start, goal, opts = {}) {
    this.maze = maze
    this.start = start
    this.goal = goal
    this.o = {
      hidden: [48, 24],
      lr: 0.0015,
      gamma: 0.99,
      batch: 32,
      bufferSize: 30000,
      epsStart: 1,
      epsEnd: 0.05,
      epsDecaySteps: 0,
      targetSync: 250,
      randomStarts: 0.5,
      ...opts,
    }
    // Currículo inverso: los episodios arrancan cerca de la salida y el radio
    // crece cuando la red aprende a salir desde esa distancia.
    this.dist = distances(maze, goal)
    this.startDist = this.dist[start[1] * maze.width + start[0]]
    this.maxDist = Math.max(...this.dist.filter((d) => d >= 0))
    this.radius = Math.min(4, this.maxDist)
    this.recent = []
    this.free = allFree(maze)
    this.index = new Int32Array(maze.width * maze.height).fill(-1)
    this.free.forEach(([x, y], i) => (this.index[y * maze.width + x] = i))
    if (!this.o.epsDecaySteps) this.o.epsDecaySteps = Math.max(3000, this.free.length * 15)
    this.inputSize = SENSOR_COUNT + 2 + this.free.length
    this.sizes = [this.inputSize, ...this.o.hidden, 4]
    this.net = new MLP(this.sizes, { lr: this.o.lr })
    this.target = new MLP(this.sizes)
    this.target.copyFrom(this.net)
    this.buffer = []
    this.bufPos = 0
    this.optimal = bfs(maze, start, goal)
    this.maxSteps = Math.max(100, this.free.length * 3)
    this.totalSteps = 0
    this.updates = 0
    this.episode = 0
    this.history = [] // { steps, reward, reached }
    this.greedy = null // { path, reached }
    this.lastLoss = 0
    this.solvedAt = null
    this.best = null // { path, episode, net } mejor ruta encontrada (checkpoint)
    this.cache = new Map()
    this.resetEpisode()
  }

  get epsilon() {
    const { epsStart, epsEnd, epsDecaySteps } = this.o
    const k = Math.min(1, this.totalSteps / epsDecaySteps)
    const floor = this.curriculumDone ? epsEnd : 0.15
    return Math.max(floor, epsStart + (epsEnd - epsStart) * k)
  }

  sensors(x, y) {
    return DIRS.map(([dx, dy]) => (isWall(this.maze, x + dx, y + dy) ? 1 : 0))
  }

  features(x, y) {
    const key = y * this.maze.width + x
    let f = this.cache.get(key)
    if (f) return f
    const idx = []
    const val = []
    this.sensors(x, y).forEach((v, i) => v && (idx.push(i), val.push(v)))
    idx.push(4, 5, SENSOR_COUNT + 2 + this.index[key])
    val.push(x / (this.maze.width - 1), y / (this.maze.height - 1), 1)
    f = sparse(idx, val)
    this.cache.set(key, f)
    return f
  }

  get curriculumDone() {
    return this.radius >= this.maxDist
  }

  resetEpisode() {
    const ready = this.radius >= this.startDist
    const useRandom = !ready || Math.random() < this.o.randomStarts
    let p = this.start
    if (useRandom) {
      const r = this.radius
      const pool = this.free.filter(([x, y]) => {
        const d = this.dist[y * this.maze.width + x]
        return d > 0 && d <= r
      })
      // Sesgo hacia la frontera del radio actual.
      const frontier = pool.filter(([x, y]) => this.dist[y * this.maze.width + x] > r * 0.6)
      const src = frontier.length && Math.random() < 0.6 ? frontier : pool
      p = src.length ? src[(Math.random() * src.length) | 0] : this.start
    }
    this.pos = [p[0], p[1]]
    this.epSteps = 0
    this.epReward = 0
    this.epFromStart = !useRandom
    this.trail = [[p[0], p[1]]]
  }

  argmax(q) {
    let b = 0
    for (let i = 1; i < q.length; i++) if (q[i] > q[b]) b = i
    return b
  }

  move(x, y, a) {
    const nx = x + DIRS[a][0]
    const ny = y + DIRS[a][1]
    if (isWall(this.maze, nx, ny)) return { x, y, reward: -0.3, hit: true, done: false }
    const done = nx === this.goal[0] && ny === this.goal[1]
    return { x: nx, y: ny, reward: done ? 1 : -0.02, hit: false, done }
  }

  // Un paso de interacción + un paso de entrenamiento.
  step() {
    const [x, y] = this.pos
    const s = this.features(x, y)
    const explore = Math.random() < this.epsilon
    const a = explore ? (Math.random() * 4) | 0 : this.argmax(this.net.predict(s))
    const r = this.move(x, y, a)
    this.store([x, y, a, r.reward, r.x, r.y, r.done])
    this.pos = [r.x, r.y]
    if (!r.hit) this.trail.push([r.x, r.y])
    this.epSteps++
    this.epReward += r.reward
    this.totalSteps++
    if (this.buffer.length >= this.o.batch * 4) this.learn()

    const limit = this.epFromStart ? this.maxSteps : Math.min(this.maxSteps, this.radius * 6 + 20)
    const timeout = this.epSteps >= limit
    const event = { from: [x, y], to: [r.x, r.y], action: a, explore, hit: r.hit, done: r.done, ended: false, improved: false }
    if (r.done || timeout) {
      if (!this.epFromStart) {
        this.recent.push(r.done && this.epSteps <= this.radius * 2 + 4 ? 1 : 0)
        if (this.recent.length > 12) this.recent.shift()
        const rate = this.recent.reduce((a, b) => a + b, 0) / this.recent.length
        if (this.recent.length >= 8 && rate >= 0.75 && this.radius < this.maxDist) {
          this.radius = Math.min(this.maxDist, Math.ceil(this.radius * 1.25 + 2))
          this.recent = []
        }
      }
      this.history.push({ steps: this.epSteps, reward: this.epReward, reached: r.done, fromStart: this.epFromStart })
      if (this.history.length > 400) this.history.shift()
      this.episode++
      this.greedy = this.greedyPath()
      if (this.greedy.reached && (!this.best || this.greedy.path.length < this.best.path.length)) {
        const net = this.best?.net || new MLP(this.sizes)
        net.copyFrom(this.net)
        this.best = { path: this.greedy.path, episode: this.episode, net }
        event.improved = true
      }
      if (this.greedy.reached && this.optimal && this.greedy.path.length === this.optimal.length && this.solvedAt == null)
        this.solvedAt = this.episode
      event.ended = true
      this.resetEpisode()
    }
    return event
  }

  store(t) {
    if (this.buffer.length < this.o.bufferSize) this.buffer.push(t)
    else this.buffer[this.bufPos] = t
    this.bufPos = (this.bufPos + 1) % this.o.bufferSize
  }

  learn() {
    const { batch, gamma } = this.o
    const xs = []
    const as = []
    const ys = []
    for (let k = 0; k < batch; k++) {
      const [x, y, a, r, nx, ny, done] = this.buffer[(Math.random() * this.buffer.length) | 0]
      let target = r
      if (!done) {
        const s2 = this.features(nx, ny)
        const a2 = this.argmax(this.net.predict(s2)) // Double DQN
        target += gamma * this.target.predict(s2)[a2]
      }
      xs.push(this.features(x, y))
      as.push(a)
      ys.push(target)
    }
    this.lastLoss = this.net.trainBatch(xs, as, ys)
    this.updates++
    if (this.updates % this.o.targetSync === 0) this.target.copyFrom(this.net)
  }

  // Política aprendida (sin exploración) desde el punto de partida.
  greedyPath(from = this.start, net = this.net) {
    let [x, y] = from
    const path = [[x, y]]
    const seen = new Set([y * this.maze.width + x])
    for (let i = 0; i < this.maxSteps; i++) {
      const a = this.argmax(net.predict(this.features(x, y)))
      const r = this.move(x, y, a)
      if (r.hit) return { path, reached: false }
      x = r.x
      y = r.y
      path.push([x, y])
      if (r.done) return { path, reached: true }
      const k = y * this.maze.width + x
      if (seen.has(k)) return { path, reached: false }
      seen.add(k)
    }
    return { path, reached: false }
  }

  // Valor máximo Q por celda (para el mapa de calor).
  valueMap(net = this.net) {
    const out = new Float32Array(this.maze.width * this.maze.height).fill(NaN)
    const best = new Int8Array(this.maze.width * this.maze.height).fill(-1)
    for (const [x, y] of this.free) {
      const q = net.predict(this.features(x, y))
      const b = this.argmax(q)
      out[y * this.maze.width + x] = q[b]
      best[y * this.maze.width + x] = b
    }
    return { values: out, best }
  }

  // Activaciones de todas las capas (entrada densa) para visualizar la red.
  activations(x = this.pos[0], y = this.pos[1], net = this.net) {
    const acts = net.forward(this.features(x, y))
    acts[0] = toDense(acts[0], this.inputSize)
    return acts
  }
}
