import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import gsap from 'gsap'
import Maze3D from './components/Maze3D.jsx'
import NetworkViz from './components/NetworkViz.jsx'
import MiniMap from './components/MiniMap.jsx'
import EpisodeChart from './components/EpisodeChart.jsx'
import { Agent } from './lib/agent.js'
import { generateMaze, mazeFromFile, mazeToBMP, isWall, distances, allFree, bfs } from './lib/maze.js'
import './App.css'

const SPEEDS = [
  { label: 'Lento', sps: 3 },
  { label: '1×', sps: 12 },
  { label: '5×', sps: 60 },
  { label: '25×', sps: 300 },
  { label: 'Turbo', sps: Infinity },
]
const ARCHS = { '16': [16], '32·16': [32, 16], '48·24': [48, 24], '64·32': [64, 32] }

// Inicio: primera celda libre (arriba-izquierda). Salida: celda libre del borde
// más lejana (en pasos) desde el inicio; si no hay en el borde, la más lejana.
function defaultEndpoints(m) {
  const free = allFree(m)
  if (free.length < 2) return { start: free[0] || null, goal: null }
  const start = free.reduce((a, b) => (b[0] + b[1] < a[0] + a[1] ? b : a))
  const d = distances(m, start)
  const onBorder = ([x, y]) => x === 0 || y === 0 || x === m.width - 1 || y === m.height - 1
  const reach = free.filter(([x, y]) => d[y * m.width + x] > 0)
  if (!reach.length) return { start, goal: null }
  const border = reach.filter(onBorder)
  const pool = border.length ? border : reach
  const goal = pool.reduce((a, b) => (d[b[1] * m.width + b[0]] > d[a[1] * m.width + a[0]] ? b : a))
  return { start, goal }
}

function initial() {
  const maze = generateMaze(10, 10)
  return { maze, ...defaultEndpoints(maze) }
}

export default function App() {
  const [{ maze, start, goal }, setWorld] = useState(initial)
  const [running, setRunning] = useState(false)
  const [speed, setSpeed] = useState(2)
  const [mode, setMode] = useState('start')
  const [size, setSize] = useState(10)
  const [arch, setArch] = useState('48·24')
  const [lr, setLr] = useState(0.0015)
  const [gamma, setGamma] = useState(0.99)
  const [autoStop, setAutoStop] = useState(true)
  const [stats, setStats] = useState(null)
  const [playing, setPlaying] = useState(false)
  const [toast, setToast] = useState(null)

  const sceneRef = useRef(null)
  const netRef = useRef(null)
  const mapRef = useRef(null)
  const chartRef = useRef(null)
  const agentRef = useRef(null)
  const loop = useRef({ running: false, speed: 2, autoStop: true, acc: 0, last: 0, t: {} }).current
  loop.speed = speed
  loop.autoStop = autoStop

  const optimal = useMemo(() => (start && goal ? bfs(maze, start, goal) : null), [maze, start, goal])

  const flash = useCallback((msg) => {
    setToast({ msg, id: Math.random() })
  }, [])

  // Crear un agente nuevo (red sin entrenar) cuando cambia el mundo o la arquitectura.
  useEffect(() => {
    loop.running = false
    setRunning(false)
    sceneRef.current?.stopRoute()
    mapRef.current?.reset()
    if (!start || !goal || !optimal) {
      agentRef.current = null
      setStats(null)
      return
    }
    const agent = new Agent(maze, start, goal, { hidden: ARCHS[arch], lr, gamma })
    agentRef.current = agent
    sceneRef.current?.showRoute(null)
    sceneRef.current?.setTrail([])
    sceneRef.current?.setHeat(new Float32Array(maze.width * maze.height).fill(0), 0, 1)
    refresh(true)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [maze, start, goal, optimal, arch, lr, gamma])

  // Actualiza visualizaciones (con límites de frecuencia para no saturar).
  const refresh = useCallback(
    (force = false, ev = null) => {
      const agent = agentRef.current
      if (!agent) return
      const now = performance.now()
      const due = (k, ms) => {
        if (force || now - (loop.t[k] || 0) >= ms) {
          loop.t[k] = now
          return true
        }
        return false
      }
      const slow = SPEEDS[loop.speed].sps <= 60
      if (due('net', slow ? 0 : 60)) {
        netRef.current?.update(agent.net, agent.activations(), {
          action: ev ? ev.action : -1,
          explore: ev?.explore,
          duration: slow ? Math.min(0.35, 1 / SPEEDS[loop.speed].sps) : 0.08,
        })
      }
      if (due('trail', slow ? 0 : 80)) {
        sceneRef.current?.setTrail(agent.trail.slice(-400))
        mapRef.current?.setAgent(agent.pos)
      }
      if (due('heat', 350)) {
        const h = agent.valueMap()
        let lo = Infinity
        let hi = -Infinity
        for (const v of h.values)
          if (!Number.isNaN(v)) {
            lo = Math.min(lo, v)
            hi = Math.max(hi, v)
          }
        sceneRef.current?.setHeat(h.values, lo, hi)
        mapRef.current?.setHeat(h)
      }
      if (due('stats', 150)) {
        setStats({
          episode: agent.episode,
          steps: agent.totalSteps,
          eps: agent.epsilon,
          radius: agent.radius,
          maxDist: agent.maxDist,
          loss: agent.lastLoss,
          best: agent.best?.path.length ?? null,
          bestEp: agent.best?.episode ?? null,
          optimal: agent.optimal.length,
          solvedAt: agent.solvedAt,
          free: agent.free.length,
          inputs: agent.inputSize,
          params: agent.net.W.reduce((a, w) => a + w.length, 0) + agent.net.b.reduce((a, b) => a + b.length, 0),
        })
        chartRef.current?.update(agent.history, loop.greedyHist || [], agent.optimal.length)
      }
    },
    [loop],
  )

  // Bucle de entrenamiento (requestAnimationFrame).
  useEffect(() => {
    let raf
    const tick = (now) => {
      raf = requestAnimationFrame(tick)
      const agent = agentRef.current
      const dt = Math.min(100, now - (loop.last || now))
      loop.last = now
      if (!loop.running || !agent) return
      const { sps } = SPEEDS[loop.speed]
      let n
      if (sps === Infinity) n = Infinity
      else {
        loop.acc += (dt * sps) / 1000
        n = Math.floor(loop.acc)
        loop.acc -= n
      }
      const t0 = performance.now()
      let ev = null
      let ended = false
      for (let i = 0; i < n; i++) {
        ev = agent.step()
        if (ev.ended) {
          ended = true
          loop.greedyHist = loop.greedyHist || []
          loop.greedyHist.push(agent.greedy.reached ? agent.greedy.path.length : null)
          if (loop.greedyHist.length > 400) loop.greedyHist.shift()
          if (ev.improved) mapRef.current?.setBest(agent.best.path)
          if (agent.solvedAt != null && loop.autoStop) break
          if (sps !== Infinity) break // en modo visual, pausa breve entre episodios
        }
        if (performance.now() - t0 > 13) break
      }
      if (!ev) return
      const scene = sceneRef.current
      if (sps <= 300) scene.moveBall(agent.pos[0], agent.pos[1], ended ? 0 : Math.min(0.3, 1.2 / sps), ev.hit && sps <= 12)
      else scene.moveBall(agent.pos[0], agent.pos[1], 0.06)
      if (ev.improved) {
        scene.showRoute(agent.best.path)
      }
      refresh(false, ev)
      if (agent.solvedAt != null && loop.autoStop) {
        loop.running = false
        setRunning(false)
        refresh(true, ev)
        flash(`¡Ruta óptima aprendida en el episodio ${agent.solvedAt}! (${agent.optimal.length - 1} pasos)`)
        setTimeout(() => playRoute(), 400)
      }
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [loop, refresh])

  const toggleRun = () => {
    if (!agentRef.current) return
    sceneRef.current.stopRoute()
    setPlaying(false)
    if (agentRef.current.solvedAt != null) loop.autoStop = false
    if (agentRef.current.solvedAt != null) setAutoStop(false)
    loop.running = !loop.running
    setRunning(loop.running)
  }

  const stepOnce = () => {
    const agent = agentRef.current
    if (!agent) return
    loop.running = false
    setRunning(false)
    const ev = agent.step()
    sceneRef.current.moveBall(agent.pos[0], agent.pos[1], ev.ended ? 0 : 0.3, ev.hit)
    refresh(true, ev)
  }

  const resetNet = () => {
    // Fuerza recrear el agente
    setWorld((w) => ({ ...w, maze: { ...w.maze } }))
  }

  // Recorre la mejor ruta aprendida con una línea de tiempo GSAP y muestra la red decidiendo.
  const playRoute = () => {
    const agent = agentRef.current
    if (!agent?.best) return flash('La red todavía no ha encontrado la salida.')
    loop.running = false
    setRunning(false)
    setPlaying(true)
    const net = agent.best.net
    const path = agent.greedyPath(agent.start, net).path
    const stepDuration = Math.max(0.06, Math.min(0.22, 6 / path.length))
    const tl = sceneRef.current.playRoute(path, { stepDuration, onDone: () => setPlaying(false) })
    path.forEach(([x, y], i) => {
      tl.call(
        () => {
          const q = net.predict(agent.features(x, y))
          const a = agent.argmax(q)
          netRef.current.update(net, agent.activations(x, y, net), { action: i < path.length - 1 ? a : -1, duration: stepDuration })
          mapRef.current.setAgent([x, y])
        },
        null,
        i * stepDuration,
      )
    })
    const h = agent.valueMap(net)
    mapRef.current.setHeat(h)
    mapRef.current.setBest(path)
  }

  const onCell = (x, y) => {
    if (playing) return
    if (mode === 'wall') {
      if ((start && x === start[0] && y === start[1]) || (goal && x === goal[0] && y === goal[1])) return
      const cells = maze.cells.slice()
      cells[y * maze.width + x] ^= 1
      setWorld({ maze: { ...maze, cells }, start, goal })
      return
    }
    if (isWall(maze, x, y)) return flash('Esa celda es una pared.')
    if (mode === 'start') {
      if (goal && x === goal[0] && y === goal[1]) return
      setWorld({ maze, start: [x, y], goal })
    } else {
      if (start && x === start[0] && y === start[1]) return
      setWorld({ maze, start, goal: [x, y] })
    }
  }

  const randomMaze = () => {
    const m = generateMaze(size, size)
    setWorld({ maze: m, ...defaultEndpoints(m) })
  }

  const onFile = async (e) => {
    const f = e.target.files?.[0]
    e.target.value = ''
    if (!f) return
    try {
      const m = await mazeFromFile(f)
      const ep = defaultEndpoints(m)
      setWorld({ maze: m, ...ep })
      flash(`BMP cargado: ${m.width}×${m.height} celdas. Haz clic para elegir el punto de partida.`)
    } catch (err) {
      flash(err.message)
    }
  }

  const downloadBMP = () => {
    const url = URL.createObjectURL(mazeToBMP(maze))
    const a = document.createElement('a')
    a.href = url
    a.download = `laberinto-${maze.width}x${maze.height}.bmp`
    a.click()
    URL.revokeObjectURL(url)
  }

  // Entrada animada del panel
  useEffect(() => {
    const ctx = gsap.context(() => {
      gsap.from('.card', { y: 24, opacity: 0, duration: 0.7, stagger: 0.08, ease: 'power3.out' })
      gsap.from('.brand > *', { y: -12, opacity: 0, duration: 0.6, stagger: 0.1, ease: 'power2.out' })
    })
    return () => ctx.revert()
  }, [])

  useEffect(() => {
    if (!toast) return
    gsap.fromTo('.toast', { y: 20, opacity: 0 }, { y: 0, opacity: 1, duration: 0.35, ease: 'back.out(2)' })
    const t = setTimeout(() => setToast(null), 4200)
    return () => clearTimeout(t)
  }, [toast])

  const pct = stats ? Math.round((Math.min(stats.radius, stats.maxDist) / stats.maxDist) * 100) : 0
  const status = !optimal
    ? 'La salida no es alcanzable desde el inicio'
    : playing
      ? 'Recorriendo la ruta aprendida'
      : stats?.solvedAt != null
        ? 'Ruta óptima aprendida'
        : running
          ? 'Entrenando…'
          : stats?.steps
            ? 'En pausa'
            : 'Listo para entrenar'

  return (
    <div className="app">
      <header className="brand">
        <div className="logo">
          <span className="dot" /> RNA<span className="slash">/</span>laberinto
        </div>
        <p className="tag">
          Detección de obstáculos y búsqueda de salida con una red neuronal entrenada por refuerzo (Deep Q-Learning)
        </p>
      </header>

      <main className="layout">
        <section className="stage card">
          <Maze3D ref={sceneRef} maze={maze} start={start} goal={goal} onCellClick={onCell} />
          <div className="overlay top">
            <div className="seg" role="group" aria-label="Modo de clic">
              {[
                ['start', 'Inicio'],
                ['goal', 'Salida'],
                ['wall', 'Pared'],
              ].map(([k, l]) => (
                <button key={k} className={mode === k ? 'on ' + k : ''} onClick={() => setMode(k)}>
                  {l}
                </button>
              ))}
            </div>
            <span className={`status ${running ? 'live' : ''}`}>{status}</span>
          </div>
          <div className="overlay bottom">
            <button className="primary" onClick={toggleRun} disabled={!optimal}>
              {running ? 'Pausar' : stats?.steps ? 'Continuar' : 'Entrenar'}
            </button>
            <button onClick={stepOnce} disabled={!optimal || running}>
              Paso
            </button>
            <button onClick={playRoute} disabled={!stats?.best || playing}>
              Recorrer ruta aprendida
            </button>
            <div className="speed">
              {SPEEDS.map((s, i) => (
                <button key={s.label} className={speed === i ? 'on' : ''} onClick={() => setSpeed(i)}>
                  {s.label}
                </button>
              ))}
            </div>
          </div>
          <p className="hint">Clic en el suelo para colocar el {mode === 'start' ? 'inicio' : mode === 'goal' ? 'la salida' : 'o quitar paredes'} · arrastra para girar la cámara</p>
        </section>

        <aside className="side">
          <section className="card">
            <h2>Laberinto</h2>
            <div className="row">
              <label className="grow">
                <span>Tamaño aleatorio: {size}×{size} ({size * 2 + 1}px)</span>
                <input type="range" min="4" max="20" value={size} onChange={(e) => setSize(+e.target.value)} />
              </label>
            </div>
            <div className="row">
              <button onClick={randomMaze}>Generar aleatorio</button>
              <label className="button">
                Cargar BMP
                <input type="file" accept=".bmp,image/bmp,image/png,image/*" onChange={onFile} hidden />
              </label>
              <button onClick={downloadBMP}>Descargar BMP</button>
            </div>
            <p className="note">Negro = pared, blanco = libre. Cada píxel es una celda.</p>
          </section>

          <section className="card">
            <h2>Red neuronal</h2>
            <NetworkViz ref={netRef} />
            <div className="row compact">
              <label>
                <span>Capas ocultas</span>
                <select value={arch} onChange={(e) => setArch(e.target.value)}>
                  {Object.keys(ARCHS).map((k) => (
                    <option key={k}>{k}</option>
                  ))}
                </select>
              </label>
              <label>
                <span>Tasa de aprendizaje</span>
                <select value={lr} onChange={(e) => setLr(+e.target.value)}>
                  {[0.0005, 0.001, 0.0015, 0.003, 0.006].map((v) => (
                    <option key={v} value={v}>
                      {v}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                <span>γ descuento</span>
                <select value={gamma} onChange={(e) => setGamma(+e.target.value)}>
                  {[0.95, 0.97, 0.99, 0.995].map((v) => (
                    <option key={v} value={v}>
                      {v}
                    </option>
                  ))}
                </select>
              </label>
            </div>
            <div className="row">
              <button onClick={resetNet}>Reiniciar pesos</button>
              <label className="check">
                <input type="checkbox" checked={autoStop} onChange={(e) => setAutoStop(e.target.checked)} />
                Parar al hallar la ruta óptima
              </label>
            </div>
          </section>

          <section className="card">
            <h2>Entrenamiento</h2>
            <div className="stats">
              <Stat k="Episodio" v={stats?.episode ?? '—'} />
              <Stat k="Pasos" v={stats ? stats.steps.toLocaleString('es') : '—'} />
              <Stat k="Exploración ε" v={stats ? stats.eps.toFixed(2) : '—'} />
              <Stat k="Pérdida" v={stats ? stats.loss.toFixed(4) : '—'} />
              <Stat k="Ruta de la red" v={stats?.best ? `${stats.best - 1} pasos` : '—'} accent={stats?.solvedAt != null} />
              <Stat k="Óptima (BFS)" v={optimal ? `${optimal.length - 1} pasos` : '—'} />
            </div>
            <div className="curriculum">
              <div className="bar">
                <i style={{ width: pct + '%' }} />
              </div>
              <span>
                Currículo: arranca a ≤{stats ? Math.min(stats.radius, stats.maxDist) : 0} pasos de la salida ({pct}%)
              </span>
            </div>
            <EpisodeChart ref={chartRef} />
            <p className="legend">
              <i className="g" /> llegó <i className="r" /> se agotó el tiempo <i className="a" /> ruta de la red <i className="w" /> óptima
            </p>
          </section>

          <section className="card">
            <h2>Mapa de valores Q</h2>
            <MiniMap ref={mapRef} maze={maze} start={start} goal={goal} optimal={optimal} onCellClick={onCell} />
            <p className="note">
              Color = valor aprendido de cada celda; flechas = acción que elegiría la red. Línea verde: mejor ruta de la red;
              punteada: óptima.
            </p>
          </section>

          {stats && (
            <section className="card about">
              <h2>Cómo funciona</h2>
              <p>
                Entradas ({stats.inputs}): 4 sensores de obstáculo (N, E, S, O), posición normalizada (x, y) y una codificación
                one-hot de la celda ({stats.free} celdas libres). Salidas: el valor Q de moverse en cada dirección.{' '}
                {stats.params.toLocaleString('es')} parámetros.
              </p>
              <p>
                Recompensas: +1 al salir, −0,02 por paso, −0,3 al chocar con una pared. Se entrena con Double DQN, replay de
                experiencias, red objetivo y Adam. Un currículo inverso empieza los episodios cerca de la salida y aleja el
                punto de partida a medida que la red aprende.
              </p>
            </section>
          )}
        </aside>
      </main>
      {toast && (
        <div className="toast" key={toast.id}>
          {toast.msg}
        </div>
      )}
    </div>
  )
}

function Stat({ k, v, accent }) {
  return (
    <div className={`stat ${accent ? 'accent' : ''}`}>
      <span>{k}</span>
      <b>{v}</b>
    </div>
  )
}
