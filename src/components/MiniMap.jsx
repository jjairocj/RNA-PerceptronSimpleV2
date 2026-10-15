import { forwardRef, useEffect, useImperativeHandle, useRef } from 'react'
import { heatColor } from '../lib/colors.js'
import { DIRS } from '../lib/maze.js'

// Mapa 2D: editor (clic = inicio / salida / pared) + mapa de calor del valor
// aprendido, flechas de la política y comparación con la ruta óptima (BFS).
const MiniMap = forwardRef(function MiniMap({ maze, start, goal, optimal, onCellClick }, ref) {
  const canvasRef = useRef(null)
  const S = useRef({ heat: null, best: null, agent: null, route: null, showArrows: true }).current
  const props = useRef({})
  props.current = { maze, start, goal, optimal }

  const draw = () => {
    const canvas = canvasRef.current
    if (!canvas) return
    const { maze: m, start: st, goal: g, optimal: opt } = props.current
    const r = canvas.getBoundingClientRect()
    const dpr = Math.min(2, window.devicePixelRatio)
    canvas.width = r.width * dpr
    canvas.height = r.height * dpr
    const ctx = canvas.getContext('2d')
    const cs = Math.min(canvas.width / m.width, canvas.height / m.height)
    const ox = (canvas.width - cs * m.width) / 2
    const oy = (canvas.height - cs * m.height) / 2
    S.geom = { cs: cs / dpr, ox: ox / dpr, oy: oy / dpr }
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    // Descarta datos de un laberinto anterior (React reinicia después de dibujar).
    if (S.maze !== m) {
      S.maze = m
      S.heat = null
      S.best = null
      S.agent = null
    }
    let lo = Infinity
    let hi = -Infinity
    if (S.heat)
      for (const v of S.heat.values)
        if (!Number.isNaN(v)) {
          lo = Math.min(lo, v)
          hi = Math.max(hi, v)
        }
    for (let y = 0; y < m.height; y++)
      for (let x = 0; x < m.width; x++) {
        const i = y * m.width + x
        if (m.cells[i]) ctx.fillStyle = '#2b3d66'
        else if (S.heat && !Number.isNaN(S.heat.values[i])) ctx.fillStyle = heatColor((S.heat.values[i] - lo) / (hi - lo || 1))
        else ctx.fillStyle = '#0f1729'
        ctx.fillRect(ox + x * cs, oy + y * cs, cs + 0.5, cs + 0.5)
      }
    // Flechas de política
    if (S.heat && S.showArrows && cs > 7 * dpr) {
      ctx.strokeStyle = 'rgba(7,11,20,0.75)'
      ctx.lineWidth = Math.max(1, cs * 0.09)
      for (let y = 0; y < m.height; y++)
        for (let x = 0; x < m.width; x++) {
          const b = S.heat.best[y * m.width + x]
          if (b < 0) continue
          const cx = ox + (x + 0.5) * cs
          const cy = oy + (y + 0.5) * cs
          const [dx, dy] = DIRS[b]
          const L = cs * 0.3
          ctx.beginPath()
          ctx.moveTo(cx - dx * L, cy - dy * L)
          ctx.lineTo(cx + dx * L, cy + dy * L)
          ctx.lineTo(cx + dx * L - (dx + dy) * L * 0.5, cy + dy * L - (dy - dx) * L * 0.5)
          ctx.moveTo(cx + dx * L, cy + dy * L)
          ctx.lineTo(cx + dx * L - (dx - dy) * L * 0.5, cy + dy * L - (dy + dx) * L * 0.5)
          ctx.stroke()
        }
    }
    const line = (path, color, width, dash = []) => {
      if (!path || path.length < 2) return
      ctx.setLineDash(dash.map((d) => d * dpr))
      ctx.strokeStyle = color
      ctx.lineWidth = width * dpr
      ctx.lineJoin = 'round'
      ctx.beginPath()
      path.forEach(([x, y], i) => ctx[i ? 'lineTo' : 'moveTo'](ox + (x + 0.5) * cs, oy + (y + 0.5) * cs))
      ctx.stroke()
      ctx.setLineDash([])
    }
    line(opt, 'rgba(255,255,255,0.55)', 1.5, [3, 3])
    line(S.best, '#3dffb0', 2.5)
    const dot = (p, color, rad) => {
      if (!p) return
      ctx.beginPath()
      ctx.arc(ox + (p[0] + 0.5) * cs, oy + (p[1] + 0.5) * cs, Math.max(2 * dpr, cs * rad), 0, Math.PI * 2)
      ctx.fillStyle = color
      ctx.fill()
    }
    dot(st, '#ffb547', 0.38)
    dot(g, '#3dffb0', 0.38)
    if (S.agent) {
      dot(S.agent, '#fff', 0.28)
    }
  }

  useEffect(() => {
    const ro = new ResizeObserver(draw)
    ro.observe(canvasRef.current)
    return () => ro.disconnect()
  })
  useEffect(draw)

  useImperativeHandle(ref, () => ({
    setHeat(h) {
      S.heat = h
      draw()
    },
    setBest(p) {
      S.best = p
      draw()
    },
    setAgent(p) {
      S.agent = p
      draw()
    },
    reset() {
      S.heat = null
      S.best = null
      S.agent = null
      draw()
    },
  }))

  const onClick = (e) => {
    const r = canvasRef.current.getBoundingClientRect()
    const { cs, ox, oy } = S.geom
    const x = Math.floor((e.clientX - r.left - ox) / cs)
    const y = Math.floor((e.clientY - r.top - oy) / cs)
    if (x >= 0 && y >= 0 && x < maze.width && y < maze.height) onCellClick(x, y)
  }

  return <canvas ref={canvasRef} className="minimap" onClick={onClick} />
})

export default MiniMap
