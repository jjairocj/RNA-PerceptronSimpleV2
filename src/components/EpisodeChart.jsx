import { forwardRef, useEffect, useImperativeHandle, useRef } from 'react'

// Barras: pasos por episodio (verde = llegó a la salida, coral = tiempo agotado).
// Línea: longitud de la ruta de la red desde el inicio; discontinua: óptimo BFS.
const EpisodeChart = forwardRef(function EpisodeChart(_, ref) {
  const canvasRef = useRef(null)
  const S = useRef({ history: [], greedy: [], optimal: 0 }).current

  const draw = () => {
    const c = canvasRef.current
    if (!c) return
    const r = c.getBoundingClientRect()
    const dpr = Math.min(2, window.devicePixelRatio)
    c.width = r.width * dpr
    c.height = r.height * dpr
    const ctx = c.getContext('2d')
    const W = c.width
    const H = c.height
    ctx.clearRect(0, 0, W, H)
    const h = S.history.slice(-160)
    if (!h.length) {
      ctx.fillStyle = '#56648a'
      ctx.font = `${11 * dpr}px "JetBrains Mono", monospace`
      ctx.textAlign = 'center'
      ctx.fillText('sin episodios todavía', W / 2, H / 2)
      return
    }
    const max = Math.max(...h.map((e) => e.steps), S.optimal * 1.2, 10)
    const bw = W / 160
    const y = (v) => H - 4 * dpr - (v / max) * (H - 10 * dpr)
    h.forEach((e, i) => {
      ctx.fillStyle = e.reached ? (e.fromStart ? 'rgba(61,255,176,0.85)' : 'rgba(61,255,176,0.35)') : 'rgba(255,107,107,0.45)'
      const top = y(e.steps)
      ctx.fillRect(i * bw, top, Math.max(1, bw - 1), H - 4 * dpr - top)
    })
    if (S.optimal) {
      ctx.setLineDash([4 * dpr, 4 * dpr])
      ctx.strokeStyle = 'rgba(255,255,255,0.6)'
      ctx.lineWidth = 1 * dpr
      ctx.beginPath()
      ctx.moveTo(0, y(S.optimal))
      ctx.lineTo(W, y(S.optimal))
      ctx.stroke()
      ctx.setLineDash([])
    }
    const g = S.greedy.slice(-160)
    ctx.strokeStyle = '#ffb547'
    ctx.lineWidth = 2 * dpr
    ctx.beginPath()
    let pen = false
    g.forEach((v, i) => {
      if (v == null) {
        pen = false
        return
      }
      const px = i * bw + bw / 2
      if (pen) ctx.lineTo(px, y(v))
      else ctx.moveTo(px, y(v))
      pen = true
    })
    ctx.stroke()
  }

  useEffect(() => {
    const ro = new ResizeObserver(draw)
    ro.observe(canvasRef.current)
    return () => ro.disconnect()
  })

  useImperativeHandle(ref, () => ({
    update(history, greedy, optimal) {
      S.history = history
      S.greedy = greedy
      S.optimal = optimal
      draw()
    },
  }))

  return <canvas ref={canvasRef} className="chart" />
})

export default EpisodeChart
