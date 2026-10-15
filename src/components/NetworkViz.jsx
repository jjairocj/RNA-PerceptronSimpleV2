import { forwardRef, useEffect, useImperativeHandle, useRef } from 'react'
import gsap from 'gsap'
import { signColor } from '../lib/colors.js'
import { ACTIONS, SENSOR_COUNT } from '../lib/agent.js'

const INPUT_LABELS = ['muro N', 'muro E', 'muro S', 'muro O', 'pos x', 'pos y', 'celda']

// Dibuja la red: entradas (sensores + posición), capas ocultas y salidas Q.
// El grosor/opacidad de cada conexión es la señal que fluye: |peso × activación|.
const NetworkViz = forwardRef(function NetworkViz(_, ref) {
  const canvasRef = useRef(null)
  const S = useRef({ disp: null, target: null, net: null, cell: -1, action: -1, explore: false }).current

  useEffect(() => {
    const canvas = canvasRef.current
    const resize = () => {
      const r = canvas.getBoundingClientRect()
      const dpr = Math.min(2, window.devicePixelRatio)
      canvas.width = r.width * dpr
      canvas.height = r.height * dpr
      S.dpr = dpr
      draw()
    }
    const ro = new ResizeObserver(resize)
    ro.observe(canvas)
    S.draw = draw
    return () => ro.disconnect()

    function draw() {
      const ctx = canvas.getContext('2d')
      const W = canvas.width
      const H = canvas.height
      const dpr = S.dpr || 1
      ctx.clearRect(0, 0, W, H)
      if (!S.disp || !S.net) return
      const net = S.net
      const layers = S.layout
      const vals = S.disp
      const padL = 104 * dpr
      const padR = 92 * dpr
      const padY = 14 * dpr
      const padB = 22 * dpr
      const xs = layers.map((_, i) => padL + ((W - padL - padR) * i) / (layers.length - 1))
      const pos = layers.map((n, li) =>
        Array.from({ length: n }, (_, j) => [xs[li], n === 1 ? (H - padB) / 2 : padY + ((H - padY - padB) * j) / (n - 1)]),
      )
      const val = (li, j) => vals[S.offsets[li] + j]

      // Conexiones
      ctx.lineCap = 'round'
      for (let li = 0; li < layers.length - 1; li++) {
        const nOut = net.sizes[li + 1]
        const Wl = net.W[li]
        for (let i = 0; i < layers[li]; i++) {
          const a = val(li, i)
          if (Math.abs(a) < 0.02) continue
          // fila real de la matriz de pesos: en la capa 0 el nodo "celda" es la one-hot activa
          const row = li === 0 && i === SENSOR_COUNT + 2 ? S.cell : i
          if (row < 0) continue
          const [x0, y0] = pos[li][i]
          for (let j = 0; j < layers[li + 1]; j++) {
            const w = Wl[row * nOut + j]
            const s = Math.abs(w * a)
            if (s < 0.04) continue
            const [x1, y1] = pos[li + 1][j]
            ctx.strokeStyle = signColor(w * a, Math.min(0.85, s * 0.9))
            ctx.lineWidth = Math.min(2.5, 0.3 + s) * dpr
            ctx.beginPath()
            ctx.moveTo(x0, y0)
            ctx.bezierCurveTo((x0 + x1) / 2, y0, (x0 + x1) / 2, y1, x1, y1)
            ctx.stroke()
          }
        }
      }

      // Neuronas
      const last = layers.length - 1
      const qs = Array.from({ length: 4 }, (_, j) => val(last, j))
      const qMin = Math.min(...qs)
      const qMax = Math.max(...qs)
      for (let li = 0; li < layers.length; li++) {
        const r = (li === 0 || li === last ? 6 : layers[li] > 30 ? 3.2 : 4.2) * dpr
        for (let j = 0; j < layers[li]; j++) {
          const [x, y] = pos[li][j]
          let v = val(li, j)
          if (li === last) v = (v - qMin) / (qMax - qMin || 1)
          ctx.beginPath()
          ctx.arc(x, y, r, 0, Math.PI * 2)
          ctx.fillStyle = li === last ? `rgba(61,255,176,${0.15 + 0.85 * v})` : signColor(v, 0.15 + 0.85 * Math.min(1, Math.abs(v)))
          ctx.fill()
          ctx.lineWidth = 1 * dpr
          ctx.strokeStyle = 'rgba(160,180,220,0.35)'
          ctx.stroke()
        }
      }

      // Etiquetas
      ctx.font = `${10.5 * dpr}px "JetBrains Mono", monospace`
      ctx.textBaseline = 'middle'
      ctx.textAlign = 'right'
      INPUT_LABELS.forEach((l, j) => {
        const v = val(0, j)
        ctx.fillStyle = v > 0.5 ? '#e8eefc' : '#6b7a99'
        const txt = j === 6 ? `celda #${S.cell - SENSOR_COUNT - 2}` : j >= 4 ? `${l} ${v.toFixed(2)}` : l
        ctx.fillText(txt, pos[0][j][0] - 10 * dpr, pos[0][j][1])
      })
      ctx.textAlign = 'left'
      qs.forEach((q, j) => {
        const [x, y] = pos[last][j]
        const chosen = j === S.action
        ctx.fillStyle = chosen ? (S.explore ? '#ffb547' : '#3dffb0') : '#8a98b8'
        ctx.fillText(`${ACTIONS[j]} ${q >= 0 ? ' ' : ''}${q.toFixed(3)}`, x + 12 * dpr, y)
        if (chosen) {
          ctx.beginPath()
          ctx.arc(x, y, 10 * dpr, 0, Math.PI * 2)
          ctx.strokeStyle = S.explore ? '#ffb547' : '#3dffb0'
          ctx.lineWidth = 2 * dpr
          ctx.stroke()
        }
      })
      ctx.fillStyle = '#56648a'
      ctx.textAlign = 'center'
      const names = ['entrada', ...net.sizes.slice(1, -1).map((n) => `oculta·${n}`), 'Q']
      names.forEach((n, i) => ctx.fillText(n, xs[i], H - 5 * dpr))
    }
  }, [S])

  useImperativeHandle(ref, () => ({
    // acts: activaciones por capa (entrada densa). cellIndex: índice en la entrada.
    update(net, acts, { action = -1, explore = false, duration = 0.2 } = {}) {
      const inputSize = acts[0].length
      let cell = -1
      for (let i = SENSOR_COUNT + 2; i < inputSize; i++)
        if (acts[0][i] !== 0) {
          cell = i
          break
        }
      const shown = [Array.from(acts[0].slice(0, SENSOR_COUNT + 2)).concat([1]), ...acts.slice(1).map((a) => Array.from(a))]
      const flat = shown.flat()
      if (!S.disp || S.disp.length !== flat.length || S.net !== net) {
        S.disp = flat.slice()
        S.layout = shown.map((a) => a.length)
        S.offsets = S.layout.reduce((o, n, i) => (o.push(i ? o[i - 1] + S.layout[i - 1] : 0), o), [])
      }
      S.net = net
      S.cell = cell
      S.action = action
      S.explore = explore
      gsap.killTweensOf(S.disp)
      if (duration > 0) gsap.to(S.disp, { endArray: flat, duration, ease: 'power2.out', onUpdate: S.draw })
      else {
        S.disp = flat
        S.draw()
      }
    },
  }))

  return <canvas ref={canvasRef} className="netviz" />
})

export default NetworkViz
