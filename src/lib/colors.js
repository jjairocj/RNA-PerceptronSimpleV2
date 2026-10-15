// Escala de calor (bajo → alto): azul profundo → violeta → ámbar → verde menta.
const STOPS = [
  [0, [20, 30, 70]],
  [0.35, [90, 50, 160]],
  [0.65, [255, 140, 60]],
  [1, [61, 255, 176]],
]

export function heatColor(t) {
  t = Math.max(0, Math.min(1, t))
  for (let i = 1; i < STOPS.length; i++) {
    const [t1, c1] = STOPS[i]
    const [t0, c0] = STOPS[i - 1]
    if (t <= t1) {
      const k = (t - t0) / (t1 - t0)
      const c = c0.map((v, j) => Math.round(v + (c1[j] - v) * k))
      return `rgb(${c[0]},${c[1]},${c[2]})`
    }
  }
  return 'rgb(61,255,176)'
}

// Peso / activación con signo: negativo = coral, positivo = cian.
export function signColor(v, alpha = 1) {
  return v >= 0 ? `rgba(79,209,255,${alpha})` : `rgba(255,107,107,${alpha})`
}
