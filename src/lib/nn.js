// Perceptrón multicapa (MLP) escrito a mano: tanh en capas ocultas, salida lineal,
// optimizador Adam. La entrada es dispersa ({ idx, val }): sólo se recorren las
// entradas activas, lo que hace barata la codificación one-hot de la posición.

export const sparse = (idx, val) => ({ idx: Int32Array.from(idx), val: Float32Array.from(val) })

export function toDense(x, size) {
  const d = new Float32Array(size)
  for (let k = 0; k < x.idx.length; k++) d[x.idx[k]] = x.val[k]
  return d
}

export class MLP {
  constructor(sizes, { lr = 0.003 } = {}) {
    this.sizes = sizes
    this.lr = lr
    this.t = 0
    this.W = [] // W[l][i * nOut + j]: peso de la neurona i (capa l) a la j (capa l+1)
    this.b = []
    this.mW = []
    this.vW = []
    this.mb = []
    this.vb = []
    this.gW = []
    this.gb = []
    for (let l = 0; l < sizes.length - 1; l++) {
      const n = sizes[l] * sizes[l + 1]
      const W = new Float32Array(n)
      const scale = Math.sqrt(1 / (l === 0 ? 8 : sizes[l]))
      for (let i = 0; i < n; i++) W[i] = gauss() * scale
      this.W.push(W)
      this.b.push(new Float32Array(sizes[l + 1]))
      this.mW.push(new Float32Array(n))
      this.vW.push(new Float32Array(n))
      this.gW.push(new Float32Array(n))
      this.mb.push(new Float32Array(sizes[l + 1]))
      this.vb.push(new Float32Array(sizes[l + 1]))
      this.gb.push(new Float32Array(sizes[l + 1]))
    }
    this.mark = new Uint8Array(sizes[0])
  }

  // Devuelve las activaciones de todas las capas; acts[0] es la entrada dispersa.
  forward(x) {
    const L = this.W.length
    const acts = [x]
    // Capa 0 (dispersa)
    const h0 = this.sizes[1]
    let a = Float32Array.from(this.b[0])
    const W0 = this.W[0]
    for (let k = 0; k < x.idx.length; k++) {
      const v = x.val[k]
      const off = x.idx[k] * h0
      for (let j = 0; j < h0; j++) a[j] += v * W0[off + j]
    }
    if (L > 1) for (let j = 0; j < h0; j++) a[j] = Math.tanh(a[j])
    acts.push(a)
    for (let l = 1; l < L; l++) {
      const nIn = this.sizes[l]
      const nOut = this.sizes[l + 1]
      const W = this.W[l]
      const z = Float32Array.from(this.b[l])
      for (let i = 0; i < nIn; i++) {
        const ai = a[i]
        const off = i * nOut
        for (let j = 0; j < nOut; j++) z[j] += ai * W[off + j]
      }
      if (l < L - 1) for (let j = 0; j < nOut; j++) z[j] = Math.tanh(z[j])
      acts.push(z)
      a = z
    }
    return acts
  }

  predict(x) {
    const acts = this.forward(x)
    return acts[acts.length - 1]
  }

  // Entrena con un lote: sólo la salida `actions[k]` recibe error (Q-learning).
  // Pérdida Huber. Devuelve la pérdida media.
  trainBatch(xs, actions, targets) {
    const L = this.W.length
    const h0 = this.sizes[1]
    for (let l = 1; l < L; l++) this.gW[l].fill(0)
    for (let l = 0; l < L; l++) this.gb[l].fill(0)
    const touched = []
    let loss = 0
    const n = xs.length
    for (let k = 0; k < n; k++) {
      const x = xs[k]
      const acts = this.forward(x)
      let delta = new Float32Array(this.sizes[L])
      const err = acts[L][actions[k]] - targets[k]
      loss += Math.abs(err) < 1 ? 0.5 * err * err : Math.abs(err) - 0.5
      delta[actions[k]] = Math.max(-1, Math.min(1, err))
      for (let l = L - 1; l >= 1; l--) {
        const nIn = this.sizes[l]
        const nOut = this.sizes[l + 1]
        const aIn = acts[l]
        const W = this.W[l]
        const gW = this.gW[l]
        const gb = this.gb[l]
        for (let j = 0; j < nOut; j++) gb[j] += delta[j]
        const prev = new Float32Array(nIn)
        for (let i = 0; i < nIn; i++) {
          const ai = aIn[i]
          const off = i * nOut
          let s = 0
          for (let j = 0; j < nOut; j++) {
            gW[off + j] += ai * delta[j]
            s += W[off + j] * delta[j]
          }
          prev[i] = s * (1 - ai * ai) // derivada de tanh
        }
        delta = prev
      }
      const gb0 = this.gb[0]
      const gW0 = this.gW[0]
      for (let j = 0; j < h0; j++) gb0[j] += delta[j]
      for (let q = 0; q < x.idx.length; q++) {
        const i = x.idx[q]
        const v = x.val[q]
        if (!this.mark[i]) {
          this.mark[i] = 1
          touched.push(i)
        }
        const off = i * h0
        for (let j = 0; j < h0; j++) gW0[off + j] += v * delta[j]
      }
    }
    this.adam(1 / n, touched)
    return loss / n
  }

  // Adam; en la capa 0 sólo se actualizan las filas usadas en el lote ("lazy Adam").
  adam(scale, touched) {
    const b1 = 0.9
    const b2 = 0.999
    const eps = 1e-8
    this.t++
    const c1 = 1 - Math.pow(b1, this.t)
    const c2 = 1 - Math.pow(b2, this.t)
    const lr = this.lr
    const upd = (P, G, M, V, from, to) => {
      for (let i = from; i < to; i++) {
        const g = G[i] * scale
        M[i] = b1 * M[i] + (1 - b1) * g
        V[i] = b2 * V[i] + (1 - b2) * g * g
        P[i] -= (lr * (M[i] / c1)) / (Math.sqrt(V[i] / c2) + eps)
      }
    }
    const h0 = this.sizes[1]
    for (const i of touched) {
      upd(this.W[0], this.gW[0], this.mW[0], this.vW[0], i * h0, (i + 1) * h0)
      this.gW[0].fill(0, i * h0, (i + 1) * h0)
      this.mark[i] = 0
    }
    for (let l = 0; l < this.W.length; l++) {
      if (l > 0) upd(this.W[l], this.gW[l], this.mW[l], this.vW[l], 0, this.W[l].length)
      upd(this.b[l], this.gb[l], this.mb[l], this.vb[l], 0, this.b[l].length)
    }
  }

  copyFrom(o) {
    for (let l = 0; l < this.W.length; l++) {
      this.W[l].set(o.W[l])
      this.b[l].set(o.b[l])
    }
  }
}

function gauss() {
  let u = 0
  let v = 0
  while (u === 0) u = Math.random()
  while (v === 0) v = Math.random()
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v)
}
