// Laberinto como rejilla: 1 = pared (negro), 0 = libre (blanco).
// Un píxel del BMP = una celda.

export function createMaze(width, height, cells) {
  return { width, height, cells }
}

export const isWall = (m, x, y) =>
  x < 0 || y < 0 || x >= m.width || y >= m.height || m.cells[y * m.width + x] === 1

// Generador aleatorio: backtracker recursivo + algunos ciclos extra
// (para que haya más de un camino y la ruta óptima no sea trivial).
export function generateMaze(cw, ch, loops = 0.06) {
  const width = cw * 2 + 1
  const height = ch * 2 + 1
  const cells = new Uint8Array(width * height).fill(1)
  const visited = new Uint8Array(cw * ch)
  const stack = [[0, 0]]
  visited[0] = 1
  cells[1 * width + 1] = 0
  const dirs = [
    [0, -1],
    [1, 0],
    [0, 1],
    [-1, 0],
  ]
  while (stack.length) {
    const [cx, cy] = stack[stack.length - 1]
    const opts = dirs.filter(([dx, dy]) => {
      const nx = cx + dx
      const ny = cy + dy
      return nx >= 0 && ny >= 0 && nx < cw && ny < ch && !visited[ny * cw + nx]
    })
    if (!opts.length) {
      stack.pop()
      continue
    }
    const [dx, dy] = opts[(Math.random() * opts.length) | 0]
    const nx = cx + dx
    const ny = cy + dy
    visited[ny * cw + nx] = 1
    cells[(cy * 2 + 1 + dy) * width + (cx * 2 + 1 + dx)] = 0
    cells[(ny * 2 + 1) * width + (nx * 2 + 1)] = 0
    stack.push([nx, ny])
  }
  // Abrir algunas paredes internas para crear ciclos.
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      if (cells[y * width + x] !== 1 || Math.random() > loops) continue
      const h = cells[y * width + x - 1] === 0 && cells[y * width + x + 1] === 0
      const v = cells[(y - 1) * width + x] === 0 && cells[(y + 1) * width + x] === 0
      if (h !== v) cells[y * width + x] = 0
    }
  }
  // Salida en el borde derecho-inferior.
  cells[(height - 2) * width + (width - 1)] = 0
  return createMaze(width, height, cells)
}

// Lee un BMP (o cualquier imagen) usando el decodificador del navegador.
export async function mazeFromFile(file, maxSize = 61) {
  const url = URL.createObjectURL(file)
  try {
    const img = await new Promise((res, rej) => {
      const i = new Image()
      i.onload = () => res(i)
      i.onerror = () => rej(new Error('No se pudo leer la imagen'))
      i.src = url
    })
    let w = img.naturalWidth
    let h = img.naturalHeight
    const k = Math.min(1, maxSize / Math.max(w, h))
    w = Math.max(3, Math.round(w * k))
    h = Math.max(3, Math.round(h * k))
    const c = document.createElement('canvas')
    c.width = w
    c.height = h
    const ctx = c.getContext('2d')
    ctx.imageSmoothingEnabled = false
    ctx.drawImage(img, 0, 0, w, h)
    const data = ctx.getImageData(0, 0, w, h).data
    const cells = new Uint8Array(w * h)
    for (let i = 0; i < w * h; i++) {
      const lum = 0.299 * data[i * 4] + 0.587 * data[i * 4 + 1] + 0.114 * data[i * 4 + 2]
      cells[i] = lum < 128 ? 1 : 0
    }
    return createMaze(w, h, cells)
  } finally {
    URL.revokeObjectURL(url)
  }
}

// Exporta el laberinto como BMP de 24 bits (blanco = libre, negro = pared).
export function mazeToBMP(m) {
  const rowSize = Math.ceil((m.width * 3) / 4) * 4
  const size = 54 + rowSize * m.height
  const buf = new ArrayBuffer(size)
  const dv = new DataView(buf)
  dv.setUint8(0, 0x42)
  dv.setUint8(1, 0x4d)
  dv.setUint32(2, size, true)
  dv.setUint32(10, 54, true)
  dv.setUint32(14, 40, true)
  dv.setInt32(18, m.width, true)
  dv.setInt32(22, m.height, true)
  dv.setUint16(26, 1, true)
  dv.setUint16(28, 24, true)
  dv.setUint32(34, rowSize * m.height, true)
  for (let y = 0; y < m.height; y++) {
    const row = 54 + (m.height - 1 - y) * rowSize // BMP va de abajo hacia arriba
    for (let x = 0; x < m.width; x++) {
      const v = m.cells[y * m.width + x] ? 0 : 255
      dv.setUint8(row + x * 3, v)
      dv.setUint8(row + x * 3 + 1, v)
      dv.setUint8(row + x * 3 + 2, v)
    }
  }
  return new Blob([buf], { type: 'image/bmp' })
}

// Salida por defecto: primera celda libre en el borde (preferentemente lejos del inicio).
export function findExit(m, start) {
  const border = []
  for (let x = 0; x < m.width; x++) {
    border.push([x, 0], [x, m.height - 1])
  }
  for (let y = 1; y < m.height - 1; y++) {
    border.push([0, y], [m.width - 1, y])
  }
  const free = border.filter(([x, y]) => !isWall(m, x, y))
  const pool = free.length ? free : allFree(m)
  if (!pool.length) return null
  const s = start || [0, 0]
  pool.sort((a, b) => dist2(b, s) - dist2(a, s))
  return pool[0]
}

export function firstFree(m) {
  for (let y = 0; y < m.height; y++)
    for (let x = 0; x < m.width; x++) if (!isWall(m, x, y)) return [x, y]
  return null
}

export function allFree(m) {
  const r = []
  for (let y = 0; y < m.height; y++)
    for (let x = 0; x < m.width; x++) if (!isWall(m, x, y)) r.push([x, y])
  return r
}

const dist2 = (a, b) => (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

// BFS: ruta óptima de referencia (para comparar con lo que aprende la red).
export function bfs(m, start, goal) {
  const prev = new Int32Array(m.width * m.height).fill(-1)
  const si = start[1] * m.width + start[0]
  const gi = goal[1] * m.width + goal[0]
  prev[si] = si
  const q = [si]
  let h = 0
  while (h < q.length) {
    const c = q[h++]
    if (c === gi) break
    const x = c % m.width
    const y = (c / m.width) | 0
    for (const [dx, dy] of DIRS) {
      const nx = x + dx
      const ny = y + dy
      if (isWall(m, nx, ny)) continue
      const ni = ny * m.width + nx
      if (prev[ni] !== -1) continue
      prev[ni] = c
      q.push(ni)
    }
  }
  if (prev[gi] === -1) return null
  const path = []
  for (let c = gi; ; c = prev[c]) {
    path.push([c % m.width, (c / m.width) | 0])
    if (c === si) break
  }
  return path.reverse()
}

// Distancia (en pasos) desde `goal` a cada celda; -1 si es inalcanzable.
export function distances(m, goal) {
  const d = new Int32Array(m.width * m.height).fill(-1)
  const gi = goal[1] * m.width + goal[0]
  d[gi] = 0
  const q = [gi]
  for (let h = 0; h < q.length; h++) {
    const c = q[h]
    const x = c % m.width
    const y = (c / m.width) | 0
    for (const [dx, dy] of DIRS) {
      const nx = x + dx
      const ny = y + dy
      if (isWall(m, nx, ny)) continue
      const ni = ny * m.width + nx
      if (d[ni] !== -1) continue
      d[ni] = d[c] + 1
      q.push(ni)
    }
  }
  return Array.from(d)
}

// N, E, S, O
export const DIRS = [
  [0, -1],
  [1, 0],
  [0, 1],
  [-1, 0],
]
