import { forwardRef, useEffect, useImperativeHandle, useRef } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import gsap from 'gsap'
import { heatColor } from '../lib/colors.js'

const WALL_H = 0.9
const BALL_R = 0.3

// Escena 3D del laberinto. Se controla de forma imperativa (ref) para no
// re-renderizar React en cada paso del entrenamiento.
const Maze3D = forwardRef(function Maze3D({ maze, start, goal, onCellClick }, ref) {
  const mountRef = useRef(null)
  const S = useRef({}).current
  const clickRef = useRef(onCellClick)
  clickRef.current = onCellClick

  // Inicialización única de renderer, cámara, luces y esfera.
  useEffect(() => {
    const mount = mountRef.current
    const renderer = new THREE.WebGLRenderer({ antialias: true })
    renderer.setPixelRatio(Math.min(2, window.devicePixelRatio))
    renderer.shadowMap.enabled = true
    renderer.shadowMap.type = THREE.PCFSoftShadowMap
    renderer.toneMapping = THREE.ACESFilmicToneMapping
    mount.appendChild(renderer.domElement)

    const scene = new THREE.Scene()
    scene.background = new THREE.Color('#070b14')
    scene.fog = new THREE.Fog('#070b14', 40, 120)
    const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 500)
    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    controls.maxPolarAngle = Math.PI * 0.47

    scene.add(new THREE.HemisphereLight('#9fc3ff', '#1a1020', 0.9))
    const sun = new THREE.DirectionalLight('#ffffff', 1.6)
    sun.castShadow = true
    sun.shadow.mapSize.set(2048, 2048)
    scene.add(sun, sun.target)

    const ball = new THREE.Mesh(
      new THREE.SphereGeometry(BALL_R, 48, 32),
      new THREE.MeshStandardMaterial({ color: '#ffb547', emissive: '#ff7a00', emissiveIntensity: 0.6, metalness: 0.3, roughness: 0.25 }),
    )
    ball.castShadow = true
    const glow = new THREE.PointLight('#ffa040', 6, 5, 1.6)
    glow.position.y = 0.4
    ball.add(glow)
    scene.add(ball)

    const group = new THREE.Group()
    scene.add(group)

    Object.assign(S, { renderer, scene, camera, controls, sun, ball, group, mount })
    S.xTo = gsap.quickTo(ball.position, 'x', { duration: 0.15, ease: 'power2.out' })
    S.zTo = gsap.quickTo(ball.position, 'z', { duration: 0.15, ease: 'power2.out' })

    const resize = () => {
      const w = mount.clientWidth
      const h = mount.clientHeight
      renderer.setSize(w, h)
      camera.aspect = w / h
      camera.updateProjectionMatrix()
    }
    const ro = new ResizeObserver(resize)
    ro.observe(mount)
    resize()

    // Clic sobre el suelo → celda (se ignora si hubo arrastre de cámara).
    const ray = new THREE.Raycaster()
    const plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0)
    let down = null
    const onDown = (e) => (down = [e.clientX, e.clientY])
    const onUp = (e) => {
      if (!down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 5) return
      const r = renderer.domElement.getBoundingClientRect()
      const ndc = new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1)
      ray.setFromCamera(ndc, camera)
      const hits = S.walls ? ray.intersectObject(S.walls) : []
      const p = new THREE.Vector3()
      if (hits.length && hits[0].point.y > 0.01) p.copy(hits[0].point).sub(hits[0].face.normal.clone().multiplyScalar(0.05))
      else if (!ray.ray.intersectPlane(plane, p)) return
      clickRef.current?.(Math.floor(p.x), Math.floor(p.z))
    }
    renderer.domElement.addEventListener('pointerdown', onDown)
    renderer.domElement.addEventListener('pointerup', onUp)

    let raf
    const loop = (t) => {
      raf = requestAnimationFrame(loop)
      controls.update()
      if (S.goalMarker) {
        S.goalMarker.rotation.y = t * 0.001
        S.goalMarker.position.y = 0.05 + Math.sin(t * 0.003) * 0.05
      }
      // Rodar la esfera según su desplazamiento.
      const last = S.lastBall || ball.position.clone()
      const d = ball.position.clone().sub(last)
      if (d.lengthSq() > 1e-8) {
        const axis = new THREE.Vector3(d.z, 0, -d.x).normalize()
        ball.rotateOnWorldAxis(axis, d.length() / BALL_R)
      }
      S.lastBall = ball.position.clone()
      renderer.render(scene, camera)
    }
    raf = requestAnimationFrame(loop)

    return () => {
      cancelAnimationFrame(raf)
      ro.disconnect()
      renderer.domElement.removeEventListener('pointerdown', onDown)
      renderer.domElement.removeEventListener('pointerup', onUp)
      renderer.dispose()
      mount.removeChild(renderer.domElement)
    }
  }, [S])

  // Reconstruir geometría cuando cambia el laberinto.
  useEffect(() => {
    const { group, camera, controls, sun } = S
    group.clear()
    const { width: W, height: H, cells } = maze

    const floor = new THREE.Mesh(
      new THREE.PlaneGeometry(W + 4, H + 4),
      new THREE.MeshStandardMaterial({ color: '#0d1424', roughness: 0.95 }),
    )
    floor.rotation.x = -Math.PI / 2
    floor.position.set(W / 2, -0.001, H / 2)
    floor.receiveShadow = true
    group.add(floor)

    const grid = new THREE.GridHelper(Math.max(W, H), Math.max(W, H), '#1d2a44', '#141d31')
    grid.position.set(Math.max(W, H) / 2, 0.001, Math.max(W, H) / 2)
    group.add(grid)

    // Paredes (instanciadas)
    const wallCount = cells.reduce((a, c) => a + c, 0)
    const walls = new THREE.InstancedMesh(
      new THREE.BoxGeometry(1, WALL_H, 1),
      new THREE.MeshStandardMaterial({ color: '#3a5288', roughness: 0.5, metalness: 0.1 }),
      Math.max(1, wallCount),
    )
    walls.castShadow = true
    walls.receiveShadow = true
    const m = new THREE.Matrix4()
    let k = 0
    for (let y = 0; y < H; y++)
      for (let x = 0; x < W; x++)
        if (cells[y * W + x]) walls.setMatrixAt(k++, m.makeTranslation(x + 0.5, WALL_H / 2, y + 0.5))
    walls.count = k
    group.add(walls)
    S.walls = walls

    // Baldosas del mapa de calor (valor aprendido por celda)
    const freeIdx = []
    for (let i = 0; i < W * H; i++) if (!cells[i]) freeIdx.push(i)
    const tiles = new THREE.InstancedMesh(
      new THREE.PlaneGeometry(0.94, 0.94),
      new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.55, depthWrite: false }),
      Math.max(1, freeIdx.length),
    )
    const rot = new THREE.Matrix4().makeRotationX(-Math.PI / 2)
    const c = new THREE.Color('#111a2e')
    freeIdx.forEach((i, n) => {
      tiles.setMatrixAt(n, new THREE.Matrix4().makeTranslation((i % W) + 0.5, 0.01, Math.floor(i / W) + 0.5).multiply(rot))
      tiles.setColorAt(n, c)
    })
    tiles.instanceColor.needsUpdate = true
    group.add(tiles)
    S.tiles = tiles
    S.freeIdx = freeIdx

    // Rastro y ruta
    S.trail = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: '#4fd1ff', transparent: true, opacity: 0.5 }))
    group.add(S.trail)
    S.route = new THREE.Group()
    group.add(S.route)

    // Cámara y sombra ajustadas al tamaño
    const size = Math.max(W, H)
    controls.target.set(W / 2, 0, H / 2)
    camera.position.set(W / 2, size * 1.25, H / 2 + size * 0.95)
    camera.far = size * 6
    camera.updateProjectionMatrix()
    sun.position.set(W * 0.2, size * 1.2, H * 0.1)
    sun.target.position.set(W / 2, 0, H / 2)
    const sc = sun.shadow.camera
    sc.left = sc.bottom = -size
    sc.right = sc.top = size
    sc.far = size * 4
    sc.updateProjectionMatrix()
    S.scene.fog.near = size * 2
    S.scene.fog.far = size * 5
  }, [maze, S])

  // Marcadores de inicio y salida
  useEffect(() => {
    const { group } = S
    if (S.startMarker) group.remove(S.startMarker)
    if (S.goalMarker) group.remove(S.goalMarker)
    if (start) {
      const sm = new THREE.Mesh(new THREE.RingGeometry(0.28, 0.42, 40), new THREE.MeshBasicMaterial({ color: '#ffb547', side: THREE.DoubleSide }))
      sm.rotation.x = -Math.PI / 2
      sm.position.set(start[0] + 0.5, 0.02, start[1] + 0.5)
      group.add(sm)
      S.startMarker = sm
      gsap.killTweensOf(S.ball.position)
      S.ball.position.set(start[0] + 0.5, BALL_R, start[1] + 0.5)
      S.xTo = gsap.quickTo(S.ball.position, 'x', { duration: 0.15, ease: 'power2.out' })
      S.zTo = gsap.quickTo(S.ball.position, 'z', { duration: 0.15, ease: 'power2.out' })
      gsap.from(sm.scale, { x: 0, y: 0, z: 0, duration: 0.6, ease: 'back.out(3)' })
    }
    if (goal) {
      const gm = new THREE.Group()
      const ring = new THREE.Mesh(new THREE.TorusGeometry(0.34, 0.06, 16, 48), new THREE.MeshStandardMaterial({ color: '#3dffb0', emissive: '#14ff9c', emissiveIntensity: 1.2 }))
      ring.rotation.x = Math.PI / 2
      const beam = new THREE.Mesh(
        new THREE.CylinderGeometry(0.3, 0.3, 3, 32, 1, true),
        new THREE.MeshBasicMaterial({ color: '#3dffb0', transparent: true, opacity: 0.12, side: THREE.DoubleSide, depthWrite: false }),
      )
      beam.position.y = 1.5
      const light = new THREE.PointLight('#3dffb0', 4, 4)
      light.position.y = 0.6
      gm.add(ring, beam, light)
      gm.position.set(goal[0] + 0.5, 0.05, goal[1] + 0.5)
      group.add(gm)
      S.goalMarker = gm
      gsap.from(gm.scale, { x: 0, y: 0, z: 0, duration: 0.8, ease: 'elastic.out(1, 0.5)' })
    }
  }, [start, goal, maze, S])

  useImperativeHandle(ref, () => ({
    // Mover la esfera a una celda (suavizado con GSAP).
    moveBall(x, y, duration = 0.15, hit = false) {
      const { ball } = S
      if (duration <= 0) {
        gsap.killTweensOf(ball.position)
        ball.position.x = x + 0.5
        ball.position.z = y + 0.5
        S.xTo = gsap.quickTo(ball.position, 'x', { duration: 0.08, ease: 'none' })
        S.zTo = gsap.quickTo(ball.position, 'z', { duration: 0.08, ease: 'none' })
        return
      }
      S.xTo.tween.duration(duration)
      S.zTo.tween.duration(duration)
      S.xTo(x + 0.5)
      S.zTo(y + 0.5)
      if (hit) {
        gsap.fromTo(ball.material, { emissiveIntensity: 2.5 }, { emissiveIntensity: 0.6, duration: 0.3 })
        ball.material.emissive.set('#ff2a2a')
        gsap.delayedCall(0.3, () => ball.material.emissive.set('#ff7a00'))
      }
    },
    setTrail(points) {
      const pts = points.map(([x, y]) => new THREE.Vector3(x + 0.5, 0.06, y + 0.5))
      S.trail.geometry.dispose()
      S.trail.geometry = new THREE.BufferGeometry().setFromPoints(pts)
    },
    setHeat(values, lo, hi) {
      if (!S.tiles) return
      const c = new THREE.Color()
      S.freeIdx.forEach((i, n) => {
        const v = values[i]
        if (Number.isNaN(v)) return
        c.set(heatColor((v - lo) / (hi - lo || 1)))
        S.tiles.setColorAt(n, c)
      })
      S.tiles.instanceColor.needsUpdate = true
    },
    // Dibuja la ruta aprendida como tubo luminoso.
    showRoute(path, color = '#3dffb0') {
      S.route.clear()
      if (!path || path.length < 2) return
      const curve = new THREE.CatmullRomCurve3(path.map(([x, y]) => new THREE.Vector3(x + 0.5, 0.12, y + 0.5)), false, 'catmullrom', 0.1)
      const tube = new THREE.Mesh(
        new THREE.TubeGeometry(curve, path.length * 6, 0.07, 8),
        new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 0.9 }),
      )
      S.route.add(tube)
      return tube
    },
    // Recorre la ruta con una línea de tiempo GSAP.
    playRoute(path, { stepDuration = 0.12, onDone } = {}) {
      const { ball } = S
      gsap.killTweensOf(ball.position)
      const tl = gsap.timeline({
        onComplete: () => {
          S.xTo = gsap.quickTo(ball.position, 'x', { duration: 0.15, ease: 'power2.out' })
          S.zTo = gsap.quickTo(ball.position, 'z', { duration: 0.15, ease: 'power2.out' })
          gsap.fromTo(ball.scale, { x: 1.6, y: 1.6, z: 1.6 }, { x: 1, y: 1, z: 1, duration: 0.6, ease: 'elastic.out(1,0.4)' })
          onDone?.()
        },
      })
      ball.position.set(path[0][0] + 0.5, BALL_R, path[0][1] + 0.5)
      const tube = this.showRoute(path)
      if (tube) {
        const total = tube.geometry.index.count
        tube.geometry.setDrawRange(0, 0)
        const prog = { v: 0 }
        tl.to(prog, { v: 1, duration: stepDuration * path.length, ease: 'none', onUpdate: () => tube.geometry.setDrawRange(0, Math.floor((prog.v * total) / 6) * 6) }, 0)
      }
      path.slice(1).forEach(([x, y], i) => {
        tl.to(ball.position, { x: x + 0.5, z: y + 0.5, duration: stepDuration, ease: 'none' }, i * stepDuration)
      })
      S.timeline = tl
      return tl
    },
    stopRoute() {
      S.timeline?.kill()
    },
  }))

  return <div ref={mountRef} className="scene" />
})

export default Maze3D
