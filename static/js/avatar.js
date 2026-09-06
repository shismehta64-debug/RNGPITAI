/* ==========================================================================
   SINA 3D avatar
   --------------------------------------------------------------------------
   What this replaces, and why:

   * Three independent requestAnimationFrame loops (avatar, particles, mouse
     parallax) plus a fourth for the chat wave background - all four ran
     forever, including while their tab was hidden and while the browser tab was
     in the background. There is now ONE loop that renders only what is visible
     and idles when the document is hidden.
   * Lip sync was `Math.sin(t * 12)` - a mouth flapping at a fixed rate with no
     relationship to the audio. It is now driven by a real WebAudio analyser:
     loudness opens the jaw, spectral balance picks the vowel shape.
   * Idle motion was a stack of raw sine waves written straight onto bone
     rotations, so every transition (idle -> speaking) was an instant snap.
     Motion now runs through critically-damped springs and blends between
     states, and secondary motion (arms, head) lags the body as it should.
   ========================================================================== */

import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { VRMLoaderPlugin, VRMUtils } from "@pixiv/three-vrm";

const REDUCED_MOTION = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const TAU = Math.PI * 2;

const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);
const lerp = (a, b, t) => a + (b - a) * t;
/** Frame-rate independent exponential smoothing. */
const damp = (a, b, lambda, dt) => lerp(a, b, 1 - Math.exp(-lambda * dt));

/** A critically damped spring - overshoot-free, physically plausible easing. */
class Spring {
  constructor(value = 0, stiffness = 90, damping = 18) {
    this.value = value;
    this.target = value;
    this.velocity = 0;
    this.stiffness = stiffness;
    this.damping = damping;
  }
  update(dt) {
    // Sub-step so a long frame cannot make the spring explode.
    const steps = Math.min(4, Math.ceil(dt / 0.016));
    const h = dt / steps;
    for (let i = 0; i < steps; i++) {
      const accel = (this.target - this.value) * this.stiffness - this.velocity * this.damping;
      this.velocity += accel * h;
      this.value += this.velocity * h;
    }
    return this.value;
  }
}

/* -------------------------------------------------------- audio analysis */

/**
 * Turns an <audio> element into per-frame mouth-shape data.
 * Falls back to a plausible envelope when WebAudio is unavailable or blocked.
 */
class VoiceAnalyser {
  constructor() {
    this.context = null;
    this.analyser = null;
    this.source = null;
    this.freq = null;
    this.time = null;
    this.energy = 0;
    this.centroid = 0.5;
    this.failed = false;
  }

  ensureContext() {
    if (this.failed) return null;
    if (!this.context) {
      const Ctx = window.AudioContext || window.webkitAudioContext;
      if (!Ctx) {
        this.failed = true;
        return null;
      }
      try {
        this.context = new Ctx();
        this.analyser = this.context.createAnalyser();
        this.analyser.fftSize = 1024;
        this.analyser.smoothingTimeConstant = 0.55;
        this.analyser.connect(this.context.destination);
        this.freq = new Uint8Array(this.analyser.frequencyBinCount);
        this.time = new Uint8Array(this.analyser.fftSize);
      } catch (err) {
        console.warn("[SINA] WebAudio unavailable, using fallback lip sync", err);
        this.failed = true;
        return null;
      }
    }
    return this.context;
  }

  /** Browsers suspend the context until a user gesture. */
  resume() {
    if (this.context && this.context.state === "suspended") {
      this.context.resume().catch(() => {});
    }
  }

  /** Route an audio element through the analyser. Returns false on failure. */
  attach(audioEl) {
    if (!this.ensureContext()) return false;
    try {
      this.detach();
      this.source = this.context.createMediaElementSource(audioEl);
      this.source.connect(this.analyser);
      this.resume();
      return true;
    } catch (err) {
      console.warn("[SINA] Could not attach audio to analyser", err);
      return false;
    }
  }

  detach() {
    if (this.source) {
      try { this.source.disconnect(); } catch { /* already gone */ }
      this.source = null;
    }
  }

  /** Sample the current frame: RMS loudness plus a coarse spectral centroid. */
  sample() {
    if (!this.analyser) return null;
    this.analyser.getByteTimeDomainData(this.time);
    this.analyser.getByteFrequencyData(this.freq);

    let sum = 0;
    for (let i = 0; i < this.time.length; i++) {
      const v = (this.time[i] - 128) / 128;
      sum += v * v;
    }
    const rms = Math.sqrt(sum / this.time.length);
    // Speech RMS sits well below 1; scale into a usable 0..1 opening.
    this.energy = clamp(rms * 5.5, 0, 1);

    // Centroid over the band that actually carries vowel identity (~150-4000Hz).
    const bins = this.freq.length;
    const lo = Math.floor(bins * 0.01);
    const hi = Math.floor(bins * 0.35);
    let weighted = 0;
    let total = 0;
    for (let i = lo; i < hi; i++) {
      const mag = this.freq[i];
      weighted += mag * (i - lo);
      total += mag;
    }
    if (total > 0) this.centroid = clamp(weighted / total / (hi - lo), 0, 1);
    return { energy: this.energy, centroid: this.centroid };
  }
}

/* ------------------------------------------------------------ the avatar */

export class SinaAvatar {
  constructor(canvas, options = {}) {
    this.canvas = canvas;
    this.onProgress = options.onProgress || (() => {});
    this.onStatus = options.onStatus || (() => {});

    this.vrm = null;
    this.state = "loading"; // loading | idle | listening | thinking | speaking
    this.clock = new THREE.Clock();
    this.time = 0;
    this.visible = true;
    this.running = false;

    this.voice = new VoiceAnalyser();
    this.audio = null;

    // ---- animation state -------------------------------------------------
    this.springs = {
      headYaw: new Spring(0, 70, 16),
      headPitch: new Spring(0, 70, 16),
      headRoll: new Spring(0, 55, 15),
      spineTwist: new Spring(0, 45, 14),
      lean: new Spring(0, 40, 13),
      armL: new Spring(0, 38, 12),
      armR: new Spring(0, 38, 12),
      jaw: new Spring(0, 210, 26),
      energy: new Spring(0, 60, 15),
      posture: new Spring(0, 26, 11),
    };
    this.gaze = { x: 0, y: 0 };
    this.pointer = { x: 0, y: 0, active: false };
    this.saccade = { x: 0, y: 0, next: 1.2 };
    this.blink = { value: 0, timer: 1.5, phase: -1, speed: 0.11, queued: 0 };
    this.gesture = null;
    this.visemes = { aa: 0, ih: 0, ou: 0, ee: 0, oh: 0 };
    this.expression = { happy: 0, relaxed: 0, surprised: 0 };

    // ---- adaptive quality ------------------------------------------------
    this.dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.fpsSamples = [];
    this.qualityChecked = 0;

    this.#initScene();
  }

  /* ------------------------------------------------------------- setup */
  #initScene() {
    this.scene = new THREE.Scene();

    this.camera = new THREE.PerspectiveCamera(26, 1, 0.1, 60);
    this.camera.position.set(0, 1.34, 1.85);

    this.renderer = new THREE.WebGLRenderer({
      canvas: this.canvas,
      alpha: true,
      antialias: true,
      powerPreference: "high-performance",
    });
    this.renderer.setPixelRatio(this.dpr);
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.1;

    // Three-point lighting plus a rim light: reads far better than the
    // original ambient + single directional.
    this.scene.add(new THREE.AmbientLight(0xffffff, 0.62));

    this.keyLight = new THREE.DirectionalLight(0xfff4e8, 1.45);
    this.keyLight.position.set(1.6, 2.6, 2.4);
    this.scene.add(this.keyLight);

    this.fillLight = new THREE.DirectionalLight(0x8f7fff, 0.55);
    this.fillLight.position.set(-2.2, 1.2, 1.0);
    this.scene.add(this.fillLight);

    this.rimLight = new THREE.DirectionalLight(0xa29bfe, 1.05);
    this.rimLight.position.set(-0.8, 2.2, -2.6);
    this.scene.add(this.rimLight);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.set(0, 1.34, 0);
    this.controls.enablePan = false;
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.rotateSpeed = 0.45;
    this.controls.minDistance = 0.9;
    this.controls.maxDistance = 3.4;
    this.controls.minPolarAngle = Math.PI * 0.22;
    this.controls.maxPolarAngle = Math.PI * 0.62;
    this.controls.update();

    this.#addParticles();
    this.#bindPointer();
    this.#observeSize();
    this.resize();
  }

  /**
   * Ambient sparks. Previously these lived in their own RAF loop that ran even
   * when the Sina tab was hidden; they are now updated inside the main loop.
   */
  #addParticles() {
    const COUNT = REDUCED_MOTION ? 0 : 110;
    if (!COUNT) {
      this.particles = null;
      return;
    }
    const positions = new Float32Array(COUNT * 3);
    const velocities = new Float32Array(COUNT);
    const phases = new Float32Array(COUNT);
    for (let i = 0; i < COUNT; i++) {
      positions[i * 3] = (Math.random() - 0.5) * 3.2;
      positions[i * 3 + 1] = Math.random() * 3.4 - 0.6;
      positions[i * 3 + 2] = (Math.random() - 0.5) * 1.8;
      velocities[i] = 0.035 + Math.random() * 0.055;
      phases[i] = Math.random() * TAU;
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    const material = new THREE.PointsMaterial({
      color: 0x9b8ffc,
      size: 0.018,
      transparent: true,
      opacity: 0.62,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });
    const points = new THREE.Points(geometry, material);
    points.frustumCulled = false;
    this.scene.add(points);
    this.particles = { points, geometry, positions, velocities, phases, count: COUNT };
  }

  #bindPointer() {
    const onMove = (clientX, clientY) => {
      const rect = this.canvas.getBoundingClientRect();
      if (!rect.width || !rect.height) return;
      this.pointer.x = ((clientX - rect.left) / rect.width - 0.5) * 2;
      this.pointer.y = ((clientY - rect.top) / rect.height - 0.5) * 2;
      this.pointer.active = true;
    };
    this.canvas.addEventListener("pointermove", (e) => onMove(e.clientX, e.clientY));
    this.canvas.addEventListener("pointerleave", () => { this.pointer.active = false; });
    // A user gesture is also our chance to unlock audio playback.
    const unlock = () => this.voice.resume();
    window.addEventListener("pointerdown", unlock, { once: true });
    window.addEventListener("keydown", unlock, { once: true });
  }

  /**
   * Track the canvas's real box rather than `window.innerWidth`.
   *
   * A ResizeObserver is not a nicety here: if the element is laid out at 0x0
   * when the scene is constructed - a hidden tab, a collapsed split pane, a
   * `display:none` ancestor - a one-shot sizing pass latches the renderer at
   * three.js's 300x150 default and nothing ever fixes it, because no `window`
   * resize event is fired when only the element's box changes.
   */
  #observeSize() {
    if (typeof ResizeObserver === "undefined") {
      window.addEventListener("resize", () => this.resize());
      return;
    }
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(this.canvas);
  }

  /**
   * Adapt to the page theme.
   *
   * The sparks use additive blending, which is invisible against a light
   * background - additive light on near-white is still near-white. On light
   * themes they switch to normal blending with a darker tint, and the rim light
   * is eased off so the model does not blow out.
   */
  setTheme(theme) {
    const light = theme === "light";
    if (this.particles) {
      const material = this.particles.points.material;
      material.blending = light ? THREE.NormalBlending : THREE.AdditiveBlending;
      material.color.setHex(light ? 0x6c5ce7 : 0x9b8ffc);
      material.opacity = light ? 0.34 : 0.62;
      material.needsUpdate = true;
    }
    this.rimLight.intensity = light ? 0.55 : 1.05;
    this.fillLight.intensity = light ? 0.35 : 0.55;
    this.keyLight.intensity = light ? 1.15 : 1.45;
  }

  resize() {
    const rect = this.canvas.getBoundingClientRect();
    const width = Math.floor(rect.width);
    const height = Math.floor(rect.height);
    if (!width || !height) {
      this.sized = false;
      return;
    }
    this.sized = true;
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height, false);
  }

  /* -------------------------------------------------------------- load */
  async load(url) {
    const loader = new GLTFLoader();
    loader.register((parser) => new VRMLoaderPlugin(parser));

    return new Promise((resolve, reject) => {
      loader.load(
        url,
        (gltf) => {
          const vrm = gltf.userData.vrm;
          if (!vrm) {
            reject(new Error("File is not a VRM model"));
            return;
          }
          // Removing unused vertices/joints measurably cuts per-frame skinning
          // cost, and combining skeletons reduces draw calls.
          VRMUtils.removeUnnecessaryVertices(gltf.scene);
          VRMUtils.combineSkeletons(gltf.scene);
          vrm.scene.traverse((obj) => { obj.frustumCulled = false; });

          this.vrm = vrm;
          this.scene.add(vrm.scene);
          this.#cacheBones();
          this.#restPose();
          this.#tameSpringBones();
          // Framing has to wait for the first real update tick: until the
          // humanoid has been applied at least once, the raw bones still report
          // their bind-pose transforms (head ~1.45 instead of ~0.47 here), and
          // the canvas may still be laid out at 0x0.
          this.needsFraming = true;

          if (vrm.lookAt) {
            this.lookTarget = new THREE.Object3D();
            this.scene.add(this.lookTarget);
            vrm.lookAt.target = this.lookTarget;
          }
          this.setState("idle");
          resolve(vrm);
        },
        (progress) => {
          if (progress.total) this.onProgress(progress.loaded / progress.total);
        },
        (error) => reject(error)
      );
    });
  }

  /**
   * Aim the camera at the model's real head, measured from the skeleton.
   *
   * Two traps here, both of which produced an apparently empty scene:
   *
   * 1. The old code hard-coded `camera.position.set(0, 1.35, 1.8)` with a
   *    target at y = 1.35, assuming a rig standing with its feet on y = 0.
   *    RNGPIT_SINA.vrm is authored centred on the origin - head at y = +0.47,
   *    feet at y = -0.79 - so the camera pointed a metre above her head.
   * 2. The obvious fix, `Box3.setFromObject`, is *also* wrong: for a
   *    SkinnedMesh three.js measures the **bind pose** geometry, not the posed
   *    result. It reports y = 0 -> 1.57 for this model, which is off by the
   *    full height of the rig.
   *
   * So we measure the posed skeleton via the raw bone nodes, and only fall
   * back to the bounding box when the model has no humanoid rig at all.
   */
  #frameCamera() {
    // World matrices are stale immediately after load.
    this.vrm.update(0);
    this.vrm.scene.updateMatrixWorld(true);

    const humanoid = this.vrm.humanoid;
    const boneWorld = (name) => {
      const bone = humanoid?.getRawBoneNode?.(name);
      return bone ? bone.getWorldPosition(new THREE.Vector3()) : null;
    };

    const head = boneWorld("head");
    const feet = ["leftToes", "rightToes", "leftFoot", "rightFoot"]
      .map(boneWorld)
      .filter(Boolean);

    let focusX = 0;
    let focusZ = 0;
    let headY;
    let groundY;

    if (head && feet.length) {
      focusX = head.x;
      focusZ = head.z;
      headY = head.y;
      groundY = Math.min(...feet.map((f) => f.y));
    } else {
      const box = new THREE.Box3().setFromObject(this.vrm.scene);
      if (box.isEmpty()) return;
      const center = box.getCenter(new THREE.Vector3());
      focusX = center.x;
      focusZ = center.z;
      groundY = box.min.y;
      headY = box.max.y - (box.max.y - box.min.y) * 0.12;
    }

    // The head bone sits at the base of the skull; add some headroom above it.
    const standing = Math.max(headY - groundY, 0.2);
    const headTop = headY + standing * 0.12;
    const totalHeight = headTop - groundY;

    // Frame head and shoulders, sitting the eyes slightly above centre.
    this.focus = new THREE.Vector3(focusX, headY - standing * 0.09, focusZ);

    const visibleHeight = Math.max(totalHeight * 0.42, 0.15);
    const fov = (this.camera.fov * Math.PI) / 180;
    const distance = visibleHeight / 2 / Math.tan(fov / 2);

    this.camera.position.set(this.focus.x, this.focus.y, this.focus.z + distance);
    this.camera.near = Math.max(0.01, distance / 100);
    this.camera.far = distance * 40;
    this.camera.updateProjectionMatrix();

    this.controls.target.copy(this.focus);
    this.controls.minDistance = distance * 0.45;
    this.controls.maxDistance = distance * 3.2;
    this.controls.update();

    // The spark field is authored for a ~1.6 m rig standing on y = 0; move and
    // scale it onto wherever this model actually is.
    if (this.particles) {
      this.particles.points.position.set(focusX, groundY, focusZ);
      this.particles.points.scale.setScalar(Math.max(totalHeight, 0.5) / 1.6);
    }
  }

  /**
   * Remove chest jiggle physics, and stop everything else bouncing on load.
   *
   * VRM models ship "secondary" spring-bone chains for hair, clothing and - in
   * this model - the bust (`J_Sec_L_Bust1/2`, `J_Sec_R_Bust1/2`). Those chest
   * springs are simply not appropriate for a college assistant, so they are
   * deleted outright rather than damped: a deleted joint can never be excited
   * by a pose change, a window resize or a dropped frame.
   *
   * The hair and skirt chains are kept - they read as natural - but the whole
   * system is re-initialised at the current pose afterwards. Without that, the
   * first simulated frame runs against the bind pose and every spring visibly
   * drops into place the moment the model appears.
   */
  #tameSpringBones() {
    const manager = this.vrm.springBoneManager;
    if (!manager) return;

    // Covers the usual naming conventions across VRM exporters.
    const CHEST = /(bust|breast|oppai|mune|chichi)/i;

    let removed = 0;
    for (const joint of Array.from(manager.joints || [])) {
      const name = joint?.bone?.name || "";
      if (!CHEST.test(name)) continue;
      try {
        manager.deleteJoint(joint);
        removed += 1;
      } catch (err) {
        // Fall back to freezing the joint if this build has no deleteJoint.
        if (joint.settings) {
          joint.settings.stiffness = 1;
          joint.settings.dragForce = 1;
          joint.settings.gravityPower = 0;
        }
        console.warn("[SINA] could not delete spring joint", name, err);
      }
    }
    if (removed) console.info(`[SINA] disabled ${removed} chest spring bones`);

    // Settle the remaining springs at the current pose so nothing swings on the
    // first frame.
    try {
      this.vrm.update(0);
      manager.setInitState();
      manager.reset();
    } catch (err) {
      console.warn("[SINA] could not reset spring bones", err);
    }
  }

  #cacheBones() {
    const get = (name) => this.vrm.humanoid?.getNormalizedBoneNode(name) || null;
    this.bones = {
      hips: get("hips"),
      spine: get("spine"),
      chest: get("chest"),
      upperChest: get("upperChest"),
      neck: get("neck"),
      head: get("head"),
      shoulderL: get("leftShoulder"),
      shoulderR: get("rightShoulder"),
      upperArmL: get("leftUpperArm"),
      upperArmR: get("rightUpperArm"),
      lowerArmL: get("leftLowerArm"),
      lowerArmR: get("rightLowerArm"),
      handL: get("leftHand"),
      handR: get("rightHand"),
      upperLegL: get("leftUpperLeg"),
      upperLegR: get("rightUpperLeg"),
    };
    const manager = this.vrm.expressionManager;
    this.hasExpression = (name) => Boolean(manager && manager.getExpression?.(name));
  }

  /** A relaxed A-pose baseline. Every animation is an offset from this. */
  #restPose() {
    const b = this.bones;
    if (b.upperArmL) b.upperArmL.rotation.set(0.05, 0, -1.22);
    if (b.upperArmR) b.upperArmR.rotation.set(0.05, 0, 1.22);
    if (b.lowerArmL) b.lowerArmL.rotation.set(0, 0, -0.2);
    if (b.lowerArmR) b.lowerArmR.rotation.set(0, 0, 0.2);
    if (b.handL) b.handL.rotation.set(0, 0, -0.08);
    if (b.handR) b.handR.rotation.set(0, 0, 0.08);
  }

  /* ------------------------------------------------------------ states */
  setState(next) {
    if (this.state === next) return;
    this.state = next;

    const posture = { idle: 0, listening: 0.35, thinking: 0.7, speaking: 0.5 }[next] ?? 0;
    this.springs.posture.target = posture;

    if (next === "thinking") {
      // Look up and away, the way people do while recalling something.
      this.saccade.x = -0.16;
      this.saccade.y = 0.3 * (Math.random() > 0.5 ? 1 : -1);
      this.saccade.next = this.time + 2.4;
    }
    if (next === "listening") {
      this.playGesture("nod", 0.55);
    }
  }

  playGesture(kind, scale = 1) {
    if (REDUCED_MOTION) return;
    this.gesture = { kind, t: 0, duration: 0.9, scale };
  }

  /* -------------------------------------------------------------- audio */
  /**
   * Play a TTS clip and drive the mouth from it.
   * Resolves when playback ends (or immediately if playback is impossible).
   */
  speak(blobUrl) {
    this.stopSpeaking();

    return new Promise((resolve) => {
      const audio = new Audio(blobUrl);
      audio.crossOrigin = "anonymous";
      this.audio = audio;

      const analysed = this.voice.attach(audio);
      if (!analysed) {
        // Without WebAudio the element still plays; the mouth uses a synthetic
        // envelope instead of real amplitude.
        audio.volume = 1;
      }

      // The resolver is held on the instance so stopSpeaking() can settle it.
      // Previously `finish()` bailed out whenever `this.audio !== audio`, so
      // interrupting playback left this promise pending forever - and the
      // caller's `finally` never ran, leaving the UI permanently "busy" and
      // silently dropping every later question until a page reload.
      let settled = false;
      let watchdog = null;
      const finish = () => {
        if (settled) return;
        settled = true;
        clearTimeout(watchdog);
        this._settleSpeech = null;
        if (this.audio === audio) {
          this.voice.detach();
          this.audio = null;
          this.setState("idle");
        }
        resolve();
      };
      this._settleSpeech = finish;

      audio.addEventListener("ended", finish);
      audio.addEventListener("error", finish);

      // Belt and braces: a clip that stalls mid-download fires neither `ended`
      // nor `error`, so cap the wait on the clip's own duration.
      const arm = () => {
        clearTimeout(watchdog);
        const duration = Number.isFinite(audio.duration) ? audio.duration : 45;
        watchdog = setTimeout(() => {
          console.warn("[SINA] audio playback stalled - releasing");
          finish();
        }, duration * 1000 * 1.4 + 6000);
      };
      audio.addEventListener("loadedmetadata", arm);
      arm();

      audio.play().then(
        () => this.setState("speaking"),
        (err) => {
          // Autoplay policy, a decode failure, or the tab being backgrounded.
          console.warn("[SINA] Audio playback blocked", err);
          finish();
        }
      );
    });
  }

  stopSpeaking() {
    const pending = this._settleSpeech;
    if (this.audio) {
      try {
        this.audio.pause();
        this.audio.currentTime = 0;
      } catch { /* nothing to stop */ }
      this.audio = null;
    }
    this.voice.detach();
    if (this.state === "speaking") this.setState("idle");
    // Release anyone awaiting speak(); without this, barge-in deadlocks the UI.
    if (pending) pending();
  }

  get isSpeaking() {
    return this.state === "speaking";
  }

  /* --------------------------------------------------------- main loop */
  start() {
    if (this.running) return;
    this.running = true;
    this.clock.getDelta();
    this.renderer.setAnimationLoop(() => this.#frame());
  }

  stop() {
    this.running = false;
    this.renderer.setAnimationLoop(null);
  }

  setVisible(visible) {
    this.visible = visible;
    if (visible) this.clock.getDelta(); // discard the gap so nothing snaps
  }

  #frame() {
    const raw = this.clock.getDelta();
    // Clamp: returning from a background tab otherwise delivers a delta of many
    // seconds and every sine-driven bone spins through dozens of cycles.
    const dt = Math.min(raw, 1 / 20);
    this.time += dt;

    // Nothing to draw into a zero-size box; the observer will wake us up.
    if (!this.visible || document.hidden || !this.sized) return;

    this.#adaptQuality(raw);
    this.#animate(dt);
    this.controls.update();
    if (this.vrm) this.vrm.update(dt);

    if (this.needsFraming && this.vrm) {
      this.needsFraming = false;
      this.#frameCamera();
    }

    this.renderer.render(this.scene, this.camera);
  }

  /** Drop the pixel ratio if the device cannot keep up; restore when it can. */
  #adaptQuality(delta) {
    if (delta <= 0) return;
    this.fpsSamples.push(1 / delta);
    if (this.fpsSamples.length < 90) return;
    const avg = this.fpsSamples.reduce((a, b) => a + b, 0) / this.fpsSamples.length;
    this.fpsSamples.length = 0;

    const max = Math.min(window.devicePixelRatio || 1, 2);
    let next = this.dpr;
    if (avg < 34 && this.dpr > 0.75) next = Math.max(0.75, this.dpr - 0.25);
    else if (avg > 57 && this.dpr < max) next = Math.min(max, this.dpr + 0.25);

    if (next !== this.dpr) {
      this.dpr = next;
      this.renderer.setPixelRatio(next);
      this.resize();
    }
  }

  /* --------------------------------------------------------- animation */
  #animate(dt) {
    this.#updateParticles(dt);
    if (!this.vrm) return;
    if (REDUCED_MOTION) {
      this.#updateBlink(dt);
      this.#applyVisemes(dt);
      return;
    }

    const t = this.time;
    const b = this.bones;
    const s = this.springs;
    const posture = s.posture.update(dt);

    // --- breathing: one cycle drives chest, shoulders and a slight body dip --
    const breathRate = this.state === "speaking" ? 1.7 : this.state === "thinking" ? 1.05 : 1.25;
    const breathPhase = Math.sin(t * breathRate);
    const breath = breathPhase * 0.016 + Math.sin(t * breathRate * 2.1) * 0.004;
    const inhale = Math.max(0, breathPhase);

    // --- weight shift: slow, asymmetric, never returns to exact centre -------
    const shift = Math.sin(t * 0.27) * 0.03 + Math.sin(t * 0.163 + 1.2) * 0.013;
    s.lean.target = shift;
    const lean = s.lean.update(dt);

    // --- gaze: pointer when present, wandering saccades otherwise -----------
    this.#updateGaze(dt);
    const gazeX = this.gaze.x;
    const gazeY = this.gaze.y;

    // --- speech energy ------------------------------------------------------
    let energy = 0;
    if (this.state === "speaking") {
      const sample = this.voice.sample();
      energy = sample
        ? sample.energy
        // Fallback envelope: layered rates so it never sounds metronomic.
        : clamp(0.34 + Math.sin(t * 11.3) * 0.24 + Math.sin(t * 6.7) * 0.16, 0, 1);
      this.#updateVisemeTargets(energy, sample ? sample.centroid : 0.5);
    } else {
      this.#updateVisemeTargets(0, 0.5);
    }
    s.energy.target = energy;
    const smoothEnergy = s.energy.update(dt);

    // --- gesture layer ------------------------------------------------------
    const g = this.#updateGesture(dt);

    // --- spine chain --------------------------------------------------------
    if (b.hips) {
      b.hips.rotation.z = lean * 0.7;
      b.hips.rotation.y = Math.sin(t * 0.31) * 0.016 + gazeY * 0.05;
      b.hips.rotation.x = -inhale * 0.006 - posture * 0.012;
      b.hips.position.y = Math.sin(t * breathRate) * 0.004;
    }
    if (b.spine) {
      b.spine.rotation.x = breath + posture * 0.05 + g.leanForward;
      b.spine.rotation.z = lean * 0.5;
      b.spine.rotation.y = s.spineTwist.update(dt) + Math.sin(t * 0.21) * 0.011;
      s.spineTwist.target = gazeY * 0.09 + g.twist;
    }
    if (b.chest) {
      b.chest.rotation.x = inhale * 0.02 + breath * 0.5 + smoothEnergy * 0.012;
      b.chest.rotation.z = lean * 0.28;
    }
    if (b.upperChest) {
      b.upperChest.rotation.x = inhale * 0.014 + breath * 0.35;
      b.upperChest.rotation.z = Math.sin(t * breathRate + 0.3) * 0.009;
    }

    // --- head / neck: the head leads, the neck follows at ~40% --------------
    s.headYaw.target = gazeY * 0.62 + g.headYaw;
    s.headPitch.target = gazeX * 0.4 + g.headPitch - posture * 0.05 + smoothEnergy * 0.02;
    s.headRoll.target =
      Math.sin(t * 0.34) * 0.02 + g.headRoll + (this.state === "thinking" ? 0.12 : 0);

    const yaw = s.headYaw.update(dt);
    const pitch = s.headPitch.update(dt);
    const roll = s.headRoll.update(dt);

    if (b.neck) {
      b.neck.rotation.y = yaw * 0.4;
      b.neck.rotation.x = pitch * 0.35;
      b.neck.rotation.z = roll * 0.35;
    }
    if (b.head) {
      b.head.rotation.y = yaw * 0.6;
      b.head.rotation.x = pitch * 0.65 + Math.sin(t * 0.83) * 0.012;
      b.head.rotation.z = roll * 0.65;
    }

    // --- shoulders + arms, with follow-through lag --------------------------
    const shoulderLift = inhale * 0.02 + smoothEnergy * 0.014;
    if (b.shoulderL) {
      b.shoulderL.rotation.z = -shoulderLift + g.shoulderL;
      b.shoulderL.rotation.x = Math.sin(t * 0.41 + 1.0) * 0.009;
    }
    if (b.shoulderR) {
      b.shoulderR.rotation.z = shoulderLift + g.shoulderR;
      b.shoulderR.rotation.x = Math.sin(t * 0.41) * 0.009;
    }

    s.armL.target = Math.sin(t * 0.49) * 0.04 + lean * 0.35 + g.armL;
    s.armR.target = Math.sin(t * 0.49 + 1.3) * 0.04 - lean * 0.35 + g.armR;
    const armL = s.armL.update(dt);
    const armR = s.armR.update(dt);

    if (b.upperArmL) {
      b.upperArmL.rotation.z = -1.22 + armL - posture * 0.05;
      b.upperArmL.rotation.x = 0.05 + Math.sin(t * 0.36 + 2.0) * 0.026 + g.armLift;
      b.upperArmL.rotation.y = Math.sin(t * 0.27 + 0.5) * 0.015;
    }
    if (b.upperArmR) {
      b.upperArmR.rotation.z = 1.22 + armR + posture * 0.05;
      b.upperArmR.rotation.x = 0.05 + Math.sin(t * 0.36 + 0.8) * 0.026 + g.armLiftR;
      b.upperArmR.rotation.y = Math.sin(t * 0.27 + 2.1) * 0.015;
    }
    // Forearms lag the upper arms - the classic secondary-motion cue.
    if (b.lowerArmL) {
      b.lowerArmL.rotation.z = -0.2 - armL * 0.5 + g.elbowL;
      b.lowerArmL.rotation.x = Math.sin(t * 0.58 + 1.4) * 0.03;
    }
    if (b.lowerArmR) {
      b.lowerArmR.rotation.z = 0.2 - armR * 0.5 + g.elbowR;
      b.lowerArmR.rotation.x = Math.sin(t * 0.58 + 2.5) * 0.03;
    }
    if (b.handL) b.handL.rotation.z = -0.08 - armL * 0.35;
    if (b.handR) b.handR.rotation.z = 0.08 - armR * 0.35;

    // --- legs take a little of the weight shift so it reads as balance ------
    if (b.upperLegL) b.upperLegL.rotation.z = lean * 0.16;
    if (b.upperLegR) b.upperLegR.rotation.z = lean * 0.16;

    this.#updateBlink(dt);
    this.#updateExpression(dt, smoothEnergy);
    this.#applyVisemes(dt);
  }

  /* ----------------------------------------------------------- gaze */
  #updateGaze(dt) {
    if (this.pointer.active && this.state !== "thinking") {
      // Track the cursor, but only within a comfortable range of head motion.
      this.saccade.x = damp(this.saccade.x, clamp(this.pointer.y * 0.3, -0.24, 0.3), 6, dt);
      this.saccade.y = damp(this.saccade.y, clamp(-this.pointer.x * 0.5, -0.5, 0.5), 6, dt);
      this.saccade.next = this.time + 1.4;
    } else if (this.time > this.saccade.next) {
      // Real gaze moves in quick jumps with uneven dwell times.
      this.saccade.x = (Math.random() - 0.5) * 0.24;
      this.saccade.y = (Math.random() - 0.5) * 0.55;
      this.saccade.next = this.time + 0.9 + Math.random() * 3.2;
      // Humans very often blink on a large gaze shift.
      if (Math.abs(this.saccade.y) > 0.3 && this.blink.phase < 0) this.blink.timer = 0.06;
    }

    // Saccades are fast, the head that follows them is slow.
    this.gaze.x = damp(this.gaze.x, this.saccade.x, 12, dt);
    this.gaze.y = damp(this.gaze.y, this.saccade.y, 12, dt);

    if (this.lookTarget && this.focus) {
      // Offsets are relative to the measured focus point, not a fixed height.
      this.lookTarget.position.set(
        this.focus.x - this.gaze.y * 1.4,
        this.focus.y + this.gaze.x * 0.9,
        this.focus.z + 2.2
      );
    }
  }

  /* ---------------------------------------------------------- blinking */
  #updateBlink(dt) {
    const bl = this.blink;
    if (bl.phase < 0) {
      bl.timer -= dt;
      if (bl.timer <= 0) {
        bl.phase = 0;
        bl.speed = 0.09 + Math.random() * 0.05;
      }
      bl.value = damp(bl.value, 0, 30, dt);
    } else {
      bl.phase += dt / bl.speed;
      if (bl.phase >= 1) {
        bl.phase = -1;
        bl.value = 0;
        if (bl.queued > 0) {
          // Double blinks are common and make the face feel alive.
          bl.queued -= 1;
          bl.timer = 0.09;
        } else {
          bl.timer = 1.8 + Math.random() * 4.2;
          if (Math.random() < 0.22) bl.queued = 1;
        }
      } else {
        // Closing is faster than opening, as with a real eyelid.
        bl.value = bl.phase < 0.4
          ? bl.phase / 0.4
          : 1 - (bl.phase - 0.4) / 0.6;
      }
    }
    this.#setExpression("blink", clamp(bl.value, 0, 1));
  }

  /* -------------------------------------------------------- expression */
  #updateExpression(dt, energy) {
    const target = { happy: 0.06, relaxed: 0.1, surprised: 0 };
    if (this.state === "speaking") { target.happy = 0.14 + energy * 0.1; target.relaxed = 0.04; }
    if (this.state === "listening") { target.happy = 0.16; target.relaxed = 0.14; }
    if (this.state === "thinking") { target.happy = 0.02; target.relaxed = 0.2; }

    for (const key of ["happy", "relaxed", "surprised"]) {
      this.expression[key] = damp(this.expression[key], target[key], 4.5, dt);
      this.#setExpression(key, this.expression[key]);
    }
  }

  /* ------------------------------------------------------------ visemes */
  /**
   * Map loudness + spectral balance onto a blend of the five VRM visemes.
   *
   * Amplitude matters as much as shape here. An earlier version scaled the
   * result three separate times (a vowel-shape factor, a 0.9 trim, then another
   * `0.4 + jaw * 0.6` in the apply step) so the mouth peaked at 0.38 and
   * averaged 0.15 on real speech - technically animating, visually a twitch.
   * The weights below are normalised against the *dominant* viseme, so the
   * strongest shape tracks loudness one-to-one and the others blend around it.
   */
  #updateVisemeTargets(energy, centroid) {
    // Noise gate: close the mouth between words instead of buzzing at silence.
    if (energy < 0.06) {
      this.visemeTargets = { aa: 0, ih: 0, ou: 0, ee: 0, oh: 0 };
      this.springs.jaw.target = 0;
      return;
    }

    const open = clamp(energy * 1.15, 0, 1);
    this.springs.jaw.target = open;

    // Spectral centroid approximates vowel frontness: low -> rounded back
    // vowels (ou/oh), high -> spread front vowels (ih/ee).
    const front = clamp((centroid - 0.25) / 0.45, 0, 1);
    const at = (position, width) => Math.max(0, 1 - Math.abs(front - position) / width);
    const weights = {
      ou: at(0.0, 0.42),
      oh: at(0.28, 0.4),
      aa: at(0.52, 0.46),
      ih: at(0.76, 0.4),
      ee: at(1.0, 0.42),
    };

    const peak = Math.max(weights.ou, weights.oh, weights.aa, weights.ih, weights.ee) || 1;
    this.visemeTargets = {
      ou: (weights.ou / peak) * open,
      oh: (weights.oh / peak) * open,
      aa: (weights.aa / peak) * open,
      ih: (weights.ih / peak) * open,
      ee: (weights.ee / peak) * open,
    };
  }

  #applyVisemes(dt) {
    const targets = this.visemeTargets || { aa: 0, ih: 0, ou: 0, ee: 0, oh: 0 };
    this.springs.jaw.update(dt);
    for (const key of ["aa", "ih", "ou", "ee", "oh"]) {
      // Fast damping: slower than this and speech looks dubbed.
      this.visemes[key] = damp(this.visemes[key], targets[key], 32, dt);
      this.#setExpression(key, clamp(this.visemes[key], 0, 1));
    }
  }

  #setExpression(name, value) {
    const manager = this.vrm?.expressionManager;
    if (!manager) return;
    try {
      manager.setValue(name, value);
    } catch { /* this model does not define that expression */ }
  }

  /* ----------------------------------------------------------- gestures */
  #updateGesture(dt) {
    const zero = {
      headYaw: 0, headPitch: 0, headRoll: 0, twist: 0, leanForward: 0,
      armL: 0, armR: 0, armLift: 0, armLiftR: 0, elbowL: 0, elbowR: 0,
      shoulderL: 0, shoulderR: 0,
    };
    if (!this.gesture) return zero;

    const g = this.gesture;
    g.t += dt;
    const p = clamp(g.t / g.duration, 0, 1);
    // Ease in and out so gestures never pop.
    const blend = Math.sin(p * Math.PI) * g.scale;
    if (p >= 1) this.gesture = null;

    if (g.kind === "nod") {
      return { ...zero, headPitch: Math.sin(p * TAU) * 0.17 * g.scale };
    }
    if (g.kind === "tilt") {
      return { ...zero, headRoll: blend * 0.2, headYaw: blend * 0.1 };
    }
    if (g.kind === "beat") {
      // A small conversational hand beat used while speaking.
      return {
        ...zero,
        armR: -0.22 * blend,
        elbowR: -0.3 * blend,
        headPitch: Math.sin(p * TAU) * 0.05 * g.scale,
      };
    }
    return zero;
  }

  /* ---------------------------------------------------------- particles */
  #updateParticles(dt) {
    const p = this.particles;
    if (!p) return;
    for (let i = 0; i < p.count; i++) {
      const y = i * 3 + 1;
      p.positions[y] += p.velocities[i] * dt;
      p.positions[i * 3] += Math.sin(this.time * 0.6 + p.phases[i]) * 0.0012;
      if (p.positions[y] > 2.9) {
        p.positions[i * 3] = (Math.random() - 0.5) * 3.2;
        p.positions[y] = -0.7;
        p.positions[i * 3 + 2] = (Math.random() - 0.5) * 1.8;
      }
    }
    p.geometry.attributes.position.needsUpdate = true;
  }

  /* ------------------------------------------------------------ cleanup */
  dispose() {
    this.stop();
    this.stopSpeaking();
    this.resizeObserver?.disconnect();
    this.controls.dispose();
    if (this.vrm) {
      VRMUtils.deepDispose(this.vrm.scene);
      this.scene.remove(this.vrm.scene);
      this.vrm = null;
    }
    if (this.particles) {
      this.particles.geometry.dispose();
      this.particles.points.material.dispose();
    }
    this.renderer.dispose();
  }
}

/* ==========================================================================
   Chat-tab wave background
   Shares the avatar's animation loop instead of running a fourth RAF loop, and
   only updates its geometry while the chat tab is actually on screen.
   ========================================================================== */
export class WaveBackground {
  constructor(container) {
    this.container = container;
    this.visible = false;
    this.time = 0;

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(72, 1, 0.1, 100);
    this.camera.position.z = 5;

    this.renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true, powerPreference: "low-power" });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
    container.appendChild(this.renderer.domElement);
    Object.assign(this.renderer.domElement.style, {
      position: "absolute", inset: "0", width: "100%", height: "100%",
    });

    const segments = REDUCED_MOTION ? 20 : 44;
    this.geometry = new THREE.PlaneGeometry(20, 20, segments, segments);
    // Kept deliberately faint: this sits behind the welcome copy and the
    // message list, and must never compete with them for attention.
    this.material = new THREE.MeshBasicMaterial({
      color: 0xa29bfe, wireframe: true, transparent: true, opacity: 0.12,
    });
    this.mesh = new THREE.Mesh(this.geometry, this.material);
    this.mesh.rotation.x = -Math.PI / 3;
    this.scene.add(this.mesh);
    this.sized = false;

    if (typeof ResizeObserver !== "undefined") {
      this.resizeObserver = new ResizeObserver(() => this.resize());
      this.resizeObserver.observe(container);
    } else {
      window.addEventListener("resize", () => this.resize());
    }

    // Cache the base grid so the per-frame loop only writes Z.
    const pos = this.geometry.attributes.position;
    this.baseX = new Float32Array(pos.count);
    this.baseY = new Float32Array(pos.count);
    for (let i = 0; i < pos.count; i++) {
      this.baseX[i] = pos.getX(i);
      this.baseY[i] = pos.getY(i);
    }
    this.resize();
  }

  setColor(hex) { this.material.color.setHex(hex); }

  setVisible(visible) {
    this.visible = visible;
    this.container.style.display = visible ? "block" : "none";
  }

  resize() {
    const rect = this.container.getBoundingClientRect();
    const width = Math.floor(rect.width);
    const height = Math.floor(rect.height);
    if (!width || !height) {
      this.sized = false;
      return;
    }
    this.sized = true;
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height, false);
  }

  update(dt) {
    if (!this.visible || document.hidden || !this.sized) return;
    this.time += dt;
    if (!REDUCED_MOTION) {
      const pos = this.geometry.attributes.position;
      const t = this.time;
      for (let i = 0; i < pos.count; i++) {
        pos.setZ(i, Math.sin(this.baseX[i] * 0.5 + t) * Math.cos(this.baseY[i] * 0.5 + t) * 0.5);
      }
      pos.needsUpdate = true;
      this.mesh.rotation.z += dt * 0.06;
    }
    this.renderer.render(this.scene, this.camera);
  }

  dispose() {
    this.resizeObserver?.disconnect();
    this.geometry.dispose();
    this.material.dispose();
    this.renderer.dispose();
  }
}
