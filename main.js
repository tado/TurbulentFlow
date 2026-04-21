import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// ========== 3D Simplex Noise ==========
// Stefan Gustavson のアルゴリズムに基づく実装

const F3 = 1.0 / 3.0;
const G3 = 1.0 / 6.0;

// 12方向のグラジェントベクトル
const GRAD3 = new Float32Array([
    1, 1, 0, -1, 1, 0, 1, -1, 0, -1, -1, 0,
    1, 0, 1, -1, 0, 1, 1, 0, -1, -1, 0, -1,
    0, 1, 1, 0, -1, 1, 0, 1, -1, 0, -1, -1
]);

// ランダムな順列テーブルを生成
const { PERM, PERM_MOD12 } = (() => {
    const p = new Uint8Array(256);
    for (let i = 0; i < 256; i++) p[i] = i;
    for (let i = 255; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        const tmp = p[i]; p[i] = p[j]; p[j] = tmp;
    }
    const PERM = new Uint8Array(512);
    const PERM_MOD12 = new Uint8Array(512);
    for (let i = 0; i < 512; i++) {
        PERM[i] = p[i & 255];
        PERM_MOD12[i] = PERM[i] % 12;
    }
    return { PERM, PERM_MOD12 };
})();

function noise3D(xin, yin, zin) {
    const s = (xin + yin + zin) * F3;
    const i = Math.floor(xin + s);
    const j = Math.floor(yin + s);
    const k = Math.floor(zin + s);
    const t = (i + j + k) * G3;
    const x0 = xin - i + t, y0 = yin - j + t, z0 = zin - k + t;

    let i1, j1, k1, i2, j2, k2;
    if (x0 >= y0) {
        if (y0 >= z0) { i1 = 1; j1 = 0; k1 = 0; i2 = 1; j2 = 1; k2 = 0; }
        else if (x0 >= z0) { i1 = 1; j1 = 0; k1 = 0; i2 = 1; j2 = 0; k2 = 1; }
        else { i1 = 0; j1 = 0; k1 = 1; i2 = 1; j2 = 0; k2 = 1; }
    } else {
        if (y0 < z0) { i1 = 0; j1 = 0; k1 = 1; i2 = 0; j2 = 1; k2 = 1; }
        else if (x0 < z0) { i1 = 0; j1 = 1; k1 = 0; i2 = 0; j2 = 1; k2 = 1; }
        else { i1 = 0; j1 = 1; k1 = 0; i2 = 1; j2 = 1; k2 = 0; }
    }

    const x1 = x0 - i1 + G3, y1 = y0 - j1 + G3, z1 = z0 - k1 + G3;
    const x2 = x0 - i2 + 2 * G3, y2 = y0 - j2 + 2 * G3, z2 = z0 - k2 + 2 * G3;
    const x3 = x0 - 1 + 3 * G3, y3 = y0 - 1 + 3 * G3, z3 = z0 - 1 + 3 * G3;

    const ii = i & 255, jj = j & 255, kk = k & 255;
    const g0 = PERM_MOD12[ii + PERM[jj + PERM[kk]]] * 3;
    const g1 = PERM_MOD12[ii + i1 + PERM[jj + j1 + PERM[kk + k1]]] * 3;
    const g2 = PERM_MOD12[ii + i2 + PERM[jj + j2 + PERM[kk + k2]]] * 3;
    const g3 = PERM_MOD12[ii + 1 + PERM[jj + 1 + PERM[kk + 1]]] * 3;

    let n0 = 0, n1 = 0, n2 = 0, n3 = 0, tt;
    tt = 0.6 - x0 * x0 - y0 * y0 - z0 * z0; if (tt > 0) { tt *= tt; n0 = tt * tt * (GRAD3[g0] * x0 + GRAD3[g0 + 1] * y0 + GRAD3[g0 + 2] * z0); }
    tt = 0.6 - x1 * x1 - y1 * y1 - z1 * z1; if (tt > 0) { tt *= tt; n1 = tt * tt * (GRAD3[g1] * x1 + GRAD3[g1 + 1] * y1 + GRAD3[g1 + 2] * z1); }
    tt = 0.6 - x2 * x2 - y2 * y2 - z2 * z2; if (tt > 0) { tt *= tt; n2 = tt * tt * (GRAD3[g2] * x2 + GRAD3[g2 + 1] * y2 + GRAD3[g2 + 2] * z2); }
    tt = 0.6 - x3 * x3 - y3 * y3 - z3 * z3; if (tt > 0) { tt *= tt; n3 = tt * tt * (GRAD3[g3] * x3 + GRAD3[g3 + 1] * y3 + GRAD3[g3 + 2] * z3); }

    return 32 * (n0 + n1 + n2 + n3);
}

// ========== Curl Noise ==========
// ポテンシャル場 F=(Fx,Fy,Fz) のカールを有限差分で計算
// curl(F) = (dFz/dy - dFy/dz, dFx/dz - dFz/dx, dFy/dx - dFx/dy)
// 3成分に独立性を持たせるためオフセットを使用

const EPS = 0.001;
const OA = 31.41592; // ポテンシャル成分を独立させるオフセット定数
const OB = 27.18281;

// Fx = noise3D(x,       y+OA, z+OB)
// Fy = noise3D(x+OB,    y,    z+OA)
// Fz = noise3D(x+OA,    y+OB, z   )

const _curl = new Float32Array(3);

function curlNoise(x, y, z) {
    const inv2e = 0.5 / EPS;

    // curl_x = dFz/dy - dFy/dz
    _curl[0] = (
        noise3D(x + OA, y + OB + EPS, z) - noise3D(x + OA, y + OB - EPS, z) -
        noise3D(x + OB, y, z + OA + EPS) + noise3D(x + OB, y, z + OA - EPS)
    ) * inv2e;

    // curl_y = dFx/dz - dFz/dx
    _curl[1] = (
        noise3D(x, y + OA, z + OB + EPS) - noise3D(x, y + OA, z + OB - EPS) -
        noise3D(x + OA + EPS, y + OB, z) + noise3D(x + OA - EPS, y + OB, z)
    ) * inv2e;

    // curl_z = dFy/dx - dFx/dy
    _curl[2] = (
        noise3D(x + OB + EPS, y, z + OA) - noise3D(x + OB - EPS, y, z + OA) -
        noise3D(x, y + OA + EPS, z + OB) + noise3D(x, y + OA - EPS, z + OB)
    ) * inv2e;

    return _curl;
}

// ========== シーン設定 ==========
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x050510);

// カメラ
const camera = new THREE.PerspectiveCamera(100, window.innerWidth / window.innerHeight, 0.1, 100);
camera.position.set(0, 0, 2.0);

// レンダラー
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
document.body.appendChild(renderer.domElement);

// ========== 左上ステータス表示 ==========
const statsEl = document.createElement('div');
Object.assign(statsEl.style, {
    position: 'fixed', top: '12px', left: '12px',
    color: 'rgba(255,255,255,0.75)', fontFamily: 'monospace',
    fontSize: '13px', lineHeight: '1.6', pointerEvents: 'none',
    textShadow: '0 1px 3px rgba(0,0,0,0.8)',
});
document.body.appendChild(statsEl);

// オービットコントロール
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;

// ========== パーティクルシステム ==========
// 平衡数 = SPAWN_RATE × avg(LIFETIME) ≒ 87 × 700 = 60,900
const PARTICLE_COUNT = 65000;
const NOISE_SCALE = 1.4;   // ノイズの空間スケール
const SPEED = 0.0015; // パーティクルの移動速度
const LIFETIME_MIN = 400;    // 寿命の最小フレーム数
const LIFETIME_MAX = 1000;   // 寿命の最大フレーム数
const SPAWN_RATE = 43;     // 1フレームあたりの発生数
const SPAWN_SPREAD = 0.04;   // 発生位置の初期ランダムばらつき半径
const HIDDEN_POS = 9999;   // 非アクティブ時の退避座標

const positions = new Float32Array(PARTICLE_COUNT * 3);
const colors = new Float32Array(PARTICLE_COUNT * 3);
const ages = new Uint16Array(PARTICLE_COUNT);   // 経過フレーム数
const lifetimes = new Uint16Array(PARTICLE_COUNT);   // 最大寿命

// アクティブリスト: 生存中スロットのインデックスを密に管理
// デッドスタック:   空きスロットをO(1)で取り出す
const activeList = new Int32Array(PARTICLE_COUNT);
const deadStack = new Int32Array(PARTICLE_COUNT);
let activeCount = 0;
let deadTop = PARTICLE_COUNT;

// 全スロットを初期状態でデッドスタックに積む
for (let i = 0; i < PARTICLE_COUNT; i++) {
    deadStack[i] = i;
    positions[i * 3 + 1] = HIDDEN_POS;
}

// O(1) 発生: デッドスタックからスロットを取り出してアクティブリストに追加
function spawnParticle() {
    if (deadTop === 0) return;
    const i = deadStack[--deadTop];
    const r = Math.random() * SPAWN_SPREAD;
    const theta = Math.random() * Math.PI * 2;
    const phi = Math.acos(2 * Math.random() - 1);
    positions[i * 3] = r * Math.sin(phi) * Math.cos(theta);
    positions[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
    positions[i * 3 + 2] = r * Math.cos(phi);
    ages[i] = 0;
    lifetimes[i] = LIFETIME_MIN + Math.random() * (LIFETIME_MAX - LIFETIME_MIN);
    activeList[activeCount++] = i;
}

// O(1) 消去: アクティブリストの末尾と交換してから縮小 (swap-and-pop)
function killAt(li) {
    const i = activeList[li];
    positions[i * 3 + 1] = HIDDEN_POS;
    deadStack[deadTop++] = i;
    activeList[li] = activeList[--activeCount];
}

const geometry = new THREE.BufferGeometry();
geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

// 円形パーティクル用テクスチャを Canvas で生成
const circleTexture = (() => {
    const size = 64;
    const canvas = document.createElement('canvas');
    canvas.width = canvas.height = size;
    const ctx = canvas.getContext('2d');
    const grad = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
    grad.addColorStop(0, 'rgba(255,255,255,1)');
    grad.addColorStop(0.4, 'rgba(255,255,255,0.8)');
    grad.addColorStop(1, 'rgba(255,255,255,0)');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, size, size);
    return new THREE.CanvasTexture(canvas);
})();

// 加算合成で輝くような見た目に
const material = new THREE.PointsMaterial({
    size: 0.02,
    map: circleTexture,
    alphaTest: 0.01,
    vertexColors: true,
    transparent: true,
    opacity: 0.9,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
    sizeAttenuation: true,
});

// ピボットグループ — パーティクル全体をまとめてゆっくり回転させる
const pivot = new THREE.Group();
pivot.add(new THREE.Points(geometry, material));
scene.add(pivot);

// ========== 動画キャプチャ ==========
// captureStream(0) + requestFrame() 方式:
// ブラウザ側でフレームを自動サンプリングせず、renderer.render() 後に手動で通知する。
// これにより実フレームレートに関わらず描画した全フレームが欠落なく記録される。
let mediaRecorder = null;
let recordedChunks = [];
let videoTrack = null; // requestFrame() 呼び出し用トラック

function startRecording() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') mediaRecorder.stop();
    recordedChunks = [];
    const stream = renderer.domElement.captureStream(0); // 0 = 手動フレーム通知モード
    videoTrack = stream.getVideoTracks()[0];
    const mimeType = MediaRecorder.isTypeSupported('video/webm;codecs=vp9')
        ? 'video/webm;codecs=vp9'
        : 'video/webm';
    mediaRecorder = new MediaRecorder(stream, { mimeType });
    mediaRecorder.ondataavailable = e => { if (e.data.size > 0) recordedChunks.push(e.data); };
    mediaRecorder.start();
    console.log('録画開始');
}

function stopRecording() {
    if (!mediaRecorder || mediaRecorder.state === 'inactive') return;
    mediaRecorder.onstop = () => {
        const blob = new Blob(recordedChunks, { type: 'video/webm' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `particles_${Date.now()}.webm`;
        a.click();
        URL.revokeObjectURL(url);
        console.log('録画保存完了');
    };
    mediaRecorder.stop();
}

// ========== キーボード操作 ==========
// r: パーティクルリセット + 録画開始 / s: 録画停止 + ファイル保存
window.addEventListener('keydown', e => {
    if (e.key === 'r' || e.key === 'R') {
        // 全アクティブパーティクルをデッドスタックに戻す
        while (activeCount > 0) killAt(0);
        startRecording();
    } else if (e.key === 's' || e.key === 'S') {
        stopRecording();
    }
});

// ========== ウィンドウリサイズ対応 ==========
window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});

// ========== アニメーションループ ==========
let time = 0;
let lastTimestamp = performance.now();
let fps = 0;

function animate() {
    requestAnimationFrame(animate);

    // FPS 計算
    const now = performance.now();
    fps = Math.round(1000 / (now - lastTimestamp));
    lastTimestamp = now;

    // ステータス更新
    statsEl.textContent = `FPS: ${fps}\nParticles: ${activeCount.toLocaleString()}`;

    time += 0.004; // 時間の進み速度 (ノイズ場がゆっくり変化する)

    // --- 中心からパーティクルを一定数ずつ発生 (O(1)/個) ---
    for (let s = 0; s < SPAWN_RATE; s++) spawnParticle();

    // --- アクティブなパーティクルのみ更新 (非アクティブスロットは一切触れない) ---
    let li = 0;
    while (li < activeCount) {
        const i = activeList[li];
        const age = ++ages[i];

        // 寿命に達したら O(1) で消去 (swap-and-pop のため li はインクリメントしない)
        if (age >= lifetimes[i]) {
            killAt(li);
            continue;
        }

        const idx = i * 3;
        const x = positions[idx], y = positions[idx + 1], z = positions[idx + 2];

        // Curl Noise によって発散ゼロの速度ベクトルを取得
        const v = curlNoise(x * NOISE_SCALE, y * NOISE_SCALE, z * NOISE_SCALE + time);

        positions[idx] = x + v[0] * SPEED;
        positions[idx + 1] = y + v[1] * SPEED;
        positions[idx + 2] = z + v[2] * SPEED;

        // 速度の大きさに応じて青→シアン→白にグラデーション
        const spd = Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        const c = Math.min(spd * 1.5, 1.0);
        colors[idx] = c * 0.25;
        colors[idx + 1] = c * 0.7 + 0.1;
        colors[idx + 2] = 1.0;
        li++;
    }

    geometry.attributes.position.needsUpdate = true;
    geometry.attributes.color.needsUpdate = true;

    // 全体をx・y・z軸それぞれ異なる速度でゆっくり回転
    pivot.rotation.x += 0.001;
    pivot.rotation.y += 0.0000;
    pivot.rotation.z += 0.0010;

    controls.update();
    renderer.render(scene, camera);

    // 録画中は render 直後にフレームを明示的に通知 → コマ落ちなし
    if (videoTrack) videoTrack.requestFrame();
}

animate();