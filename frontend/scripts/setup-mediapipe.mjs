// Provisions the MediaPipe assets that ship too large for git: the WASM
// vision runtime (copied straight out of the installed npm package) and the
// pretrained gesture recognizer model (downloaded once from Google's model
// storage). Runs automatically via `npm install`'s postinstall hook.

import { existsSync, mkdirSync, copyFileSync, readdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const root = path.dirname(fileURLToPath(import.meta.url)) + '/..';

const wasmSrcDir = path.join(root, 'node_modules/@mediapipe/tasks-vision/wasm');
const wasmDestDir = path.join(root, 'public/mediapipe-wasm');
const modelDestPath = path.join(root, 'public/models/gesture_recognizer.task');
const modelUrl =
  'https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task';

function copyWasmRuntime() {
  if (!existsSync(wasmSrcDir)) {
    console.warn('[setup-mediapipe] @mediapipe/tasks-vision not installed yet, skipping wasm copy.');
    return;
  }
  mkdirSync(wasmDestDir, { recursive: true });
  for (const file of readdirSync(wasmSrcDir)) {
    if (file.endsWith('.wasm') || file.endsWith('.js')) {
      copyFileSync(path.join(wasmSrcDir, file), path.join(wasmDestDir, file));
    }
  }
  console.log('[setup-mediapipe] Copied WASM vision runtime to public/mediapipe-wasm/');
}

async function downloadModel() {
  if (existsSync(modelDestPath)) {
    console.log('[setup-mediapipe] Gesture recognizer model already present, skipping download.');
    return;
  }
  mkdirSync(path.dirname(modelDestPath), { recursive: true });
  console.log('[setup-mediapipe] Downloading pretrained gesture recognizer model...');
  const res = await fetch(modelUrl);
  if (!res.ok) {
    console.error(`[setup-mediapipe] Failed to download model: HTTP ${res.status}`);
    process.exitCode = 1;
    return;
  }
  const buffer = Buffer.from(await res.arrayBuffer());
  const { writeFileSync } = await import('node:fs');
  writeFileSync(modelDestPath, buffer);
  console.log(`[setup-mediapipe] Saved model to public/models/gesture_recognizer.task (${buffer.length} bytes)`);
}

copyWasmRuntime();
await downloadModel();
