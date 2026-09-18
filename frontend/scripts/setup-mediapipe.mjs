// Provisions the model assets that ship too large/binary for `public/` to be
// tracked in git (the whole directory is gitignored): the MediaPipe WASM
// vision runtime (copied from the installed npm package), the pretrained
// gesture recognizer model (downloaded once from Google's model storage),
// and the self-trained gesture MLP's weights (copied from ml/experiments/,
// which IS committed — see ml/train.py). Runs automatically via `npm
// install`'s postinstall hook.

import { existsSync, mkdirSync, copyFileSync, readdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const root = path.dirname(fileURLToPath(import.meta.url)) + '/..';

const wasmSrcDir = path.join(root, 'node_modules/@mediapipe/tasks-vision/wasm');
const wasmDestDir = path.join(root, 'public/mediapipe-wasm');
const modelDestPath = path.join(root, 'public/models/gesture_recognizer.task');
const modelUrl =
  'https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task';

const trainedWeightsSrcPath = path.join(root, '../ml/experiments/gesture_mlp_weights.json');
const trainedWeightsDestPath = path.join(root, 'public/models/gesture_mlp_weights.json');

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

function copyTrainedGestureWeights() {
  if (!existsSync(trainedWeightsSrcPath)) {
    console.warn(
      '[setup-mediapipe] No ml/experiments/gesture_mlp_weights.json yet — run `uv run python -m ml.train` first.'
    );
    return;
  }
  mkdirSync(path.dirname(trainedWeightsDestPath), { recursive: true });
  copyFileSync(trainedWeightsSrcPath, trainedWeightsDestPath);
  console.log('[setup-mediapipe] Copied trained gesture MLP weights to public/models/');
}

copyWasmRuntime();
await downloadModel();
copyTrainedGestureWeights();
