# whisper.cpp server on the GPU box

These instructions stand up the STT side of the omilog architecture: a
`whisper-server` process running alongside the existing `llama-server` on the
GPU host, exposing an HTTP `/inference` endpoint that the Pi will call to
transcribe `.opus` files.

Assumed setup (matches the omilog spec):

- NVIDIA GPU with CUDA already installed (you already run `llama.cpp` here)
- Ubuntu / Debian Linux
- Tailscale running, the machine reachable as something like `gpu-host.tailnet`
- Pi reaches the GPU box over the tailnet only — **never expose this to the
  public internet**

If your GPU is AMD / Intel / Apple Silicon, swap the CUDA build flag at the
build step; everything else is identical.

## 1. Build

```bash
sudo apt install -y build-essential cmake git
cd ~/src      # or wherever you keep source
git clone https://github.com/ggerganov/whisper.cpp.git
cd whisper.cpp

# CUDA build (NVIDIA). For AMD use GGML_HIPBLAS=1; for Apple Silicon, drop
# the CUDA flag — Metal builds by default.
cmake -B build -DGGML_CUDA=1
cmake --build build --config Release -j

# Verify the server binary exists.
./build/bin/whisper-server --help | head -20
```

If the build fails with a CUDA-related error, your CUDA toolkit version
probably doesn't match the cmake version expected by `ggml-cuda`. Update
`cmake` (`pip install -U cmake` or `snap install cmake --classic`) and retry.

## 2. Pick a model — French-capable

You want **`large-v3-turbo`**. Reasoning:

| Model | Size (ggml q5_0) | Speed vs real-time on a 4090 | French quality |
| --- | --- | --- | --- |
| `tiny` / `base` / `small` | 40 MB – 250 MB | very fast | poor — not usable for French speech |
| `medium` | ~530 MB | ~10× | OK, noticeable errors on accented speech |
| `large-v3` | ~1.1 GB | ~3-4× | best multilingual, but slower |
| **`large-v3-turbo`** | ~530 MB (q5_0) / ~810 MB (f16) | ~8-10× | ~95% of large-v3 on French, much faster |
| `distil-large-v3` | 600 MB | 12× | **English only — skip** |

For passive personal capture in French, `large-v3-turbo` is the sweet spot.
The q5_0 quantization gives you a real-time-on-consumer-GPU footprint with
quality essentially indistinguishable from the f16 version for speech.

Download (one of these — pick quantized for smaller VRAM, f16 if you have
margin):

```bash
mkdir -p models
cd models

# Quantized q5_0 — recommended (small + fast + good)
curl -L -o ggml-large-v3-turbo-q5_0.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q5_0.bin

# Or the full f16 if you don't mind the extra ~280 MB
# curl -L -o ggml-large-v3-turbo.bin \
#   https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin

cd ..
```

Quick sanity check (transcribes the built-in JFK sample):

```bash
./build/bin/whisper-cli \
  -m models/ggml-large-v3-turbo-q5_0.bin \
  -f samples/jfk.wav -l en
```

For French specifically, drop in any French `.wav` and pass `-l fr` or `-l auto`.

## 3. Run the server

The server speaks HTTP and exposes `/inference` (multipart upload, returns
JSON with `text` and `segments`).

```bash
TAILSCALE_IP=$(tailscale ip -4 | head -1)

./build/bin/whisper-server \
  --model models/ggml-large-v3-turbo-q5_0.bin \
  --host $TAILSCALE_IP \
  --port 8080 \
  --inference-path /inference \
  --language auto \
  --threads 4 \
  --processors 1
```

Notes:

- `--host $TAILSCALE_IP` binds only to the tailnet interface. **Do not use
  `0.0.0.0` unless you're sure your firewall blocks 8080 elsewhere.**
- `--language auto` lets Whisper detect; if 100% of your input is French and
  detection is borderline, force it with `--language fr`.
- `--threads 4` is for CPU pre/post-processing; the heavy lifting is on the
  GPU. Bump to your physical core count for marginal wins.
- `--processors 1` keeps memory predictable; multi-processor only helps with
  parallel files, which we don't have.

You'll see something like:

```
whisper_init_from_file_with_params_no_state: loading model from ...
whisper_model_load: ...
whisper server listening at http://100.x.x.x:8080
```

## 4. Verify from the Pi

```bash
# On the Pi (also tailnet-joined):
curl -F "file=@/some/test.wav" \
     -F "language=fr" \
     -F "response_format=verbose_json" \
     http://gpu-host.tailnet:8080/inference
```

Expected response (truncated):

```json
{
  "text": "Bonjour, ceci est un test.",
  "segments": [
    {"start": 0.0, "end": 1.3, "text": "Bonjour, ceci est un test."}
  ],
  "language": "fr"
}
```

If you get a connection refused → check `tailscale ping gpu-host.tailnet` from
the Pi. If you get HTTP 400 with "unsupported audio format" → whisper-server
needs WAV (16-bit PCM); on the omilog side we'll feed it through `ffmpeg` from
the Ogg-Opus capture before posting.

## 5. Keep it running (systemd)

Save as `/etc/systemd/system/whisper-server.service` on the GPU box:

```ini
[Unit]
Description=whisper.cpp HTTP inference server
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=YOUR_USERNAME
WorkingDirectory=/home/YOUR_USERNAME/src/whisper.cpp
ExecStart=/home/YOUR_USERNAME/src/whisper.cpp/build/bin/whisper-server \
  --model /home/YOUR_USERNAME/src/whisper.cpp/models/ggml-large-v3-turbo-q5_0.bin \
  --host 100.x.x.x \
  --port 8080 \
  --inference-path /inference \
  --language auto \
  --threads 4
Restart=on-failure
RestartSec=5

# Optional: only start once your tailscale interface is up.
After=tailscaled.service
Wants=tailscaled.service

[Install]
WantedBy=multi-user.target
```

Replace `100.x.x.x` with `tailscale ip -4` output. Then:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now whisper-server
sudo systemctl status whisper-server
journalctl -u whisper-server -f
```

## 6. Hook into omilog

Once the server responds, add these to `.env` on the **Pi**:

```
OMILOG_STT_BASE_URL=http://gpu-host.tailnet:8080
OMILOG_STT_LANGUAGE=fr
OMILOG_STT_INFERENCE_PATH=/inference
```

(Settings module needs adding — coming in the next push alongside the
pipeline runner.)

## Troubleshooting

| Symptom | Likely cause |
| --- | --- |
| `cudaErrorNoDevice` at startup | NVIDIA driver too old, or container without `--gpus all`. Run `nvidia-smi`. |
| "Out of memory" loading model | Quantize down (`q5_0` → `q4_0`) or switch to `medium` |
| Detection picks `en` for French speech | Force `--language fr` |
| 50 % of words missing on French | Probably accidentally loaded `distil-large-v3` (English only). `ls -la models/` and recheck filename. |
| Throughput < 1× real-time | Verify CUDA is actually used: the startup log mentions `ggml_init_cublas: found N CUDA devices`. If it shows 0, the build wasn't CUDA-enabled. |
| HTTP request hangs forever | `whisper-server` is single-threaded for inference — only one request at a time. Concurrent calls queue. For omilog that's fine (single user). |

## When to revisit

- **Speaker diarization (Phase 3 spec item):** whisper.cpp doesn't do this. If
  you decide you want "this voice is Alice / Bob," add `pyannote-audio` as a
  sidecar service. Out of scope for now.
- **Streaming STT:** `whisper-server` is request/response. For near-real-time
  you'd want a streaming endpoint (`whisper.cpp/examples/stream`). Not worth
  it until Phase 2 + UI exists.
- **Custom French model:** Whisper's vanilla `large-v3-turbo` is genuinely
  good at French. Don't bother with fine-tuning unless you have a specific
  vocabulary (medical, legal, technical jargon) the base model mangles.
