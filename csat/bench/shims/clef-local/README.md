# Clef local runtime

Runs the official public Clef release without paid API calls. All inference stays
on the selected host; the HTTP server binds only to `127.0.0.1`. Model downloads
read the public Hugging Face repository without a token.

`serve.py` imports the exact release's `joint_schema_model.py`. The tokenizer,
record encoding, backbone, joint schema head, forward pass, and SystemOne
probability rounding are unchanged. No quantization, prompt edits, option
rewrites, few-shot examples, or alternate pretrained Qwen weights are used.
The server pre-encodes the full request, then returns HTTP 413 if it exceeds the
configured 16,384-token limit. This prevents the official encoder's otherwise
silent state truncation from altering benchmark inputs.

The server serializes inference (batch size 1). `GET /health` stays responsive
during inference and reports actual parameter dtypes, device, pinned revision,
library versions, official-code SHA256, request count, and current memory use.
`POST /v1/systemone` accepts the official body and adds execution `metadata` to
the official response. It only accepts text records for this benchmark.

## Mac Studio runtime

The dedicated environment and a single copy of weights are stored outside the
repository and existing benchmark environments:

```sh
uv venv --python 3.12 /Users/chenjing/.local/share/jev-clef/venv
uv pip install --python /Users/chenjing/.local/share/jev-clef/venv/bin/python \
  -r csat/bench/shims/clef-local/requirements.lock.txt
/Users/chenjing/.local/share/jev-clef/venv/bin/python \
  csat/bench/shims/clef-local/download.py \
  --directory /Users/chenjing/.local/share/jev-clef/models/clef-flash-17f0b0a
/Users/chenjing/.local/share/jev-clef/venv/bin/python \
  csat/bench/shims/clef-local/serve.py \
  --model-path /Users/chenjing/.local/share/jev-clef/models/clef-flash-17f0b0a \
  --model clef-flash --device mps --dtype bfloat16 --port 8818
```

The downloader pins revision `17f0b0ad64efb65d273590632833508766b2aae6`,
resumes partial files, checks exact byte counts and all LFS SHA256 hashes, and
does not create a duplicate Hugging Face snapshot cache. The complete release
contains 19,083,377,402 bytes.

MPS uses the official Transformers PyTorch fallback for Qwen3.5's linear
attention. Do not install the CUDA-only `flash-linear-attention` or
`causal-conv1d` packages into this environment. SDPA is selected for full
attention. No MPS memory-limit override or system-wide configuration change is
made. The CPU thread limit is process-local.

### Verified smoke, 2026-10-03

On the Mac Studio M1 Max (64 GiB), the unmodified official Clef-flash release
loaded in 65.57 seconds. All 9,531,576,564 parameters were BF16 on `mps:0`.
The public model-card invoice example (260 tokens, two questions) returned
`overdue=0.9732` and `large=true=0.9732` on both runs. The first call took 8.69
seconds and the warmed call took 1.09 seconds. This is a runtime smoke check,
not a benchmark accuracy result or a general latency estimate. Its complete
request, responses, and health snapshot are in
`smoke-flash-2026-10-03.json`.

The warmed server used 19,063,166,720 MPS allocation bytes and
19,964,674,048 MPS driver bytes. An 18,124-token text record was separately
verified to return HTTP 413 before model inference. No model compatibility
patch was needed. The official Torch fallback emits a warning about absent
CUDA fast-path packages; this is expected on this Mac.

## Other hosts

The same script supports `--device cuda`, `--dtype bfloat16`, `--model clef`,
`--model-path PATH`, `--revision SHA`, and `--port 8817`. If the model directory
has the downloader's `download-manifest.json`, the revision is read from it.
No secret or authorization header is needed. Keep any remote access behind a
local SSH tunnel rather than rebinding the service to a public interface.
