# Clef / Clef-flash local comparison, 2026-10-03

Execution uses the pinned, public Apache-2.0 weights and official joint schema head. Paid API calls are prohibited. The earlier unexecuted API plan is preserved in `manifest-api-plan-not-executed.json` and `PROTOCOL-api-plan-not-executed.md`; it produced no API results.

## Fixed inputs and scoring

`manifest.json` freezes the input and answer files, hashes, request counts, model revisions, and local result tags. CSAT: 884 items, 4,420 calls per model. Game: 1,064 items, 4,216 calls. English subset: 180 items, 694 calls. Claims: 32 sets, 14,113 items, 45,333 calls. Korean language results reuse game requests.

Every call preserves the original state, instructions, option IDs/descriptions, and the original rotation scheme. Calls contain exactly one `choice` question. The official encoder and schema head are used without rewriting prompts or changing probabilities. The official encoder sorts option keys internally and substitutes the question ID for empty instructions; those are model behavior. No text is truncated. Full-input token count is checked before inference, and the maximum frozen request is 10,819 tokens, within the official Python encoder default 16,384.

The existing scorers perform original-option probability averaging, unsupported-item handling, calibration, bootstrap and paired comparisons. `score.py --systems clef-flash` can publish one completed system before the other finishes. It requires exact `(item id, rotation, request SHA)` coverage, manifest input/answer hashes, and no unresolved execution errors. Existing reference records are linked into the new tag only; baseline files are not overwritten.

## Local execution

- Clef 27B: amd Windows, Radeon 8060S, ROCm. Dedicated directory `C:/Users/chenj/dev/jev-likes/clef-20261003`. Official revision `2f3de3dd85f379784083b0814d997ab627200f0c`.
- Clef-flash 9B: Mac Studio M1 Max, PyTorch MPS. Official revision `17f0b0ad64efb65d273590632833508766b2aae6`.
- Local ports: 8817 for Clef via SSH tunnel and 8818 for Clef-flash. Both implement `/v1/systemone` and `/health`.
- Batch size 1, unquantized bfloat16 is the initial setting. Runtime, actual device, dtype, precision changes, model revision, source SHA and latency are recorded in health and call metadata. A CPU fallback must be recorded as CPU and may not be labeled GPU performance.
- The AMD runtime is an isolated venv using existing ROCm torch2.9.1 read-only plus its own transformers5.10.2. Existing environments are not modified. The official model card reports testing torch2.11/transformers5.10.2 on H200; our hardware/runtime are therefore a separate local observation, not a reproduction of that hardware result.
- Use official direct-device `from_pretrained` loading and `HF_DEACTIVATE_ASYNC_LOAD=1` to limit simultaneous shard-loading memory. No BIOS/UMA change, global autocast, or unrelated-process termination.
- Before the full run, measure representative short and long requests, actual peak memory, throughput and estimated finish time. Loading successfully does not establish that the longest 10.8K-token claims input fits. GPU/CPU out-of-memory is an execution error, not an incorrect model answer.

New tags: `2027-09-clef-local-20261003`, `v1-clef-local-20261003`, `lang-en-clef-local-20261003`, `claims-clef-local-20261003`. The per-call schema and scorer remain compatible with the prior benchmark.

## Sources

- https://huggingface.co/Cloudflare/clef/tree/2f3de3dd85f379784083b0814d997ab627200f0c
- https://huggingface.co/Cloudflare/clef-flash/tree/17f0b0ad64efb65d273590632833508766b2aae6

The initial run is text-only; multimodal capabilities are outside this comparison. API charges are zero for the local path. Local electricity, bandwidth and hardware time are not estimated as API costs.

## Migration and long-input validation (2026-10-03)

The user cancelled the 27B Clef benchmark and requested moving the running
Clef-flash work from Mac to amd. The 27B checkpoint and diagnostic records were
not removed by this task. `active_systems` contains only Clef-flash; the launcher
refuses cancelled systems. Completed Mac CSAT records and reports are preserved.
Game, English and claims use new `*-clef-local-amd-20261003` tags. Inference is on
amd; the existing resumable runners and scoring remain on the Mac.

Before AMD predictions, `flash-migration-parity-plan.json` fixed 40 identical
original CSAT requests (25 subjects, 326–2,359 tokens). Acceptance required all
40 valid, at least 99% answer agreement (40/40 at this sample size), and maximum
per-option probability difference at most 0.02. Both the original AMD runtime
and the final runtime passed 40/40, maximum difference 0.0187; token counts
matched. Resource-refused requests were recorded as unmeasured and resumed only
after the unchanged reserve check passed, not treated as model disagreements.

An unchanged 10,819-token claims request exposed HIP OOM in the original runtime.
The final runtime installs `sdpa_chunked.py` only for CUDA inference with dropout
zero and query length above 4,096. It evaluates query rows in blocks of 512 while
retaining all keys/values, absolute causal row offsets, the original scale, masks
and dtype. Shorter game/English/CSAT requests delegate directly to the original
SDPA function. This is a disclosed attention allocation strategy; it does not
truncate the input or quantize the weights.

HIP BF16 tests with the actual head width 256 passed six numeric cases with zero
observed difference, including causal offsets, GQA, masks and broadcast cases.
The final 40-request model parity test passed again, with zero chunked calls for
those short requests. The full 10,819-token request then completed in 34.31 seconds;
the server counted 8 chunked calls and largest query length 10,819. This verifies
that the long path executed without truncation; it is not a direct full-length
Mac-versus-AMD output comparison. Results and failed attempts are preserved.

AMD loading uses official safetensors `pread/get_tensor` to avoid Windows COW
mapping and duplicate full-slice staging, with allocator warmup disabled. Between
completed requests the server synchronizes, collects unreferenced objects and
returns unused CUDA allocator cache. Input, weight revision, BF16 precision,
encoder and schema-head code remain fixed. Runtime metadata discloses the wrapper
configuration and hashes. Mac and AMD kernels need not be bit-identical.

Resource checks remain active: 8 GiB free result disk on Mac, 8 GiB free disk on
amd, and 512 MiB available Windows commit before a request. No BIOS/UMA/pagefile
configuration was changed by this task. Windows may resize its pagefile under
its existing settings. Process-scoped keep-awake assertions cover the benchmark
controller and its dedicated AMD server without changing power plans or keeping
the display on. The task-owned Mac GPU model was stopped only after AMD production
records increased and their actual device/revision metadata were checked.
