# Clef / Clef-flash final comparison, 2026-10-03

The existing inputs, option rotations, input hashes, and scorers are retained.
No model weights are downloaded. The hosted Cloudflare Workers AI deployment is
the evaluated system, not an independently version-pinned local checkpoint.

## Frozen inputs and run order

`manifest.json` records input hashes, exact full-input token counts from the
published tokenizer/framing, call counts, new result tags, and cost estimates.
Run CSAT (884 questions), game v1 (1,064), English language subset (180), then
claims (32 sets, 14,113). Korean language-subset results reuse game-v1 requests.
Each model retains one choice question per request and the existing rotation
rules: five rotations for CSAT and min(option count, 5) for other suites.

New tags: `2027-09-clef-20261003`, `v1-clef-20261003`,
`lang-en-clef-20261003`, `claims-clef-20261003`. New system names are `clef` and
`clef-flash`. Existing records, reports, and dirty worktree changes are preserved.

## Model interface

- REST POST to `/client/v4/accounts/{account}/ai/run/@cf/cloudflare/{model}`.
- State, instructions (including empty strings), and option keys/descriptions
  are sent unchanged as one `choice` question with ID `answer`.
- Credentials are read from Agent-vault references through `op-away run op run`.
  Secret values are never saved in this directory or logged.
- The public implementation sorts criteria internally and substitutes the
  question ID when instructions are empty. These are model-side behavior, not
  transformations added by the benchmark adapter.
- Neither the benchmark nor the adapter truncates input. All frozen requests are
  below 10,820 full-input tokens, below both the public Python encoder's 16,384
  default and the API's documented 65,536 limit. The adapter refuses any future
  input exceeding the documented API limit before sending it.

## Execution and errors

The initial limit is 2.4 requests/second per model process (4.8 requests/second
with both models active), below the published default 300 requests/minute.
Use the existing resumable runners with four workers per model. Retries follow
their existing policy; authentication, quota, and server failures are execution
errors, not unsupported items or wrong answers. System input-limit rejection is
unsupported and the existing scoring convention counts it as incorrect.

Every successful raw record additionally stores returned model, usage,
Cloudflare ray ID, estimated full-input tokens, and network latency.
`latency_ms` includes the adapter's rate-limit wait; use
`response_metadata.network_latency_ms` for latency summaries and label the
client/API environment. Latency is not directly comparable with original
benchmarks run on other dates and hardware.

## Scoring and reporting

Use the existing CSAT/game/claims scoring and bootstrap implementations. Old
reference records may be linked into new tag directories for paired comparisons;
they retain their original execution timestamps and model identity. Do not run
old reference models again as part of this extension. Missing/error records make
the suite incomplete; do not rank a partial suite as a completed result.

The reported model-card revisions freeze the reference documents and tokenizer,
not the API deployment weights. Results are dated API observations. Clef's
multimodal abilities are outside this text-only comparison.

## Official references, inspected 2026-10-03

- https://developers.cloudflare.com/workers-ai/models/clef/
- https://developers.cloudflare.com/workers-ai/models/clef-flash/
- https://developers.cloudflare.com/workers-ai/platform/limits/
- https://huggingface.co/Cloudflare/clef/tree/2f3de3dd85f379784083b0814d997ab627200f0c
- https://huggingface.co/Cloudflare/clef-flash/tree/17f0b0ad64efb65d273590632833508766b2aae6

Input-token unit prices: Clef $0.24/million and Clef-flash $0.09/million. The
frozen full suite is estimated at $10.95513936 combined before retries or any
service-side billing/framing difference; CSAT + game are about $1.55 combined.
