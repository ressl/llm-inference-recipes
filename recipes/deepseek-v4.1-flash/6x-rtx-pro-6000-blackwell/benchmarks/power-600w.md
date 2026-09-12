# Follow-up: 600 W power limits

On 2026-09-12, the operator raised the remaining selected GPUs from 350 W to
600 W, so all six inference GPUs have the same 600 W ceiling. The unused seventh
card also received that limit and remains excluded from inference because its
PCIe link is x8. The model process was not restarted, and its GPU order, layer
partition `8,12,20`, 24-request limit, original weights and fixed KV allocation
stayed unchanged. The host's periodic power-limit service preserves the setting.

The [earlier concurrency report](concurrency.md), including the one-hour soak,
used four 600 W and two 350 W limits. Its measurements remain historical results
for that configuration. This follow-up is a short comparison, not another
one-hour soak or a qualification of every long-context workload at 600 W.

## Short-input comparison

Each request has exactly 1,024 input and 512 forced output tokens. There are
two repetitions at each concurrency and power setting, using fixture seed 42,
fresh cache salts and temperature zero. All request markers, API usage and
server counters passed; there were zero prefix hits and zero preemptions.
Aggregate throughput includes prefill and decode. The following cells list
both runs rather than hiding their variation behind an average.

| Concurrent requests | Four 600 W + two 350 W: aggregate tokens/s | Six 600 W: aggregate tokens/s |
| ---: | ---: | ---: |
| 1 | 98.72, 99.53 | 99.07, 99.53 |
| 8 | 381.12, 349.34 | 383.97, 381.56 |
| 24 | 477.23, 478.17 | 477.92, 707.13 |

Single-stream decode remains approximately **104 tokens/s** in both settings.
The 24-stream result varies substantially. Similar variation was present in
the earlier mixed-power qualification, and seeded inputs do not produce
bitwise-identical batched outputs. These measurements therefore do **not**
establish a causal percentage speedup from raising the limit. Long-prefill
performance was not compared in this follow-up.

[Numeric evidence](power-600w-2026-09-12.json) contains all twelve completed
comparison runs, per-stream timings, response hashes and counter deltas.

## Interpretation

A power limit is a ceiling, not constant consumption. Higher limits can allow
higher clocks when power capping is active, but do not add VRAM or KV capacity.
See NVIDIA's [clock-event documentation](https://docs.nvidia.com/deploy/nvidia-smi/#clocks-event-reasons).
The same GPU assignment can be retained once the limits are equalized; this
experiment did not test a different assignment or different layer boundaries.

Power settings belong to the host operator. The portable launcher does not
change them automatically. Record your actual limits with your own results;
the recipe's earlier mixed-power configuration remains a tested reference.
