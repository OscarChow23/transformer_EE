# Training input noise

`normalized_noise` augments the selected model inputs while each training batch
is assembled. For a physical, unnormalized feature value `x`, it implements

```text
x_noisy = x * (1 + delta)
```

`delta` is a fractional change: `0.05` means +5%, and `-0.05` means -5%. Noise is
drawn again on each batch collation, including subsequent epochs. It does not
alter cached examples, targets, event weights, or the padding mask. Padded vector
rows are preserved. Validation and test loaders do not apply this augmentation.

## Correlation between particles and features

The `noise.mode` setting controls which inputs share a draw:

| Mode | Vector draws | Selected feature components within one particle |
| --- | --- | --- |
| `coherent` (default) | One draw per event, shared by every particle | Share the event draw |
| `particle` | One independent draw per particle per event | Share that particle's draw |

With vector tensor shape `(batch_size, number_of_particles, number_of_features)`,
the corresponding broadcast shapes are `(batch_size, 1, 1)` and
`(batch_size, number_of_particles, 1)`. Draws are independent across events in
both modes. Only features named in `noise.vector` are changed.

For example, if two particles have calorimetric energies of 1 and 2 GeV, a
coherent draw of +10% produces 1.1 and 2.2 GeV. In particle mode, draws of +10%
and -5% produce 1.1 and 1.9 GeV. If another feature is selected for the first
particle, it also receives +10% in this example.

Particle mode does **not** draw separately for each feature component, group
particles by species, or give all protons a common proton-specific shift. It
decorrelates reconstructed particles, including two particles of the same
species. Whether this augmentation improves a species calibration study must
be measured; the two perturbations encode different correlations.

## Distribution parameters

| `noise.name` | Parameters | Defaults | Meaning |
| --- | --- | --- | --- |
| `gaussian` | `mean`, `std` | `0.0`, `0.2` | Gaussian fractional change with the specified mean and standard deviation |
| `uniform` | `low`, `high` | `-0.2`, `0.2` | Uniform fractional change over the specified interval |

Gaussian noise is unbounded. No clipping to positive feature values is applied,
so a draw below -1 reverses the sign of a positive physical feature. Uniform
`low: -0.2, high: 0.2` keeps the multiplicative factor between 0.8 and 1.2.

The default widths are not matched in variance: a uniform interval `[-a, a]`
has standard deviation `a / sqrt(3)`. To match the variance of a zero-mean
Gaussian with `std: 0.2`, use uniform bounds of approximately
`-0.3464101615` and `0.3464101615`. Choose widths according to the intended
augmentation or uncertainty, rather than treating a bound as a Gaussian sigma.

## Event-level scalar inputs

There is no particle axis in an event-level scalar such as total calorimetric
energy. `noise.scalar_mode` makes its treatment explicit:

| `scalar_mode` | With `mode: "coherent"` | With `mode: "particle"` |
| --- | --- | --- |
| `auto` (default) | Selected scalars share the vector event draw | Scalars are unchanged |
| `none` | Scalars are unchanged | Scalars are unchanged |
| `coherent` | Selected scalars share the vector event draw | Selected scalars share a separate, independent draw per event |

Only scalars named in `noise.scalar` are eligible for changes. Empty feature
lists disable noise for that input category. In particle mode, an existing
scalar list is intentionally inactive under `scalar_mode: "auto"`.

The noiser does not infer or recompute scalar totals from perturbed particle
features. Leaving totals unchanged is an augmentation policy, not a guarantee
of a physically consistent calibration shift. Explicit scalar noising in
particle mode also does not enforce such consistency: its event draw is
independent of all particle draws. For calibration studies requiring dependent
quantities to agree, propagate the underlying particle changes into those
quantities using their physical definitions.

## JSON configuration

Add the following top-level `noise` member to a full training JSON config whose
top-level `vector` and `scalar` lists contain the named fields. Feature names
must match exactly. This fragment reproduces coherent Gaussian augmentation:

```json
{
  "noise": {
    "name": "gaussian",
    "mode": "coherent",
    "mean": 0.0,
    "std": 0.2,
    "scalar_mode": "auto",
    "vector": [
      "png.calE",
      "png.bpf[0].overlapE",
      "png.bpf[1].overlapE",
      "png.bpf[2].overlapE"
    ],
    "scalar": ["calE", "orphCalE", "remPngCalE"]
  }
}
```

For independent particle shifts uniformly distributed between -20% and +20%,
use this alternative block:

```json
{
  "noise": {
    "name": "uniform",
    "mode": "particle",
    "low": -0.2,
    "high": 0.2,
    "scalar_mode": "none",
    "vector": [
      "png.calE",
      "png.bpf[0].overlapE",
      "png.bpf[1].overlapE",
      "png.bpf[2].overlapE"
    ],
    "scalar": []
  }
}
```

The examples are configuration fragments, not complete training configs. To
use particle Gaussian noise, set `mode` to `particle` in the first example;
`scalar_mode: "auto"` then leaves the scalars unchanged. To use coherent uniform
noise, set `mode` to `coherent` in the second example and select scalar fields
with `scalar_mode: "auto"` if they should share the event shift.

Existing Gaussian JSON blocks without `mode` or `scalar_mode` retain their
coherent behavior. Omitting the entire `noise` block disables augmentation.

## CLI configuration in `train_wide.py`

CLI options update the top-level `noise` block before training:

| Option | Effect |
| --- | --- |
| `--enable-noise` | Enable augmentation, adding a block when absent |
| `--disable-noise` | Disable augmentation for this run |
| `--noise-mode coherent\|particle` | Set particle correlation |
| `--noise-name gaussian\|uniform` | Select distribution |
| `--noise-mean VALUE` / `--noise-std VALUE` | Set Gaussian parameters |
| `--noise-low VALUE` / `--noise-high VALUE` | Set uniform parameters |
| `--noise-vector [NAME ...]` | Replace the selected vector feature list |
| `--noise-scalar [NAME ...]` | Replace the selected scalar feature list |
| `--noise-scalar-mode auto\|none\|coherent` | Set event-level scalar handling |

An explicitly supplied option overrides its corresponding JSON value. An
omitted option preserves the JSON value. Distribution defaults are supplied
when the relevant parameters are absent. Noise-tuning options require an
existing JSON noise block or `--enable-noise`; they do not silently enable
augmentation. `--enable-noise` and `--disable-noise` are mutually exclusive.

For an existing coherent Gaussian noise block, override only the correlation:

```bash
python3 train_wide.py --base-config config.json --noise-mode particle
```

For a config without a noise block, enable particle uniform noise with an
explicit feature list (substitute names from your config):

```bash
python3 train_wide.py --base-config config.json \
  --enable-noise --noise-mode particle --noise-name uniform \
  --noise-low -0.2 --noise-high 0.2 \
  --noise-vector 'png.calE' 'png.bpf[0].overlapE' \
  --noise-scalar --noise-scalar-mode none
```

`--noise-scalar` with no following names clears the scalar list. The same syntax
works for `--noise-vector`. When enabling a new block, specify lists appropriate
to your dataset; the launcher's legacy default names need not exist in a NOvA
or other custom configuration.

To disable a JSON-configured augmentation for a comparison run:

```bash
python3 train_wide.py --base-config config.json --disable-noise
```

These commands illustrate noise options only. Supply the data, output, model,
and training settings appropriate to your campaign as usual.

## Normalization and interpretation

The data loader caches standardized inputs `z = (x - mean) / std`. Simply
multiplying `z` by a noisy factor would implement a different transformation
when `mean` is nonzero. The noiser therefore applies

```text
z_noisy = z + delta * (z + mean / std)
```

This is algebraically identical to scaling the physical value `x`, then
standardizing it with the original training statistics. Consequently, identical
fractional physical shifts need not look like identical percentage changes of
the standardized tensor entries. Compare values after undoing normalization
when checking the correlation structure.
