"""Multiplicative training noise for normalized, cached event inputs.

Correlation mode and distribution are independent choices:

* coherent (legacy default): one draw per event, shared by all selected
  particles/features and, by default, selected event-level scalars.
* particle: one draw per particle in each event, shared by that particle's
  selected features. Different particles receive independent draws, even when
  they have the same species. This is NOT independent noise per feature and is
  NOT a species-wide calibration nuisance.

Both Gaussian and uniform draws are fractional changes epsilon, with physical
scaling x -> x*(1 + epsilon). Cached inputs are already standardized:

    z = (x - mean) / std
    z_noisy = (x*(1 + epsilon) - mean) / std
            = z + z*epsilon + (mean/std)*epsilon

The correction term is essential: multiplying z alone would scale around the
training-set mean instead of zero physical energy/momentum. Equal fractional
shifts in physical units need not look equal in normalized units. No clipping
is applied: a Gaussian is unbounded, and uniform bounds are user-controlled.
"""

import math

import torch
from torch.utils.data._utils.collate import default_collate

from transformer_ee.noise_config import validate_noise_config


class normalized_noise:
    """Training DataLoader collator for cached pandas or polars datasets.

    Example input configuration block::

        "noise": {
            "mode": "particle", "name": "gaussian",
            "mean": 0.0, "std": 0.2,
            "vector": ["png.calE"], "scalar": [],
            "scalar_mode": "auto"
        }

    For uniform draws use name="uniform", low=-0.2, high=0.2 instead.
    scalar_mode="auto" shares the coherent event draw with scalars in coherent
    mode, but leaves scalars unchanged in particle mode. Explicit "coherent" in
    particle mode draws a SEPARATE event shift for all selected scalars; "none"
    always leaves scalars unchanged. Scalars are never recomputed from particles:
    that requires a problem-specific mapping of retained prongs, orphan energy,
    thresholds, and other contributions.
    """

    def __init__(self, config, stats):
        """Cache corrections from stats: feature_name -> [training mean, std].

        Store plain parameters and use a bound sampling method, not a lambda,
        so spawned DataLoader workers (e.g. Windows) can pickle this collator.
        """
        self.config = validate_noise_config(config)
        self.mode = self.config["mode"]
        self.scalar_mode = self.config["scalar_mode"]
        if self.scalar_mode == "auto":
            self.scalar_mode = "coherent" if self.mode == "coherent" else "none"

        self.vector_indices = [config["vector"].index(name) for name in self.config["vector"]]
        self.scalar_indices = [config["scalar"].index(name) for name in self.config["scalar"]]
        self.vector_correction = self._corrections(self.config["vector"], stats)
        # Unchanged scalars need no normalization statistics. Selecting particle
        # mode must not invent a relationship between totals and particle draws.
        self.scalar_correction = self._corrections(
            self.config["scalar"] if self.scalar_mode != "none" else [], stats
        )

    @staticmethod
    def _corrections(names, stats):
        """Reject unusable statistics rather than injecting NaN/inf into batches."""
        corrections = []
        for name in names:
            try:
                mean, std = stats[name]
                mean, std = float(mean), float(std)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Missing or invalid normalization statistics for {name!r}") from exc
            if not math.isfinite(mean) or not math.isfinite(std) or std <= 0:
                raise ValueError(f"Normalization statistics for {name!r} need finite mean and positive std")
            correction = mean / std
            if not math.isfinite(correction):
                raise ValueError(f"Non-finite normalization correction for {name!r}")
            corrections.append(correction)
        return torch.tensor(corrections)

    def noise(self, size):
        """Sample fractional shifts using the worker's torch RNG.

        Do not reseed here: DataLoader seeds workers, and reseeding every batch
        would repeat perturbations. The coherent Gaussian call matches the
        historical implementation to preserve seeded draw order. Gaussian std=0
        and equal uniform bounds deliberately permit deterministic fixed shifts.
        """
        if self.config["name"] == "gaussian":
            return torch.normal(mean=self.config["mean"], std=self.config["std"], size=size)
        return self.config["low"] + torch.rand(size=size) * (
            self.config["high"] - self.config["low"]
        )

    def __call__(self, batch):
        """Collate before modifying inputs, so cached samples are never mutated.

        vector: (B, L, F); scalar: (B, S); mask: (B, L), True for padding.
        Targets, weights, and masks pass through unchanged. Editing dataset
        tensors before collation would accumulate noise across training epochs.
        """
        vector, scalar, mask, target, weight = default_collate(batch)
        batch_size, num_prongs, _ = vector.shape

        if self.mode == "coherent":
            # (B, 1, 1) broadcasts over BOTH particle and feature dimensions.
            # Reuse the same event draw for scalars to preserve old correlations.
            event_noise = self.noise(size=(batch_size,))
            vector_noise = event_noise[:, None, None]
        else:
            # (B, L, 1) broadcasts only over FEATURES. Each particle has its own
            # draw, but its selected energy/momentum components move together.
            # Padding consumes RNG draws too; its contents are preserved below.
            vector_noise = self.noise(size=(batch_size, num_prongs))[:, :, None]
            event_noise = None

        if self.vector_indices:
            selected = vector[:, :, self.vector_indices]
            epsilon = vector_noise.to(vector)
            correction = self.vector_correction.to(vector)[None, None, :]
            shifted = selected + (selected * epsilon + correction * epsilon)
            # Padding is not a real particle. Preserve it exactly in both modes.
            # The legacy implementation also smeared padding, which the model
            # masks out; real coherent entries retain the historical arithmetic.
            vector[:, :, self.vector_indices] = torch.where(mask[:, :, None], selected, shifted)

        if self.scalar_indices and self.scalar_mode == "coherent":
            # Particle draws have no unique scalar equivalent. An explicit
            # scalar opt-in uses a separate event draw, never an arbitrary
            # particle's draw or an unweighted average of particle shifts.
            if event_noise is None:
                event_noise = self.noise(size=(batch_size,))
            epsilon = event_noise.to(scalar)[:, None]
            correction = self.scalar_correction.to(scalar)[None, :]
            selected = scalar[:, self.scalar_indices]
            scalar[:, self.scalar_indices] = selected + (
                selected * epsilon + correction * epsilon
            )

        return vector, scalar, mask, target, weight
