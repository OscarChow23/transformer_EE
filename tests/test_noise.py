"""Physical and statistical contracts for calorimetric training augmentation.

Run with ``python -m unittest discover -s tests -p 'test_noise.py'``.  Tests
recover the physical (unstandardized) inputs before comparing scale changes:
relative changes in standardized tensor coordinates are not calibration shifts.
"""

import copy
import importlib.util
from pathlib import Path
import pickle
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from transformer_ee.dataloader.noise import normalized_noise
from transformer_ee.noise_config import validate_noise_config


VECTOR_NAMES = ["energy", "overlap", "direction"]
SCALAR_NAMES = ["total", "remainder", "count"]
STATS = {
    "energy": (4.0, 2.0),
    "overlap": (3.0, 4.0),
    "direction": (0.5, 0.25),
    "total": (7.0, 2.0),
    "remainder": (3.0, 5.0),
    "count": (2.0, 1.0),
}


def make_config(**noise_overrides):
    noise = {
        "name": "gaussian",
        "mean": 0.0,
        "std": 0.2,
        "vector": VECTOR_NAMES[:2],
        "scalar": SCALAR_NAMES[:2],
    }
    noise.update(noise_overrides)
    return {"vector": VECTOR_NAMES[:], "scalar": SCALAR_NAMES[:], "noise": noise}


def make_batch():
    """Unequal means/stds catch incorrect scaling in standardized coordinates."""
    vector = torch.arange(1, 37, dtype=torch.float32).reshape(3, 4, 3) / 5
    scalar = torch.arange(1, 10, dtype=torch.float32).reshape(3, 3) / 4
    mask = torch.tensor([[False, False, False, True],
                         [False, False, False, False],
                         [False, False, True, True]])
    vector[mask] = 0.0
    target = torch.tensor([[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]])
    weight = torch.tensor([[1.0], [0.5], [2.0]])
    return list(zip(vector, scalar, mask, target, weight))


def physical(tensor, names):
    mean = tensor.new_tensor([STATS[name][0] for name in names])
    std = tensor.new_tensor([STATS[name][1] for name in names])
    return tensor * std + mean


def fractional_changes(before, after):
    original = physical(before, VECTOR_NAMES)
    changed = physical(after, VECTOR_NAMES)
    return (changed - original) / original


class NoiseBehaviorTests(unittest.TestCase):
    def assertTensorClose(self, actual, expected):
        torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)

    def test_constant_shifts_reproduce_physical_multiplication(self):
        """Both distributions and modes obey x -> (1 + epsilon) x."""
        for mode, scalar_mode in (
            ("coherent", "coherent"), ("coherent", "independent"),
            ("particle", "independent"), ("particle", "particle_mean"),
        ):
            for distribution, parameters in (
                ("gaussian", {"mean": 0.25, "std": 0.0}),
                ("uniform", {"low": 0.25, "high": 0.25}),
            ):
                with self.subTest(mode=mode, scalar_mode=scalar_mode, distribution=distribution):
                    batch = make_batch()
                    before = default_collate(batch)
                    after = normalized_noise(make_config(
                        mode=mode, name=distribution, scalar_mode=scalar_mode,
                        **parameters), STATS)(batch)
                    selected = ~before[2]
                    self.assertTensorClose(
                        physical(after[0], VECTOR_NAMES)[selected, :2],
                        physical(before[0], VECTOR_NAMES)[selected, :2] * 1.25)
                    self.assertTensorClose(
                        physical(after[1], SCALAR_NAMES)[:, :2],
                        physical(before[1], SCALAR_NAMES)[:, :2] * 1.25)

    def test_zero_and_signed_physical_values_scale_without_a_spurious_offset(self):
        for mode in ("coherent", "particle"):
            with self.subTest(mode=mode):
                batch = make_batch()
                # A physical zero has normalized coordinate -mean/std.  Noise
                # must keep it at zero; signed values must scale about zero.
                batch[0][0][0, 0] = -STATS["energy"][0] / STATS["energy"][1]
                batch[0][0][1, 0] = (-3 - STATS["energy"][0]) / STATS["energy"][1]
                before = default_collate(batch)
                after = normalized_noise(make_config(
                    mode=mode, mean=0.5, std=0.0), STATS)(batch)
                actual = physical(after[0], VECTOR_NAMES)[0, :2, 0]
                self.assertTensorClose(actual, torch.tensor([0.0, -4.5]))
                self.assertTensorClose(
                    physical(after[0], VECTOR_NAMES)[~before[2], :2],
                    physical(before[0], VECTOR_NAMES)[~before[2], :2] * 1.5)

    def test_coherent_default_keeps_legacy_draws_and_physical_results(self):
        batch = make_batch()
        before = default_collate(batch)
        torch.manual_seed(881)
        expected = torch.normal(mean=0.0, std=0.2, size=(len(batch),))
        torch.manual_seed(881)
        after = normalized_noise(make_config(), STATS)(batch)
        changes = fractional_changes(before[0], after[0])
        for event in range(len(batch)):
            self.assertTensorClose(
                changes[event, ~before[2][event], :2],
                expected[event].expand((~before[2][event]).sum(), 2))
        self.assertTensorClose(
            physical(after[1], SCALAR_NAMES)[:, :2],
            physical(before[1], SCALAR_NAMES)[:, :2] * (1 + expected[:, None]))

    def test_each_particle_shares_features_but_has_its_own_draw(self):
        for name in ("gaussian", "uniform"):
            with self.subTest(distribution=name):
                batch = make_batch()
                before = default_collate(batch)
                torch.manual_seed(211)
                after = normalized_noise(make_config(
                    mode="particle", name=name, low=-0.2, high=0.2), STATS)(batch)
                changes = fractional_changes(before[0], after[0])
                valid_changes = changes[~before[2], :2]
                self.assertTensorClose(valid_changes[:, 0], valid_changes[:, 1])
                # There are several particles per event and several events; no
                # selected particle should inherit an event-wide/common draw.
                self.assertEqual(torch.unique(valid_changes[:, 0]).numel(),
                                 valid_changes.shape[0])
                self.assertTrue(torch.equal(before[1], after[1]))

    def test_uniform_coherent_shares_the_event_draw_with_scalars(self):
        batch = make_batch()
        before = default_collate(batch)
        torch.manual_seed(91)
        after = normalized_noise(make_config(
            mode="coherent", name="uniform", low=-0.1, high=0.3), STATS)(batch)
        changes = fractional_changes(before[0], after[0])
        event_draws = changes[:, 0, 0]
        self.assertEqual(torch.unique(event_draws).numel(), len(batch))
        for event in range(len(batch)):
            self.assertTensorClose(
                changes[event, ~before[2][event], :2],
                event_draws[event].expand((~before[2][event]).sum(), 2))
        self.assertTensorClose(
            physical(after[1], SCALAR_NAMES)[:, :2],
            physical(before[1], SCALAR_NAMES)[:, :2] * (1 + event_draws[:, None]))

    def test_independent_scalar_features_use_their_own_sequential_draws(self):
        for mode in ("coherent", "particle"):
            for distribution in ("gaussian", "uniform"):
                with self.subTest(mode=mode, distribution=distribution):
                    batch = make_batch()
                    # Independent scalar noise still applies to an event with
                    # no retained particles. Its policy does not use the mask.
                    batch[2][2][:] = True
                    batch[2][0][:] = 0.0
                    batch[0][1][0] = -STATS["total"][0] / STATS["total"][1]
                    batch[0][1][1] = (-4 - STATS["remainder"][0]) / STATS["remainder"][1]
                    before = default_collate(batch)
                    config = make_config(mode=mode, name=distribution,
                                         low=-0.2, high=0.2,
                                         scalar_mode="independent",
                                         scalar=["remainder", "total"])

                    def sample(size):
                        if distribution == "gaussian":
                            return torch.normal(mean=0.0, std=0.2, size=size)
                        return -0.2 + torch.rand(size=size) * 0.4

                    torch.manual_seed(201)
                    vector_shape = (3,) if mode == "coherent" else (3, 4)
                    expected_vector = sample(vector_shape)
                    expected_scalar = sample((3, 2))
                    torch.manual_seed(201)
                    after = normalized_noise(config, STATS)(batch)
                    changes = fractional_changes(before[0], after[0])
                    expected_vector = (expected_vector[:, None].expand(3, 4)
                                       if mode == "coherent" else expected_vector)
                    self.assertTensorClose(changes[~before[2], 0], expected_vector[~before[2]])
                    # Feature selection order determines the independent draws;
                    # selection here is deliberately reversed to detect mixing.
                    self.assertTensorClose(
                        physical(after[1], SCALAR_NAMES)[:, [1, 0]],
                        physical(before[1], SCALAR_NAMES)[:, [1, 0]] * (1 + expected_scalar))
                    self.assertTrue(torch.all(expected_scalar[:, 0] != expected_scalar[:, 1]))
                    self.assertTrue(torch.equal(after[1][:, 2], before[1][:, 2]))
                    self.assertFalse(torch.equal(after[1][2, :2], before[1][2, :2]))

    def test_particle_mean_reuses_only_real_particle_draws_without_energy_weighting(self):
        for distribution in ("gaussian", "uniform"):
            with self.subTest(distribution=distribution):
                batch = make_batch()
                batch[2][2][:] = True
                batch[2][0][:] = 0.0
                # Scalar normalization includes physical zeros and negatives.
                batch[0][1][0] = -STATS["total"][0] / STATS["total"][1]
                batch[0][1][1] = (-4 - STATS["remainder"][0]) / STATS["remainder"][1]
                before = default_collate(batch)
                throws = torch.tensor([
                    [-0.1, 0.2, 0.05, 0.2],   # Last particle is padding: mean=0.05.
                    [-0.1, 0.1, 0.2, 0.2],    # Four real particles: mean=0.1.
                    [0.2, 0.2, 0.2, 0.2],     # No real particles: mean=0, not 0.2.
                ])
                mean = torch.tensor([0.05, 0.1, 0.0])
                noiser = normalized_noise(make_config(mode="particle", name=distribution,
                                                     scalar_mode="particle_mean"), STATS)
                # A second draw would exhaust side_effect and fail. The scalar
                # mean must use the very draws that moved the vector entries.
                with patch.object(noiser, "noise", side_effect=[throws]) as sampler:
                    after = noiser(batch)
                sampler.assert_called_once_with(size=(3, 4))
                changes = fractional_changes(before[0], after[0])
                for feature in (0, 1):
                    self.assertTensorClose(changes[~before[2], feature], throws[~before[2]])
                self.assertTensorClose(
                    physical(after[1], SCALAR_NAMES)[:, :2],
                    physical(before[1], SCALAR_NAMES)[:, :2] * (1 + mean[:, None]))
                self.assertTrue(torch.equal(after[1][2], before[1][2]))
                self.assertTrue(torch.equal(after[0][before[2]], before[0][before[2]]))
                self.assertTrue(torch.equal(after[1][:, 2], before[1][:, 2]))

    def test_particle_mean_of_one_real_particle_is_its_exact_fractional_shift(self):
        batch = make_batch()
        batch[0][2][:] = True
        batch[0][2][1] = False
        before = default_collate(batch)
        throws = torch.tensor([[0.2, -0.3, 0.4, 0.5],
                               [0.1, 0.2, 0.3, 0.4],
                               [0.2, 0.4, 0.6, 0.8]])
        noiser = normalized_noise(make_config(mode="particle", scalar_mode="particle_mean"), STATS)
        with patch.object(noiser, "noise", return_value=throws):
            after = noiser(batch)
        self.assertTensorClose(physical(after[1], SCALAR_NAMES)[0, :2],
                               physical(before[1], SCALAR_NAMES)[0, :2] * 0.7)

    def test_explicit_scalar_none_preserves_scalars_in_coherent_mode(self):
        batch = make_batch()
        after = normalized_noise(make_config(scalar_mode="none"), STATS)(batch)
        self.assertTrue(torch.equal(default_collate(batch)[1], after[1]))

    def test_padding_unselected_features_and_cached_samples_are_unchanged(self):
        for mode, scalar_mode in (("coherent", "auto"), ("particle", "auto"),
                                  ("coherent", "independent"), ("particle", "independent"),
                                  ("particle", "particle_mean")):
            with self.subTest(mode=mode, scalar_mode=scalar_mode):
                batch = make_batch()
                before = default_collate(batch)
                after = normalized_noise(make_config(mode=mode, scalar_mode=scalar_mode), STATS)(batch)
                self.assertTrue(torch.equal(after[0][before[2]], before[0][before[2]]))
                self.assertTrue(torch.equal(after[0][:, :, 2], before[0][:, :, 2]))
                self.assertTrue(torch.equal(after[1][:, 2], before[1][:, 2]))
                for index in (2, 3, 4):
                    self.assertTrue(torch.equal(after[index], before[index]))
                # Collation must allocate output storage, otherwise epochs would
                # compound noise in the cached dataset instead of redrawing it.
                for cached, original in zip(default_collate(batch), before):
                    self.assertTrue(torch.equal(cached, original))

    def test_successive_collations_draw_fresh_noise(self):
        for mode in ("coherent", "particle"):
            with self.subTest(mode=mode):
                noiser = normalized_noise(make_config(mode=mode), STATS)
                batch = make_batch()
                torch.manual_seed(19)
                first, second = noiser(batch), noiser(batch)
                self.assertFalse(torch.equal(first[0], second[0]))

    def test_empty_feature_selections_are_valid(self):
        batch = make_batch()
        for mode in ("coherent", "particle"):
            after = normalized_noise(make_config(mode=mode, vector=[], scalar=[]), STATS)(batch)
            for actual, expected in zip(after, default_collate(batch)):
                self.assertTrue(torch.equal(actual, expected))

    def test_all_padding_event_remains_unchanged(self):
        batch = make_batch()
        batch[0][2][:] = True
        batch[0][0][:] = 0.0
        for mode in ("coherent", "particle"):
            after = normalized_noise(make_config(mode=mode), STATS)(batch)
            self.assertTrue(torch.equal(after[0][0], batch[0][0]))

    def test_distribution_parameters_and_uniform_bounds(self):
        torch.manual_seed(81)
        gaussian = normalized_noise(make_config(mean=0.1, std=0.3), STATS).noise((32768,))
        self.assertAlmostEqual(gaussian.mean().item(), 0.1, delta=0.01)
        self.assertAlmostEqual(gaussian.std().item(), 0.3, delta=0.01)
        uniform = normalized_noise(make_config(name="uniform", low=-0.4, high=0.2), STATS).noise((32768,))
        self.assertTrue(torch.all((uniform >= -0.4) & (uniform <= 0.2)))
        self.assertAlmostEqual(uniform.mean().item(), -0.1, delta=0.01)

    def test_noiser_survives_pickle_and_spawn_dataloader_worker(self):
        """A lambda sampler would fail Windows/spawn DataLoader workers."""
        for mode, scalar_mode in (("coherent", "auto"), ("particle", "auto"),
                                  ("coherent", "independent"), ("particle", "independent"),
                                  ("particle", "particle_mean")):
            for name, parameters in (
                ("gaussian", {"mean": 0.2, "std": 0.0}),
                ("uniform", {"low": 0.2, "high": 0.2}),
            ):
                with self.subTest(mode=mode, scalar_mode=scalar_mode, distribution=name):
                    noiser = pickle.loads(pickle.dumps(normalized_noise(
                        make_config(mode=mode, scalar_mode=scalar_mode,
                                    name=name, **parameters), STATS)))
                    batch = make_batch()
                    loader = DataLoader(batch, batch_size=len(batch), num_workers=1,
                                        multiprocessing_context="spawn", collate_fn=noiser,
                                        timeout=30)
                    result = list(loader)
                    self.assertEqual(len(result), 1)
                    for actual, expected in zip(result[0], noiser(batch)):
                        self.assertTrue(torch.equal(actual, expected))


class NoiseConfigurationTests(unittest.TestCase):
    def test_defaults_do_not_modify_source_json(self):
        config = make_config()
        config["noise"] = {"vector": VECTOR_NAMES[:2], "scalar": SCALAR_NAMES[:2]}
        original = copy.deepcopy(config)
        result = validate_noise_config(config)
        self.assertEqual(config, original)
        self.assertEqual(result["mode"], "coherent")
        self.assertEqual(result["name"], "gaussian")
        self.assertEqual(result["mean"], 0.0)
        self.assertEqual(result["std"], 0.2)

    def test_uniform_defaults_and_degenerate_distributions(self):
        result = validate_noise_config(make_config(name="uniform"))
        self.assertEqual(result["low"], -0.2)
        self.assertEqual(result["high"], 0.2)
        validate_noise_config(make_config(std=0.0))
        validate_noise_config(make_config(name="uniform", low=0.1, high=0.1))

    def test_scalar_policy_combinations_are_validated(self):
        valid = (
            ("coherent", "auto"), ("coherent", "none"), ("coherent", "coherent"),
            ("coherent", "independent"), ("particle", "auto"), ("particle", "none"),
            ("particle", "independent"), ("particle", "particle_mean"),
        )
        for mode, scalar_mode in valid:
            with self.subTest(mode=mode, scalar_mode=scalar_mode):
                result = validate_noise_config(make_config(mode=mode, scalar_mode=scalar_mode))
                self.assertEqual(result["scalar_mode"], scalar_mode)
        for overrides in (
            {"mode": "particle", "scalar_mode": "coherent"},
            {"mode": "coherent", "scalar_mode": "particle_mean"},
            {"mode": "particle", "scalar_mode": "particle_mean", "vector": []},
        ):
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    validate_noise_config(make_config(**overrides))

    def test_invalid_statistics_fail_before_loading_training_batches(self):
        for mean, std in ((float("nan"), 1.0), (float("inf"), 1.0),
                          (1.0, 0.0), (1.0, -1.0), (1.0, float("nan")),
                          (1.0, float("inf"))):
            with self.subTest(mean=mean, std=std):
                stats = dict(STATS)
                stats["energy"] = (mean, std)
                with self.assertRaises((ValueError, TypeError)):
                    normalized_noise(make_config(), stats)

    def test_invalid_distribution_correlation_and_parameters_fail_early(self):
        invalid = [
            {"name": "poisson"}, {"mode": "independent"}, {"scalar_mode": "particle"},
            {"std": -0.1}, {"std": float("nan")}, {"std": float("inf")},
            {"mean": float("nan")}, {"mean": float("inf")}, {"std": True},
            {"mean": "0"}, {"name": "uniform", "low": 0.2, "high": -0.2},
            {"name": "uniform", "low": float("nan")},
            {"name": "uniform", "high": float("inf")},
            {"name": "uniform", "low": False},
            {"vector": ["unknown"]}, {"scalar": ["unknown"]},
            {"vector": "energy"}, {"scalar": "total"},
            {"vector": ["energy", "energy"]}, {"scalar": ["total", "total"]},
        ]
        for overrides in invalid:
            with self.subTest(overrides=overrides):
                with self.assertRaises((ValueError, TypeError)):
                    validate_noise_config(make_config(**overrides))


class NoiseCLITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Training imports are deferred by the CLI, so configuration tests do
        # not need W&B, a GPU, an experiment dataset, or mocked runtime modules.
        path = Path(__file__).resolve().parents[1] / "train_wide.py"
        spec = importlib.util.spec_from_file_location("train_wide_noise_tests", path)
        cls.cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.cli)

    def apply_cli(self, config, *flags):
        args = self.cli.build_argparser().parse_args(flags)
        self.cli.configure_noise(config, args)
        return config

    def test_unspecified_flags_preserve_configured_json_values(self):
        config = make_config(name="uniform", mode="particle", low=-0.37,
                             high=0.13, scalar_mode="none")
        original = copy.deepcopy(config["noise"])
        self.apply_cli(config)
        for key, value in original.items():
            self.assertEqual(config["noise"][key], value)

    def test_explicit_flags_override_only_the_named_json_values(self):
        config = make_config(mean=0.04, std=0.17)
        self.apply_cli(config, "--noise-mode", "particle", "--noise-std", "0.31")
        self.assertEqual(config["noise"]["mode"], "particle")
        self.assertEqual(config["noise"]["std"], 0.31)
        self.assertEqual(config["noise"]["mean"], 0.04)
        self.assertEqual(config["noise"]["vector"], VECTOR_NAMES[:2])

    def test_cli_can_enable_all_mode_and_distribution_combinations(self):
        for mode in ("coherent", "particle"):
            for name in ("gaussian", "uniform"):
                with self.subTest(mode=mode, distribution=name):
                    config = {"vector": VECTOR_NAMES[:], "scalar": SCALAR_NAMES[:]}
                    flags = ["--enable-noise", "--noise-mode", mode,
                             "--noise-name", name, "--noise-vector", "energy", "overlap",
                             "--noise-scalar"]
                    if name == "uniform":
                        flags.extend(["--noise-low", "-0.1", "--noise-high", "0.3"])
                    self.apply_cli(config, *flags)
                    self.assertEqual(config["noise"]["mode"], mode)
                    self.assertEqual(config["noise"]["name"], name)
                    self.assertEqual(config["noise"]["scalar"], [])
                    # This configuration must also be consumable by the actual
                    # collator, not merely accepted by the argument parser.
                    normalized_noise(config, STATS)(make_batch())

    def test_explicit_scalar_policy_and_empty_feature_lists(self):
        config = make_config()
        self.apply_cli(config, "--noise-mode", "particle", "--noise-scalar-mode", "independent",
                       "--noise-vector", "--noise-scalar", "total")
        self.assertEqual(config["noise"]["scalar_mode"], "independent")
        self.assertEqual(config["noise"]["vector"], [])
        self.assertEqual(config["noise"]["scalar"], ["total"])

    def test_cli_supports_all_three_explicit_scalar_policies(self):
        for mode, scalar_mode in (("coherent", "coherent"),
                                  ("coherent", "independent"),
                                  ("particle", "independent"),
                                  ("particle", "particle_mean")):
            with self.subTest(mode=mode, scalar_mode=scalar_mode):
                config = make_config()
                self.apply_cli(config, "--noise-mode", mode,
                               "--noise-scalar-mode", scalar_mode)
                self.assertEqual(config["noise"]["scalar_mode"], scalar_mode)
                normalized_noise(config, STATS)(make_batch())

    def test_cli_rejects_incompatible_scalar_policy_and_vector_mode(self):
        for mode, scalar_mode in (("particle", "coherent"), ("coherent", "particle_mean")):
            with self.subTest(mode=mode, scalar_mode=scalar_mode):
                with self.assertRaises(ValueError):
                    self.apply_cli(make_config(), "--noise-mode", mode,
                                   "--noise-scalar-mode", scalar_mode)
        with self.assertRaises(ValueError):
            self.apply_cli(make_config(), "--noise-mode", "particle",
                           "--noise-scalar-mode", "particle_mean", "--noise-vector")

    def test_disable_removes_json_noise_and_absent_noise_stays_absent(self):
        config = make_config()
        self.apply_cli(config, "--disable-noise")
        self.assertNotIn("noise", config)
        self.apply_cli(config)
        self.assertNotIn("noise", config)

    def test_tuning_without_enabled_noise_and_disable_conflicts_are_rejected(self):
        for config, flags in (
            ({"vector": VECTOR_NAMES, "scalar": SCALAR_NAMES}, ["--noise-mode", "particle"]),
            (make_config(), ["--disable-noise", "--noise-std", "0.1"]),
            (make_config(), ["--noise-name", "uniform", "--noise-low", "0.3", "--noise-high", "-0.2"]),
        ):
            with self.subTest(flags=flags):
                with self.assertRaises(ValueError):
                    self.apply_cli(config, *flags)


if __name__ == "__main__":
    unittest.main()
