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
        for mode in ("coherent", "particle"):
            for distribution, parameters in (
                ("gaussian", {"mean": 0.25, "std": 0.0}),
                ("uniform", {"low": 0.25, "high": 0.25}),
            ):
                with self.subTest(mode=mode, distribution=distribution):
                    batch = make_batch()
                    before = default_collate(batch)
                    after = normalized_noise(make_config(
                        mode=mode, name=distribution, scalar_mode="coherent",
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

    def test_particle_scalar_opt_in_uses_a_separate_shared_event_draw(self):
        batch = make_batch()
        before = default_collate(batch)
        torch.manual_seed(201)
        after = normalized_noise(make_config(
            mode="particle", scalar_mode="coherent"), STATS)(batch)
        original_scalar = physical(before[1], SCALAR_NAMES)
        scalar_changes = (physical(after[1], SCALAR_NAMES) - original_scalar) / original_scalar
        self.assertTensorClose(scalar_changes[:, 0], scalar_changes[:, 1])
        particle_changes = fractional_changes(before[0], after[0])[:, :, 0]
        self.assertTrue(torch.all(particle_changes[~before[2]] !=
                                  scalar_changes[:, 0, None].expand_as(particle_changes)[~before[2]]))

    def test_explicit_scalar_none_preserves_scalars_in_coherent_mode(self):
        batch = make_batch()
        after = normalized_noise(make_config(scalar_mode="none"), STATS)(batch)
        self.assertTrue(torch.equal(default_collate(batch)[1], after[1]))

    def test_padding_unselected_features_and_cached_samples_are_unchanged(self):
        for mode in ("coherent", "particle"):
            with self.subTest(mode=mode):
                batch = make_batch()
                before = default_collate(batch)
                after = normalized_noise(make_config(mode=mode), STATS)(batch)
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
        for mode in ("coherent", "particle"):
            for name, parameters in (
                ("gaussian", {"mean": 0.2, "std": 0.0}),
                ("uniform", {"low": 0.2, "high": 0.2}),
            ):
                with self.subTest(mode=mode, distribution=name):
                    noiser = pickle.loads(pickle.dumps(normalized_noise(
                        make_config(mode=mode, name=name, **parameters), STATS)))
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
        self.apply_cli(config, "--noise-mode", "particle", "--noise-scalar-mode", "coherent",
                       "--noise-vector", "--noise-scalar", "total")
        self.assertEqual(config["noise"]["scalar_mode"], "coherent")
        self.assertEqual(config["noise"]["vector"], [])
        self.assertEqual(config["noise"]["scalar"], ["total"])

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
