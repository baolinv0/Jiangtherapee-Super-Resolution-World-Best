"""Numerical contracts of the independent model and public module imports."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from jsr_repro.frontend import canonicalize, finish_feature, local_amplitude, make_controller_input, phase_splat, safe_sqrt
from jsr_repro.model import Controller, JSRModel, REFERENCE_AMPLITUDE, RefineNet
from jsr_repro.weights import load_controller_npz, load_refinenet_npz, load_static_affine_npz, verify_manifest_file

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "Core-Only-Source-Code"


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(77)
        self.raw = torch.rand(1, 4, 1, 16, 16) * 0.2
        self.shifts = torch.tensor([[[0., 0.], [.25, -.35], [-.4, .5], [.7, .2]]])
        self.model = JSRModel(controller_width=4, refine_width=8, refine_blocks=2)

    def test_shapes_variable_k_and_reference_validation(self):
        for k in (1, 4):
            outputs = self.model(self.raw[:, :k], self.shifts[:, :k])
            self.assertEqual(set(outputs), {"rgb", "legacy", "learned"})
            for value in outputs.values():
                self.assertEqual(value.shape, (1, 3, 32, 32))
                self.assertTrue(torch.isfinite(value).all())
        with self.assertRaises(ValueError):
            self.model(self.raw, self.shifts + 1)
        with self.assertRaises(ValueError):
            self.model(self.raw.expand(1, 4, 2, 16, 16), self.shifts)

    def test_zero_and_gain_equivariance(self):
        for value in self.model(torch.zeros_like(self.raw), self.shifts).values():
            self.assertEqual(value.abs().max().item(), 0)
        expected = self.model(self.raw, self.shifts)
        for gain in (0.001, 0.125, 3.7):
            actual = self.model(self.raw * gain, self.shifts)
            for key in expected:
                torch.testing.assert_close(actual[key], expected[key] * gain, rtol=2e-5, atol=2e-7)

    def test_all_trainable_layers_have_finite_nonzero_gradients(self):
        output = self.model(self.raw, self.shifts)["rgb"]
        target = torch.rand_like(output) * 0.3
        (output - target).square().mean().backward()
        for name, param in self.model.named_parameters():
            self.assertIsNotNone(param.grad, name)
            self.assertTrue(torch.isfinite(param.grad).all(), name)
            self.assertGreater(param.grad.abs().sum().item(), 0, name)

    def test_zero_sqrt_and_amplitude_gradients(self):
        zero = torch.zeros(1, 3, 3, 4, requires_grad=True)
        local_amplitude(zero).sum().backward()
        self.assertTrue(torch.isfinite(zero.grad).all())
        self.assertEqual(zero.grad.abs().max().item(), 0)
        x = torch.tensor([0., 4.], requires_grad=True)
        safe_sqrt(x).sum().backward()
        torch.testing.assert_close(x.grad, torch.tensor([0., .25]))

    def test_constant_cfa_colors_survive_splat_and_missing_evidence(self):
        raw = torch.zeros_like(self.raw)
        raw[:, :, 0, ::2, ::2] = .2
        raw[:, :, 0, 1::2, ::2] = .4
        raw[:, :, 0, ::2, 1::2] = .4
        raw[:, :, 0, 1::2, 1::2] = .7
        # Extreme non-reference shifts also exercise discarded observations.
        shifts = self.shifts.clone()
        shifts[:, 1:] = 1000
        evidence = phase_splat(raw, shifts)
        expected = raw.new_tensor([.2, .4, .7]).view(1, 3, 1, 1).expand_as(evidence["legacy"])
        torch.testing.assert_close(evidence["legacy"], expected, rtol=1e-5, atol=1e-6)

    def test_splat_shift_sign(self):
        raw = torch.zeros(1, 2, 1, 32, 32)
        raw[0, 1, 0, 16, 16] = 1
        shifts = torch.tensor([[[0., 0.], [1., 0.]]])
        total = phase_splat(raw, shifts)["sum"][0, 0]
        yy, xx = torch.meshgrid(torch.arange(64), torch.arange(64), indexing="ij")
        self.assertAlmostEqual(((total * xx).sum() / total.sum()).item(), 34.5, places=5)
        self.assertAlmostEqual(((total * yy).sum() / total.sum()).item(), 32.5, places=5)

    def test_synthesis_sensel_centers_match_linear_ramp(self):
        from jsr_repro.data import synthesize
        yy, xx = torch.meshgrid(torch.arange(128), torch.arange(128), indexing="ij")
        source = torch.stack([.1 + .001 * xx + .0007 * yy, .2 + .0005 * xx + .0011 * yy, .15 + .0008 * xx + .0004 * yy])
        options = {"native_size": 32, "scale": 2, "frames": 1, "max_shift": 0., "blur_sigma": [0., 0.], "exposure_ev": [0., 0.], "color_gain": [1., 1.], "shot_noise": [0., 0.], "read_noise": [0., 0.], "quantization_bits": 0, "clip_sensor": False, "augment": False}
        sample = synthesize(source, options, torch.Generator().manual_seed(41))
        evidence = phase_splat(sample["raw"][None], sample["shifts"][None])
        # Sample integration exactly preserves a linear ramp at each native
        # sensel center. Period-averaged interior error removes the residual
        # phase ripple of finite Gaussian interpolation, without fitting a gain
        # or shift; a half-HR-pixel coordinate error fails this assertion.
        error = evidence["legacy"][0, :, 8:-8, 8:-8] - sample["target"][:, 8:-8, 8:-8]
        torch.testing.assert_close(error.mean((-2, -1)), torch.zeros(3), rtol=0, atol=2e-7)

    def test_affine_buffers_persist_and_reject_invalid_std(self):
        mean, std = torch.ones(1, 151, 1, 1) * .02, torch.ones(1, 151, 1, 1) * 1.2
        self.model.set_feature_affine(mean, std)
        other = JSRModel(controller_width=4, refine_width=8, refine_blocks=2)
        other.load_state_dict(self.model.state_dict())
        torch.testing.assert_close(other(self.raw, self.shifts)["rgb"], self.model(self.raw, self.shifts)["rgb"])
        with self.assertRaises(ValueError):
            self.model.set_feature_affine(mean, torch.zeros_like(std))

    def test_small_controller_and_factorized_refine_algebra(self):
        self.assertEqual(Controller(2)(torch.rand(1, 151, 3, 5)).shape, (1, 36, 3, 5))
        net = RefineNet(1, 1)
        with torch.no_grad():
            for param in net.parameters():
                param.zero_()
            net.inp.weight[0, 1, 1, 1] = 1
            net.guide.weight[0, 0, 1, 1] = 1
            net.body[0].c1.weight[0, 0, 1, 1] = 1
            net.body[0].c2.weight[0, 0, 1, 1] = 1
            net.gammas[0].weight[0, 0, 0, 0] = 1
            net.betas[0].bias[0] = .03
            net.out.weight[:, 0, 0, 0] = torch.tensor([.1, .2, .3])
        value = torch.tensor([.1, .2, .3, .4, .5, .6]).view(1, 6, 1, 1)
        h = (torch.tensor(.2) + F.gelu(F.gelu(torch.tensor(.2)))) * (1 + .1 * torch.tanh(torch.tensor(.2))) + .03
        delta = net(value)
        torch.testing.assert_close(delta.flatten(), h * torch.tensor([.1, .2, .3]))
        amplitude = torch.full((1, 1, 1, 1), REFERENCE_AMPLITUDE)
        expected = torch.tensor([.1, .2, .3]) + h * torch.tensor([.3, .1, .4])
        torch.testing.assert_close(net.factorized(value, amplitude).flatten(), expected)

    def test_feature_normalization_degrees(self):
        phase = torch.rand(1, 144, 5, 7)
        legacy = torch.rand(1, 3, 5, 7)
        factor = 3.5
        scaled_phase = phase.clone().reshape(1, 3, 48, 5, 7)
        scaled_phase[:, :, 16:] *= factor
        first, amp = make_controller_input(phase, legacy)
        second, amp2 = make_controller_input(scaled_phase.reshape_as(phase), legacy * factor)
        torch.testing.assert_close(first, second, rtol=3e-5, atol=1e-6)
        torch.testing.assert_close(amp2, amp * factor)

    @unittest.skipUnless(importlib.util.find_spec("cv2"), "optional OpenCV for exact upstream frontend parity")
    def test_original_frontend_parity(self):
        spec = importlib.util.spec_from_file_location("official_local_v7", CORE / "local_v7_frontend.py")
        official = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(official)
        rng = np.random.default_rng(62)
        legacy = rng.random((37, 39, 3), dtype=np.float32)
        phase = rng.random((144, 37, 39), dtype=np.float32)
        actual, amplitude = make_controller_input(torch.from_numpy(phase)[None], torch.from_numpy(legacy.transpose(2, 0, 1).copy())[None])
        expected, expected_amp = official.canonicalize(official.finish_feature(phase, legacy), legacy)
        np.testing.assert_allclose(actual.numpy()[0], expected, rtol=4e-5, atol=2e-5)
        np.testing.assert_allclose(amplitude.numpy()[0], expected_amp, rtol=2e-6, atol=1e-7)


class WeightTests(unittest.TestCase):
    @unittest.skipUnless((CORE / "weights/v97_k04_controller_wgpu.npz").exists(), "official weights optional")
    def test_official_module_import_and_manifest(self):
        paths = CORE / "weights"
        controller = load_controller_npz(paths / "v97_k04_controller_wgpu.npz")
        refine = load_refinenet_npz(paths / "v97_k04_rarm_wgpu.npz")
        self.assertEqual(refine.inp.out_channels, 116)
        self.assertEqual(len(refine.body), 8)
        self.assertEqual(len(controller.state_dict()), 30)
        mean, std = load_static_affine_npz(paths / "local_v7_static_affine_v1.npz")
        self.assertEqual(mean.shape, (1, 151, 1, 1))
        self.assertTrue((std > 0).all())
        for filename in ("v97_k04_controller_wgpu.npz", "v97_k04_rarm_wgpu.npz", "local_v7_static_affine_v1.npz"):
            self.assertEqual(len(verify_manifest_file(paths / filename, paths / "manifest.json")), 64)
        # A standalone residual-module inference proves real weights are usable.
        self.assertTrue(torch.isfinite(refine(torch.rand(1, 6, 9, 11))).all())

    def test_strict_import_rejects_extra_and_nonfinite_weights(self):
        net = RefineNet(2, 1)
        state = {name: value.detach().numpy() for name, value in net.state_dict().items()}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "refine.npz"
            np.savez(path, **state)
            loaded = load_refinenet_npz(path)
            torch.testing.assert_close(loaded(torch.ones(1, 6, 2, 2)), net(torch.ones(1, 6, 2, 2)))
            np.savez(path, **state, unexpected=np.zeros(1, dtype=np.float32))
            with self.assertRaises(ValueError):
                load_refinenet_npz(path)
            state["inp.bias"][0] = np.nan
            np.savez(path, **state)
            with self.assertRaises(ValueError):
                load_refinenet_npz(path)


if __name__ == "__main__":
    unittest.main()
