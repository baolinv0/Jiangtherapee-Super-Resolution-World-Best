"""Hand-derived probes for release-coordinate tap production and consumption."""
import math

import pytest
import torch


def test_phase_bins_use_two_pixel_period_and_count_is_normalized_by_k():
    from jsr_repro.recovered_frontend import phase_statistics
    frames = torch.tensor([[[[1., 2.], [2., 3.]]]])
    flow = torch.zeros(1, 1, 2, 2, 2)
    covariance = torch.zeros(1, 3, 1, 1)
    first = phase_statistics(frames, flow, covariance)
    expected = torch.zeros(16)
    expected[[0, 2, 8, 10]] = 14
    torch.testing.assert_close(first["phase"][0, :16, 0, 0], expected)
    torch.testing.assert_close(first["legacy"].flatten(), torch.tensor([1., 2., 3.]))
    duplicate = phase_statistics(frames.repeat(1, 2, 1, 1), flow.repeat(1, 2, 1, 1, 1), covariance)
    torch.testing.assert_close(first["phase"], duplicate["phase"])


def test_phase_dispersion_is_standard_deviation_not_rms():
    from jsr_repro.recovered_frontend import phase_statistics
    frames = torch.tensor([[[[1., 2.], [2., 3.]], [[3., 4.], [4., 5.]]]])
    result = phase_statistics(frames, torch.zeros(1, 2, 2, 2, 2), torch.zeros(1, 3, 1, 1))
    phase = result["phase"].reshape(1, 3, 48, 1, 1)
    supported = phase[:, :, :16] > 0
    torch.testing.assert_close(phase[:, :, 32:][supported], torch.ones(12))
    torch.testing.assert_close(result["legacy"].flatten(), torch.tensor([2., 3., 4.]))


def test_continuous_dx_basis_changes_sensor_weights_before_ratio():
    from jsr_repro.recovered_frontend import learned_merge, phase_statistics
    frames = torch.arange(6.).view(1, 1, 1, 6).expand(1, 1, 6, 6).clone()
    flow, covariance = torch.zeros(1, 1, 6, 6, 2), torch.zeros(1, 3, 1, 1)
    qmap = torch.zeros(1, 36, 1, 1)
    base = phase_statistics(frames, flow, covariance, origin_yx=(2., 2.))["legacy"]
    torch.testing.assert_close(learned_merge(frames, flow, covariance, qmap, origin_yx=(2., 2.)), base)
    qmap[:, 2] = 1.
    actual = learned_merge(frames, flow, covariance, qmap, origin_yx=(2., 2.))
    w = math.exp(5 * math.tanh(2 / 5))
    assert actual[0, 0, 0, 0].item() == pytest.approx((2 + 4 * w) / (1 + w), abs=1e-6)


def test_recovered_merge_has_finite_raw_and_control_gradients():
    from jsr_repro.recovered_frontend import learned_merge
    torch.manual_seed(75)
    frames = torch.rand(1, 2, 4, 4, requires_grad=True)
    flow = torch.zeros(1, 2, 4, 4, 2)
    covariance = torch.tensor([1., .1, 1.1]).view(1, 3, 1, 1).expand(1, 3, 3, 5)
    qmap = (torch.randn(1, 36, 3, 5) * .1).requires_grad_()
    result = learned_merge(frames, flow, covariance, qmap)
    result.square().mean().backward()
    for value in (frames.grad, qmap.grad):
        assert value is not None and torch.isfinite(value).all() and value.abs().sum() > 0


def test_recovered_stage_rejects_nonfinite_inputs_and_invalid_cap():
    from jsr_repro.recovered_frontend import learned_merge, phase_statistics
    frames = torch.ones(1, 1, 2, 2)
    flow, covariance = torch.zeros(1, 1, 2, 2, 2), torch.zeros(1, 3, 1, 1)
    with pytest.raises(ValueError):
        phase_statistics(frames * float("nan"), flow, covariance)
    with pytest.raises(ValueError):
        learned_merge(frames, flow, covariance, torch.zeros(1, 36, 1, 1), cap=0)


def test_recovered_assembly_is_gain_equivariant_and_trainable():
    from jsr_repro.recovered_model import RecoveredJSR
    torch.set_num_threads(2)
    torch.manual_seed(52)
    model = RecoveredJSR(controller_width=2, refine_width=4, refine_blocks=1)
    frames = torch.rand(1, 2, 6, 6) * .2
    flow = torch.randn(1, 2, 6, 6, 2) * .15
    covariance = torch.tensor([1., .1, 1.1]).view(1, 3, 1, 1).expand(1, 3, 8, 10)
    result = model(frames, flow, covariance)
    gained = model(frames * 3, flow, covariance)
    torch.testing.assert_close(gained["rgb"], result["rgb"] * 3, atol=2e-6, rtol=2e-4)
    result["rgb"].square().mean().backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name


def test_recovered_loader_verifies_k_route_and_affine():
    from pathlib import Path
    from jsr_repro.recovered_model import load_recovered_jsr
    from jsr_repro.weights import load_controller_npz
    weight_dir = Path(__file__).resolve().parents[1] / "Core-Only-Source-Code/weights"
    model = load_recovered_jsr(weight_dir, 13)
    controller14 = load_controller_npz(weight_dir / "v97_k14_controller_wgpu.npz")
    torch.testing.assert_close(model.controller.enc[0][0].weight, controller14.enc[0][0].weight)
    assert model.feature_std.min() > 0
    with pytest.raises(ValueError):
        model(torch.ones(1, 1, 2, 2), torch.zeros(1, 1, 2, 2, 2), torch.ones(1, 3, 1, 1))


def test_recovered_cli_saves_linear_stage_outputs_and_provenance(tmp_path):
    import json
    from pathlib import Path
    import numpy as np
    from jsr_repro.infer_recovered import run
    path = tmp_path / "boundary.npz"
    np.savez(path, frames=np.ones((1, 4, 4), np.float32) * .1,
             flow=np.zeros((1, 4, 4, 2), np.float32),
             covariance=np.broadcast_to(np.array([1., 0., 1.], np.float32)[:, None, None], (3, 8, 8)))
    weights = Path(__file__).resolve().parents[1] / "Core-Only-Source-Code/weights"
    report = run(path, weights, tmp_path / "result", refinement=False)
    assert report["k"] == 1 and report["lca"] == "disabled"
    saved = np.load(tmp_path / "result/stages.npz", allow_pickle=False)
    np.testing.assert_allclose(saved["learned"], .1, atol=1e-6)
    assert saved["rgb"].shape == (8, 8, 3)
    assert json.loads((tmp_path / "result/provenance.json").read_text())["input_sha256"]


def test_enabled_zero_lca_clips_both_branches_and_recomputes_amplitude():
    from jsr_repro.recovered_model import RecoveredJSR
    model = RecoveredJSR(controller_width=2, refine_width=4, refine_blocks=1, refinement=False)
    frames, flow = torch.full((1, 1, 4, 4), -.1), torch.zeros(1, 1, 4, 4, 2)
    covariance = torch.tensor([1., 0., 1.]).view(1, 3, 1, 1).expand(1, 3, 8, 8)
    disabled = model(frames, flow, covariance, diagnostics=True)
    enabled = model(frames, flow, covariance, lca_coefficients_rb=[0., 0.], diagnostics=True)
    assert disabled["learned"].max() < 0 and disabled["amplitude"].min() > 0
    for name in ("learned", "legacy", "rgb", "amplitude"):
        assert enabled[name].abs().max() == 0


def test_controller_frontend_matches_released_extension_with_bundled_dependencies():
    from pathlib import Path
    import numpy as np
    from jsr_repro.frontend import make_controller_input
    from jsr_repro.weights import load_static_affine_npz
    root = Path(__file__).resolve().parents[1]
    mean, std = load_static_affine_npz(root / 'Core-Only-Source-Code/weights/local_v7_static_affine_v1.npz')
    with np.load(root / 'tests/data/recovered_frontend_oracle.npz', allow_pickle=False) as fixture:
        for i in range(int(fixture['case_count'])):
            feature, amplitude = make_controller_input(torch.from_numpy(fixture[f'phase_{i}']),
                                                       torch.from_numpy(fixture[f'legacy_{i}']), mean, std)
            torch.testing.assert_close(feature, torch.from_numpy(fixture[f'feature_{i}']), atol=3e-4, rtol=2e-4)
            torch.testing.assert_close(amplitude, torch.from_numpy(fixture[f'amplitude_{i}']), atol=2e-6, rtol=2e-6)


def test_recovered_assembly_rejects_roi_smaller_than_original_unet_pooling():
    from jsr_repro.recovered_model import RecoveredJSR
    model = RecoveredJSR(controller_width=1, refine_width=2, refine_blocks=1, refinement=False)
    with pytest.raises(ValueError, match='at least 8'):
        model(torch.ones(1, 1, 8, 8), torch.zeros(1, 1, 8, 8, 2), torch.ones(1, 3, 3, 5))
