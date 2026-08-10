import math

import pytest
import torch

from xtuner.v1.rl.loss import GRPOLossConfig
from xtuner.v1.rl.on_policy_distillation import (
    OPDConfig,
    OPDTeacherConfig,
    apply_opd_kl_to_advantages,
    compute_reverse_kl_distribution_metrics,
)


def _config(**kwargs) -> OPDConfig:
    return OPDConfig(
        teachers=[OPDTeacherConfig(name="teacher", endpoints=["http://unused"])],
        data_source_teacher_map={"test": "teacher"},
        **kwargs,
    )


class TestReverseKLDistributionMetrics:
    def test_statistics(self) -> None:
        values = torch.linspace(-4.0, 10.0, 10001)
        metrics = compute_reverse_kl_distribution_metrics(values)
        expected_quantiles = torch.quantile(values, torch.tensor([0.01, 0.99, 0.999]))
        histogram_bin_width = (values.max() - values.min()).item() / 8192

        torch.testing.assert_close(metrics["reverse_kl"], values.mean())
        torch.testing.assert_close(metrics["abs_logprob_loss"], values.abs().mean())
        torch.testing.assert_close(metrics["reverse_kl_variance"], values.var(unbiased=False))
        torch.testing.assert_close(metrics["reverse_kl_max_abs"], values.abs().max())
        for metric_name, expected in zip(
            ("reverse_kl_p1", "reverse_kl_p99", "reverse_kl_p999"),
            expected_quantiles,
        ):
            assert abs(metrics[metric_name].item() - expected.item()) <= histogram_bin_width * 2


class TestReverseKLClamp:
    def test_only_changes_advantage(self) -> None:
        old_logprobs = torch.tensor([-3.0, -0.5, 2.0, 7.0])
        teacher_logprobs = torch.zeros_like(old_logprobs)
        shifted_labels = torch.tensor([0, 0, 0, -100])
        loss_cfg = GRPOLossConfig(policy_loss_cfg={"loss_type": "vanilla"})
        loss_ctx = loss_cfg.build(
            {
                "shifted_labels": shifted_labels,
                "advantages": torch.zeros_like(old_logprobs),
                "old_logprobs": old_logprobs,
                "teacher_logprobs": teacher_logprobs,
            }
        )
        assert loss_ctx is not None

        reverse_kl_sum, abs_logprob_loss_sum = apply_opd_kl_to_advantages(
            loss_ctx,
            config=_config(reverse_kl_clamp=(-1.0, 1.0)),
        )

        result_device = loss_ctx.loss_kwargs.advantages.device
        torch.testing.assert_close(
            loss_ctx.loss_kwargs.advantages,
            torch.tensor([1.0, 0.5, -1.0, 0.0], device=result_device),
        )
        torch.testing.assert_close(reverse_kl_sum, torch.tensor(-1.5, device=result_device))
        torch.testing.assert_close(abs_logprob_loss_sum, torch.tensor(5.5, device=result_device))

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="lower bound"):
            _config(reverse_kl_clamp=(1.0, -1.0))

        with pytest.raises(ValueError, match="finite"):
            _config(reverse_kl_clamp=(-math.inf, 1.0))
