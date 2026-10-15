"""Numerical regression checks for the inference-only Euler loop optimization."""

from types import SimpleNamespace
import unittest

import torch

from modules.flow_matching import BASECFM


class ConditionedEstimator(torch.nn.Module):
    def forward(self, x, prompt, lengths, t, style, mu):
        # Exercise every conditioning input, including the CFG null branch.
        return (x * .125 + prompt * .25 + mu.transpose(1, 2) * .5
                + style[:, 0, None, None] / 8 + t[:, None, None] / 4)


class EulerRegressionTests(unittest.TestCase):
    def test_preserves_legacy_outputs(self):
        # Frozen CPU float32 results from d5eb976, before the loop optimization.
        cases = [
            (0., False, [[0., .36287838220596313, .5645471811294556, .766215980052948],
                         [0., .7111151814460754, .9127839803695679, 1.114452838897705]]),
            (.7, False, [[0., .5004099011421204, .7937663793563843, 1.087122917175293],
                         [0., .8944905996322632, 1.1878470182418823, 1.481203556060791]]),
            (.7, True, [[0., .27773982286453247, .3484262228012085, .4191126227378845],
                        [0., .8944905996322632, 1.1878470182418823, 1.481203556060791]]),
        ]
        for cfg, zero_prompt, expected in cases:
            with self.subTest(cfg=cfg, zero_prompt=zero_prompt):
                model = BASECFM(SimpleNamespace(
                    DiT=SimpleNamespace(in_channels=2, zero_prompt_speech_token=zero_prompt),
                    reg_loss_type='l2',
                ))
                model.estimator = ConditionedEstimator()
                with torch.inference_mode():
                    actual = model.solve_euler(
                        torch.arange(8).reshape(1, 2, 4).float() / 16,
                        torch.tensor([4]), torch.tensor([[[.25], [-.5]]]),
                        torch.arange(8).reshape(1, 4, 2).float() / 8,
                        torch.tensor([[.5, 1.]]), None, torch.linspace(0, 1, 5), cfg,
                    )
                torch.testing.assert_close(actual, torch.tensor([expected]), rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
