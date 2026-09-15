import unittest

import torch

from scripts.interpolate_checkpoint import interpolate_state_dicts


class InterpolateCheckpointTests(unittest.TestCase):
    def test_interpolates_floats_and_preserves_integer_buffers(self) -> None:
        merged = interpolate_state_dicts(
            {"weight": torch.tensor([0.0, 2.0]), "index": torch.tensor([3])},
            {"weight": torch.tensor([2.0, 6.0]), "index": torch.tensor([3])},
            0.25,
        )
        self.assertTrue(torch.equal(merged["weight"], torch.tensor([0.5, 3.0])))
        self.assertTrue(torch.equal(merged["index"], torch.tensor([3])))

    def test_rejects_different_tensor_contracts(self) -> None:
        with self.assertRaisesRegex(ValueError, "keys differ"):
            interpolate_state_dicts({"a": torch.ones(1)}, {"b": torch.ones(1)}, 0.5)


if __name__ == "__main__":
    unittest.main()
