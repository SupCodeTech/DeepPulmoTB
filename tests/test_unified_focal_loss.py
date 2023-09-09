# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import unittest

import numpy as np
import torch
from parameterized import parameterized

from monai.losses import AsymmetricUnifiedFocalLoss

TEST_CASES = [
    [  # shape: (2, 1, 2, 2), (2, 1, 2, 2)
        {
            "y_pred": torch.tensor([[[[1.0, 0], [0, 1.0]]], [[[1.0, 0], [0, 1.0]]]]),
            "y_true": torch.tensor([[[[1.0, 0], [0, 1.0]]], [[[1.0, 0], [0, 1.0]]]]),
        },
        0.0,
    ],
    [  # shape: (2, 1, 2, 2), (2, 1, 2, 2)
        {
            "y_pred": torch.tensor([[[[1.0, 0], [0, 1.0]]], [[[1.0, 0], [0, 1.0]]]]),
            "y_true": torch.tensor([[[[1.0, 0], [0, 1.0]]], [[[1.0, 0], [0, 1.0]]]]),
        },
        0.0,
    ],
]


class TestAsymmetricUnifiedFocalLoss(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_result(self, input_data, expected_val):
        loss = AsymmetricUnifiedFocalLoss()
        result = loss(**input_data)
        np.testing.assert_allclose(result.detach().cpu().numpy(), expected_val, atol=1e-4, rtol=1e-4)

    def test_ill_shape(self):
        loss = AsymmetricUnifiedFocalLoss()
        with self.assertRaisesRegex(ValueError, ""):
            loss(torch.ones((2, 2, 2)), torch.ones((2, 2, 2, 2)))

    def test_with_cuda(self):
        loss = AsymmetricUnifiedFocalLoss()
        i = torch.tensor([[[[1.0, 0], [0, 1.0]]], [[[1.0, 0], [0, 1.0]]]])
        j = torch.tensor([[[[1.0, 0], [0, 1.0]]], [[[1.0, 0], [0, 1.0]]]])
        if torch.cuda.is_available():
            i = i.cuda()
            j = j.cuda()
        output = loss(i, j)
        print(output)
        np.testing.assert_allclose(output.detach().cpu().numpy(), 0.0, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
