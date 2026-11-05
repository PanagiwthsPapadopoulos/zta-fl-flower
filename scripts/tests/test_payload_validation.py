"""Malformed model updates must fail atomically, before changing a live model."""
import copy
import unittest

import numpy as np
import torch
from torch import nn
from flwr.common import Parameters, parameters_to_ndarrays

from src.shared.network.model_payload import (
    decode_model, encode_model, load_model_arrays, validate_region_result, validate_sample_count,
)


class PayloadValidationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = nn.Sequential(nn.Linear(2, 3), nn.BatchNorm1d(3), nn.Linear(3, 2))
        self.original = copy.deepcopy(self.model.state_dict())

    def reject(self, arrays, bits):
        with self.assertRaises(ValueError):
            load_model_arrays(self.model, arrays, bits)
        self.assertTrue(all(torch.equal(v, self.original[k]) for k, v in self.model.state_dict().items()))

    def test_bad_count_shape_dtype_and_nonfinite_arrays_are_atomic(self):
        good = parameters_to_ndarrays(encode_model(self.model, 32))
        self.reject(good[:-1], 32)
        self.reject(good + [np.zeros(1)], 32)
        for replacement in (np.zeros((7, 7)), np.array(['bad']), np.array([1j]),
                            np.full_like(good[-1], np.nan), np.full_like(good[-1], np.inf),
                            np.full_like(good[-1], -np.inf), np.full(good[-1].shape, 1e300)):
            arrays = [v.copy() for v in good]
            arrays[0] += 7  # Detect any partial application before the bad last tensor.
            arrays[-1] = replacement
            with self.subTest(dtype=replacement.dtype, shape=replacement.shape):
                self.reject(arrays, 32)

    def test_invalid_int8_metadata_and_encoded_dtype_are_atomic(self):
        good = parameters_to_ndarrays(encode_model(self.model, 8))
        for meta in (np.array([0., 0.]), np.array([0., -1.]), np.array([0.]),
                     np.array([np.nan, 1.]), np.array([0., np.inf])):
            arrays = [v.copy() for v in good]
            arrays[1][0] += 7
            arrays[-1] = meta
            with self.subTest(meta=meta): self.reject(arrays, 8)
        arrays = [v.copy() for v in good]
        arrays[-2] = arrays[-2].astype(np.int16)
        self.reject(arrays, 8)
        self.reject(good, 7)

    def test_negative_variance_and_invalid_counter_are_atomic(self):
        good = parameters_to_ndarrays(encode_model(self.model, 32))
        for name, value in [('1.running_var', -1.), ('1.num_batches_tracked', -1),
                            ('1.num_batches_tracked', 1e30)]:
            arrays = [v.copy() for v in good]
            index = list(self.original).index(name)
            arrays[index] = np.full(arrays[index].shape, value)
            with self.subTest(name=name, value=value): self.reject(arrays, 32)

    def test_corrupt_flower_blob_does_not_modify_reference(self):
        with self.assertRaisesRegex(ValueError, 'Malformed serialized'):
            decode_model(self.model, Parameters(tensors=[b'not a numpy tensor'], tensor_type='numpy.ndarray'), 8)
        self.assertTrue(all(torch.equal(v, self.original[k]) for k, v in self.model.state_dict().items()))

    def test_sample_counts_must_be_positive_integers(self):
        for value in (0, -1, True, 1.5, float('nan'), '4'):
            with self.subTest(value=value), self.assertRaises(ValueError): validate_sample_count(value)
        validate_sample_count(1)
        validate_sample_count(np.int64(4))

    def test_empty_region_status_cannot_hide_a_model_or_positive_weight(self):
        for status in ('all_rejected', 'no_results', 'no_positive_trust', 'error'):
            self.assertEqual(validate_region_result(0, False, {'aggregation_status': status, 'total_regional_trust': 0}), status)
            for count, present, trust in ((1, False, 0), (0, True, 0), (0, False, 1)):
                with self.subTest(status=status, count=count, present=present, trust=trust), self.assertRaises(ValueError):
                    validate_region_result(count, present, {'aggregation_status': status, 'total_regional_trust': trust})
        for trust in (0, -1, float('nan'), float('inf')):
            with self.subTest(trust=trust), self.assertRaises(ValueError):
                validate_region_result(1, True, {'aggregation_status': 'aggregated', 'total_regional_trust': trust})


if __name__ == '__main__':
    unittest.main()
