from __future__ import annotations

import math
import unittest

from core.silence_distribution import (
    build_truncated_weibull,
    target_mean_from_bounds,
)


class TruncatedWeibullTests(unittest.TestCase):
    def assert_distribution_mean(
        self,
        minimum: float,
        maximum: float,
        ratio: float,
        shape: float,
        expected: float,
    ) -> None:
        distribution = build_truncated_weibull(
            minimum,
            maximum,
            target_mean_from_bounds(minimum, maximum, ratio),
            shape,
        )
        sample_count = 2_000
        estimated = (
            sum(
                distribution.quantile((index + 0.5) / sample_count)
                for index in range(sample_count)
            )
            / sample_count
        )
        self.assertAlmostEqual(estimated, expected, places=2)

    def test_personal_target_mean_is_100_minutes(self) -> None:
        self.assertEqual(target_mean_from_bounds(40, 200, 0.375), 100)
        self.assert_distribution_mean(40, 200, 0.375, 1.7, 100)

    def test_upstream_defaults_follow_ratio(self) -> None:
        self.assertEqual(target_mean_from_bounds(30, 600, 0.375), 243.75)
        self.assert_distribution_mean(30, 600, 0.375, 1.7, 243.75)

    def test_boundaries_and_cdf_inverse(self) -> None:
        distribution = build_truncated_weibull(40, 200, 100, 1.7)
        self.assertEqual(distribution.cdf(39), 0)
        self.assertEqual(distribution.cdf(200), 1)
        self.assertEqual(distribution.quantile(0), 40)
        self.assertEqual(distribution.quantile(1), 200)
        for probability in (0.01, 0.1, 0.5, 0.9, 0.99):
            value = distribution.quantile(probability)
            self.assertGreaterEqual(value, 40)
            self.assertLessEqual(value, 200)
            self.assertAlmostEqual(distribution.cdf(value), probability, places=9)

    def test_conditional_resample_is_strictly_later(self) -> None:
        distribution = build_truncated_weibull(40, 200, 100, 1.7)
        for draw in (0.0, 0.01, 0.5, 0.99, 1.0):
            sampled = distribution.sample(lambda draw=draw: draw, after=125)
            self.assertGreater(sampled, 125)
            self.assertLessEqual(sampled, 200)
        self.assertEqual(distribution.sample(lambda: 0.5, after=200), 200)

    def test_invalid_target_falls_back_without_becoming_unusable(self) -> None:
        distribution = build_truncated_weibull(40, 200, 325, 1.7)
        self.assertTrue(distribution.used_mean_fallback)
        self.assertEqual(distribution.requested_mean, 325)
        self.assertEqual(distribution.target_mean, 120)
        self.assertTrue(math.isfinite(distribution.scale))
        self.assertGreater(distribution.sample(lambda: 0.5), 40)
        self.assertLess(distribution.sample(lambda: 0.5), 200)

    def test_all_exposed_slider_combinations_are_valid(self) -> None:
        for shape_index in range(6):
            shape = 1.5 + shape_index * 0.1
            for ratio_index in range(16):
                ratio = 0.2 + ratio_index * 0.025
                distribution = build_truncated_weibull(
                    40,
                    200,
                    target_mean_from_bounds(40, 200, ratio),
                    shape,
                )
                self.assertFalse(
                    distribution.used_mean_fallback,
                    msg=f"shape={shape}, ratio={ratio}",
                )

    def test_invalid_bounds_and_shape_are_safely_normalized(self) -> None:
        distribution = build_truncated_weibull(-5, -10, float("nan"), 0.5)
        self.assertEqual(distribution.minimum, 0)
        self.assertEqual(distribution.maximum, 1)
        self.assertEqual(distribution.shape, 1.7)
        self.assertTrue(math.isfinite(distribution.scale))


if __name__ == "__main__":
    unittest.main()
