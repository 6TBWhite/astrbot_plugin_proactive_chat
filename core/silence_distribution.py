"""私聊沉默时间分布。

这里实现移位并截断到有限区间的 Weibull 分布。配置中的目标均值是
截断后的真实均值，而不是截断前的理论均值。
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache

DEFAULT_WEIBULL_SHAPE = 1.7
DEFAULT_MEAN_POSITION_RATIO = 0.375
_INTEGRATION_STEPS = 512
_BISECTION_STEPS = 64
_PROBABILITY_EPSILON = 1e-12


@dataclass(frozen=True, slots=True)
class TruncatedWeibull:
    """一个定义在 ``[minimum, maximum]`` 上的截断 Weibull 分布。"""

    minimum: float
    maximum: float
    target_mean: float
    shape: float
    scale: float
    requested_mean: float
    used_mean_fallback: bool = False

    @property
    def span(self) -> float:
        return self.maximum - self.minimum

    def cdf(self, value: float) -> float:
        """返回截断分布的累计概率。"""
        if value <= self.minimum:
            return 0.0
        if value >= self.maximum:
            return 1.0
        elapsed = value - self.minimum
        cap_probability = _base_cdf(self.span, self.scale, self.shape)
        if cap_probability <= 0:
            return min(1.0, max(0.0, elapsed / self.span))
        return min(
            1.0,
            max(
                0.0,
                _base_cdf(elapsed, self.scale, self.shape) / cap_probability,
            ),
        )

    def quantile(self, probability: float) -> float:
        """返回截断分布的分位点。"""
        if probability <= 0:
            return self.minimum
        if probability >= 1:
            return self.maximum
        cap_probability = _base_cdf(self.span, self.scale, self.shape)
        inner_probability = min(
            1.0 - _PROBABILITY_EPSILON,
            max(0.0, probability * cap_probability),
        )
        elapsed = self.scale * (-math.log1p(-inner_probability)) ** (1.0 / self.shape)
        return min(self.maximum, max(self.minimum, self.minimum + elapsed))

    def sample(
        self,
        random_value: Callable[[], float] | None = None,
        *,
        after: float | None = None,
    ) -> float:
        """采样一个时间。

        ``after`` 给出条件下界；传入时采样 ``T | T > after``。这用于心念
        克制后的剩余时间窗重抽，不会重新开启一整轮等待。
        """
        lower_bound = self.minimum if after is None else max(self.minimum, after)
        if lower_bound >= self.maximum:
            return self.maximum

        draw = random_value or random.random
        unit = min(
            1.0 - _PROBABILITY_EPSILON,
            max(_PROBABILITY_EPSILON, float(draw())),
        )
        lower_probability = self.cdf(lower_bound)
        conditional_probability = lower_probability + unit * (1.0 - lower_probability)
        sampled = self.quantile(conditional_probability)

        # 浮点精度可能在非常接近上界时把分位点舍入回 lower_bound。
        if sampled <= lower_bound:
            sampled = math.nextafter(lower_bound, self.maximum)
        return min(self.maximum, sampled)


def build_truncated_weibull(
    minimum: float,
    maximum: float,
    target_mean: float,
    shape: float = DEFAULT_WEIBULL_SHAPE,
) -> TruncatedWeibull:
    """校验配置并创建截断 Weibull 分布。

    Weibull 的形状参数必须大于 1，才能表达“沉默越久，越容易产生心念”。
    目标均值不合法或超出当前形状可达到的范围时，临时回退到区间中点；
    调用方可以通过 ``used_mean_fallback`` 记录警告，但不需要改写配置文件。
    """
    minimum_value = _finite_float(minimum, 30.0)
    maximum_value = _finite_float(maximum, 600.0)
    if minimum_value < 0:
        minimum_value = 0.0
    if maximum_value <= minimum_value:
        maximum_value = minimum_value + 1.0

    shape_value = _finite_float(shape, DEFAULT_WEIBULL_SHAPE)
    if shape_value <= 1.0:
        shape_value = DEFAULT_WEIBULL_SHAPE

    requested_mean = _finite_float(target_mean, (minimum_value + maximum_value) / 2.0)
    span = maximum_value - minimum_value
    feasible_maximum = minimum_value + (shape_value / (shape_value + 1.0)) * span
    margin = max(1e-7, span * 1e-9)
    used_fallback = not (
        minimum_value + margin < requested_mean < feasible_maximum - margin
    )
    effective_mean = (
        (minimum_value + maximum_value) / 2.0 if used_fallback else requested_mean
    )

    # shape > 1 时中点一定落在可达到的均值范围内。
    scale = _solve_scale_cached(
        minimum_value,
        maximum_value,
        effective_mean,
        shape_value,
    )
    return TruncatedWeibull(
        minimum=minimum_value,
        maximum=maximum_value,
        target_mean=effective_mean,
        shape=shape_value,
        scale=scale,
        requested_mean=requested_mean,
        used_mean_fallback=used_fallback,
    )


def target_mean_from_bounds(
    minimum: float,
    maximum: float,
    ratio: float = DEFAULT_MEAN_POSITION_RATIO,
) -> float:
    """按区间位置比例计算目标均值。

    默认比例来自 ``40-200`` 区间内的 ``100`` 分钟目标：
    ``(100 - 40) / (200 - 40) = 0.375``。
    """
    minimum_value = _finite_float(minimum, 30.0)
    maximum_value = _finite_float(maximum, 600.0)
    if maximum_value <= minimum_value:
        maximum_value = minimum_value + 1.0
    ratio_value = _finite_float(ratio, DEFAULT_MEAN_POSITION_RATIO)
    ratio_value = min(1.0, max(0.0, ratio_value))
    return minimum_value + ratio_value * (maximum_value - minimum_value)


def _finite_float(value: object, fallback: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    return number if math.isfinite(number) else fallback


def _base_cdf(elapsed: float, scale: float, shape: float) -> float:
    if elapsed <= 0:
        return 0.0
    power = (elapsed / scale) ** shape
    return -math.expm1(-power)


def _mean_for_scale(
    minimum: float,
    maximum: float,
    shape: float,
    scale: float,
) -> float:
    """用 Simpson 积分计算截断分布均值。"""
    span = maximum - minimum
    cap_probability = _base_cdf(span, scale, shape)
    if cap_probability <= 0:
        return minimum + (shape / (shape + 1.0)) * span

    step = span / _INTEGRATION_STEPS
    weighted_sum = 0.0
    for index in range(_INTEGRATION_STEPS + 1):
        elapsed = index * step
        truncated_cdf = _base_cdf(elapsed, scale, shape) / cap_probability
        survival = max(0.0, 1.0 - truncated_cdf)
        if index in {0, _INTEGRATION_STEPS}:
            weight = 1
        elif index % 2 == 0:
            weight = 2
        else:
            weight = 4
        weighted_sum += weight * survival
    return minimum + (step / 3.0) * weighted_sum


@lru_cache(maxsize=128)
def _solve_scale_cached(
    minimum: float,
    maximum: float,
    target_mean: float,
    shape: float,
) -> float:
    """二分求出令截断均值等于目标值的尺度参数。"""
    span = maximum - minimum
    low = max(1e-9, span * 1e-8)
    high = max(1.0, span)

    while _mean_for_scale(minimum, maximum, shape, high) < target_mean:
        high *= 2.0
        if high >= span * 1e8:
            break

    for _ in range(_BISECTION_STEPS):
        middle = (low + high) / 2.0
        current_mean = _mean_for_scale(minimum, maximum, shape, middle)
        if current_mean < target_mean:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0
