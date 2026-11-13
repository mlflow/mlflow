"""Statistical tests and confidence intervals on paired differences."""

import inspect
import logging
import math
import warnings
from typing import TYPE_CHECKING, Literal

# This module is imported by `import mlflow`, which must work on `mlflow-skinny`, so the data
# science libraries are only imported inside the functions that use them.
if TYPE_CHECKING:
    import numpy as np

_logger = logging.getLogger(__name__)

# Below this many discordant pairs the chi-square approximation of McNemar's test is
# unreliable, so the exact binomial test is used instead.
_MCNEMAR_EXACT_MAX_DISCORDANT = 25
# Up to this many non-zero differences the null distribution of the signed-rank statistic is
# enumerated exactly; above it the normal approximation is used.
_WILCOXON_EXACT_MAX_NONZERO = 50
_BOOTSTRAP_SEED = 0
_BOOTSTRAP_RESAMPLES = 10_000
# Bounds the memory of one bootstrap batch (resamples x pairs).
_BOOTSTRAP_MAX_BATCH_ELEMENTS = 1_000_000


def _bootstrap_mean_ci(deltas: "np.ndarray", confidence_level: float) -> tuple[float, float]:
    """
    BCa bootstrap confidence interval of the mean of ``deltas``, which must not be constant
    (the BCa acceleration is undefined then). Falls back to the percentile interval in the
    rare case that BCa still yields a non-finite bound.
    """
    import numpy as np
    from scipy import stats

    # scipy 1.15 renamed the seed argument from `random_state` to `rng`.
    seed_kwarg = "rng" if "rng" in inspect.signature(stats.bootstrap).parameters else "random_state"

    def bootstrap(method: str) -> tuple[float, float]:
        with warnings.catch_warnings():
            # A degenerate BCa interval is handled below. Its warnings add nothing.
            warnings.simplefilter("ignore", RuntimeWarning)
            warnings.simplefilter("ignore", stats.DegenerateDataWarning)
            interval = stats.bootstrap(
                (deltas,),
                np.mean,
                method=method,
                confidence_level=confidence_level,
                n_resamples=_BOOTSTRAP_RESAMPLES,
                batch=max(1, _BOOTSTRAP_MAX_BATCH_ELEMENTS // len(deltas)),
                **{seed_kwarg: np.random.default_rng(_BOOTSTRAP_SEED)},
            ).confidence_interval
        return float(interval.low), float(interval.high)

    low, high = bootstrap("BCa")
    if not (math.isfinite(low) and math.isfinite(high)):
        _logger.warning(
            "The BCa bootstrap confidence interval is undefined for this sample; reporting "
            "the percentile bootstrap interval instead."
        )
        low, high = bootstrap("percentile")
    return low, high


def _mcnemar(deltas: "np.ndarray") -> tuple[Literal["mcnemar_exact", "mcnemar_chi2"], float]:
    import numpy as np
    from scipy import stats

    improved = int(np.sum(deltas > 0))
    regressed = int(np.sum(deltas < 0))
    discordant = improved + regressed
    if discordant == 0:
        return "mcnemar_exact", 1.0
    if discordant < _MCNEMAR_EXACT_MAX_DISCORDANT:
        return "mcnemar_exact", float(stats.binomtest(improved, discordant, 0.5).pvalue)
    statistic = (abs(improved - regressed) - 1) ** 2 / discordant
    return "mcnemar_chi2", float(stats.chi2.sf(statistic, df=1))


def _wilcoxon(
    deltas: "np.ndarray",
) -> tuple[Literal["wilcoxon_exact", "wilcoxon_asymptotic"], float]:
    import numpy as np
    from scipy import stats

    # scipy's ``method="auto"`` switches between the exact distribution, a permutation test
    # (randomized above 13 pairs) and the normal approximation depending on the sample
    # size, ties and the scipy version, so one extra row could move the p-value to another
    # regime. The choice is pinned on the number of non-zero differences instead. The exact
    # distribution is conservative when absolute differences tie. scipy >= 1.13 calls the
    # normal approximation "asymptotic" and keeps "approx" as an alias; older versions
    # only know "approx".
    n_nonzero = int(np.sum(deltas != 0))
    exact = n_nonzero <= _WILCOXON_EXACT_MAX_NONZERO
    method = "wilcoxon_exact" if exact else "wilcoxon_asymptotic"
    # The signed-rank test discards zero differences and is undefined when all are zero.
    # They are removed up front because older scipy versions (1.13 and earlier) fall back
    # to the normal approximation, with a warning, when the exact method is given zeros.
    p_value = (
        float(stats.wilcoxon(deltas[deltas != 0], method="exact" if exact else "approx").pvalue)
        if n_nonzero
        else 1.0
    )
    return method, p_value
