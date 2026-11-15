import numpy as np
import pandas as pd
import pytest

from sidx import analysis as A


def test_boot_ci_contains_mean_and_shrinks_with_n():
    rng = np.random.default_rng(0)
    m, lo, hi = A.boot_ci(rng.normal(5, 1, 20))
    assert lo < m < hi
    m2, lo2, hi2 = A.boot_ci(rng.normal(5, 1, 400))
    assert (hi2 - lo2) < (hi - lo)
    assert A.boot_ci([3.0]) == (3.0, 3.0, 3.0)
    assert all(np.isnan(A.boot_ci([])))


def test_add_derived_struct_dev_is_zero_for_perfect_ratios():
    df = pd.DataFrame({"endpoint_excess": [1.0, 2.0], "mean_seg_len_ratio": [1.0, 0.5],
                       "grey_ratio": [1.0, 1.0], "edge_density_ratio": [1.0, 1.0]})
    out = A.add_derived(df)
    assert out["struct_dev"].iloc[0] == pytest.approx(0.0)
    assert out["struct_dev"].iloc[1] == pytest.approx((1 + 1 + 0 + 0) / 4)      # |log2 2|, |log2 .5|


def test_paired_delta_uses_matched_images_only():
    df = pd.DataFrame({"job": ["base"] * 3 + ["abl"] * 2, "image_id": ["a", "b", "c", "a", "b"],
                       "m": [1.0, 2.0, 3.0, 2.0, 3.0]})
    d = A.paired_delta(df, "base", ["m"])
    assert d.loc[0, "n_images"] == 2 and d.loc[0, "delta_mean"] == pytest.approx(1.0)
    assert d.loc[0, "frac_increase"] == 1.0


def test_spearman_table_finds_monotone_relation():
    x = np.arange(30.0)
    df = pd.DataFrame({"p": x, "m": x ** 2, "noise": np.random.default_rng(0).normal(size=30)})
    t = A.spearman_table(df, ["p"], ["m", "noise"])
    assert t.iloc[0]["metric"] == "m" and t.iloc[0]["rho"] == pytest.approx(1.0)


def test_factor_effects_groups_by_factor_and_includes_base():
    df = pd.DataFrame({
        "factor": ["base", "thickness", "thickness", "length"],
        "thickness": [2.0, 1.0, 4.0, 2.0], "length": ["long", "long", "long", "short"],
        "m": [1.0, 2.0, 3.0, 9.0]})
    t = A.factor_effects(df, [], ["m"])
    th = t[t.factor == "thickness"].set_index("level")["m"].to_dict()
    assert th["1.0"] == 2.0 and th["4.0"] == 3.0 and th["2.0 (base)"] == 1.0
    assert t[(t.factor == "length") & (t.level == "short")]["m"].iloc[0] == 9.0


def test_markdown_table_shape():
    s = A.to_markdown(pd.DataFrame({"a": [1.5, float("nan")], "b": ["x", "y"]}))
    assert s.splitlines()[0] == "| a | b |" and "n/a" in s
