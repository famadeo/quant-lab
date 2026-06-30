import importlib.util
import pathlib

import numpy as np

SPEC = pathlib.Path("scripts/analyze_metals_convenience_yield_mechanism.py")
mod = importlib.util.module_from_spec(importlib.util.spec_from_file_location("cym", SPEC))
importlib.util.spec_from_file_location("cym", SPEC).loader.exec_module(mod)


def test_reconstructed_z_matches_event_log_entry_z():
    # Reconstructing the z-score from curve_panel must reproduce the entry_z
    # the backtest actually traded, for the best sync variant.
    zpanel, events, _params = mod.load_variant(mod.SOURCES["sync"], "target3m_minv10")
    merged = events.merge(
        zpanel[["root", "date", "carry_z"]],
        left_on=["root", "entry_date"], right_on=["root", "date"], how="left",
    )
    ok = merged.dropna(subset=["carry_z", "entry_z"])
    assert len(ok) > 50  # noqa: PLR2004
    assert np.allclose(ok["carry_z"], ok["entry_z"], atol=1e-6)


def test_load_variant_pins_single_best_variant():
    # load_variant must resolve each short name to exactly ONE full variant
    # string and return only that variant's events (no double-counting across
    # the six entry_z/side_mode sub-variants sharing the short prefix).
    for short in mod.VARIANTS:
        _zpanel, events, params = mod.load_variant(mod.SOURCES["sync"], short)
        assert events["variant"].nunique() == 1
        assert params["variant"] == events["variant"].iloc[0]

    _zpanel, events, params = mod.load_variant(mod.SOURCES["sync"], "target3m_minv10")
    assert params["variant"] == "target3m_minv10_lb126_entry1p5_exit0p25_both_costx1p0"
    assert len(events) == 196  # noqa: PLR2004
