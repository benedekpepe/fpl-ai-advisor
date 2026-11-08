"""Small forecast edges must not be worded as certain gains."""
import pytest
from src.advisor.decisions import transfer_caution

@pytest.mark.parametrize("net", [0.01,1.4,2.0])
def test_small_total_edge_has_uncertainty_note(net):
    assert "uncertain decision" in transfer_caution(net)
    assert "whole forecast window" in transfer_caution(net)

def test_threshold_is_not_applied_to_larger_gains():
    assert transfer_caution(2.01) is None

def test_small_gain_after_a_hit_highlights_spending_points():
    assert "hit" in transfer_caution(1.5,hits=1)
