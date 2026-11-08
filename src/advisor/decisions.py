"""Decision wording; thresholds are display heuristics, not confidence intervals."""

SMALL_TRANSFER_EDGE = 2.0


def transfer_caution(net, hits=0):
    """Flag a small total horizon gain after hits without changing the optimiser."""
    if net > SMALL_TRANSFER_EDGE:
        return None
    message = (
        "Small estimated edge · uncertain decision. "
        "The gain over the whole forecast window could be erased by forecast error. "
        "Keeping the player and transfer flexibility is a reasonable alternative."
    )
    if hits:
        message += " This move also spends points on a hit."
    return message
