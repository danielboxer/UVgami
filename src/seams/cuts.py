# an exposed vertex repels a cut a tenth as hard as a painted one, so it only breaks ties
VISIBILITY_SHARE = 0.1


# painted weights win where both apply, exposure is 1 on open surface and 0 in a crevice
def restriction_weights(painted, exposure):
    weights = {v: VISIBILITY_SHARE * e for v, e in (exposure or {}).items() if e > 0}
    for v, w in (painted or {}).items():
        weights[v] = max(weights.get(v, 0.0), w)
    return weights or None
