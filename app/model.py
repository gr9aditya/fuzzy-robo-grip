"""Explainable static two-finger grip estimator; no external dependencies.

The nominal total normal force is m*g/mu. Equal jaws each supply half.
For assumed upper mass error p and possible relative friction reduction q,
the analytical bound is m*g/mu * (1+p)/(1-q). The zero-order Sugeno system
interpolates this multiplier at a 3x3 grid, using product AND and overlapping
triangular/shoulder memberships. It is exact in p and piecewise linear in q.
Because 1/(1-q) is convex, interpolation is at least the analytical bound
throughout the supported domain. This is conservativeness within this model,
not physical validation or a guarantee against object damage.

Assumptions: static vertical holding, symmetric equal jaw contacts, identical
contact friction, no acceleration/rotation/moments, and an established grasp.
Uncertainty percentages are assumed bounds, not probabilities or confidence
levels. Profiles are demonstration assumptions, not measured material data.
"""

import math

GRAVITY = 9.81
ANCHORS = (0.0, 0.25, 0.5)
LABELS = ('Low', 'Medium', 'High')
RULE_MATRIX = tuple(tuple((1 + p) / (1 - q) for q in ANCHORS) for p in ANCHORS)


def profiles():
    """Return fresh editable demonstration profiles, with no empirical claims."""
    return [
        {
            'id': key,
            'name': name,
            'mass_g': mass,
            'mu': 0.4,
            'contact': contact,
            'note': 'Illustrative example only. Mass and contact friction are assumptions; replace them with values for your object and pads.',
        }
        for key, name, mass, contact in (
            ('glass', 'Empty drinking glass', 250.0, 'Glass surface against gripper pads (assumed)'),
            ('filled_glass', 'Filled drinking glass', 500.0, 'Glass surface against gripper pads (assumed)'),
            ('phone', 'Phone', 180.0, 'Phone surface against gripper pads (assumed)'),
            ('box', 'Small box', 750.0, 'Box surface against gripper pads (assumed)'),
        )
    ]


def _number(data, key, lo, hi, *, strict_lower=False):
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{key} must be a number.')
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f'{key} must be finite.')
    if value > hi or (value <= lo if strict_lower else value < lo):
        condition = 'greater than' if strict_lower else 'at least'
        raise ValueError(f'{key} must be {condition} {lo:g} and at most {hi:g}.')
    return value


def _text(data, key, maximum, *, required=False):
    value = data.get(key, '')
    if not isinstance(value, str):
        raise ValueError(f'{key} must be text.')
    value = ' '.join(value.split())
    if required and not value:
        raise ValueError(f'{key} must not be empty.')
    if len(value) > maximum:
        raise ValueError(f'{key} must be no longer than {maximum} characters.')
    return value


def memberships(relative_error):
    """Low/medium/high membership degrees for a bound in [0, 0.5]."""
    if isinstance(relative_error, bool) or not isinstance(relative_error, (int, float)):
        raise ValueError('Relative error must be a number.')
    if not math.isfinite(relative_error) or not 0 <= relative_error <= 0.5:
        raise ValueError('Relative error must be finite and between 0 and 0.5.')
    x = relative_error / 0.5
    return [max(0.0, 1 - 2*x), max(0.0, 1 - abs(2*x - 1)), max(0.0, 2*x - 1)]


def calculate(data):
    """Validate JSON-like inputs and return a complete JSON-safe calculation.

    Required: name, mass_g, mu, mass_uncertainty_pct, friction_uncertainty_pct.
    Optional: contact (text), limit_n (positive total normal force or None).
    Numeric strings and bools are rejected. Limit is a check, never a command
    clamp. The named object is a label; its name does not influence the physics.
    """
    if not isinstance(data, dict):
        raise ValueError('Inputs must be an object.')
    inputs = {
        'name': _text(data, 'name', 80, required=True),
        'mass_g': _number(data, 'mass_g', 1, 20000),
        'mu': _number(data, 'mu', 0.05, 1.5),
        'mass_uncertainty_pct': _number(data, 'mass_uncertainty_pct', 0, 50),
        'friction_uncertainty_pct': _number(data, 'friction_uncertainty_pct', 0, 50),
        'contact': _text(data, 'contact', 120),
        'limit_n': None,
    }
    if data.get('limit_n') is not None:
        inputs['limit_n'] = _number(data, 'limit_n', 0, 100000, strict_lower=True)

    p = inputs['mass_uncertainty_pct'] / 100
    q = inputs['friction_uncertainty_pct'] / 100
    mass_degrees = memberships(p)
    friction_degrees = memberships(q)
    rules = []
    for i, mass_label in enumerate(LABELS):
        for j, friction_label in enumerate(LABELS):
            activation = mass_degrees[i] * friction_degrees[j]
            consequent = RULE_MATRIX[i][j]
            rules.append({
                'mass_label': mass_label,
                'friction_label': friction_label,
                'consequent': consequent,
                'activation': activation,
                'contribution': activation * consequent,
            })
    denominator = sum(rule['activation'] for rule in rules)
    multiplier = sum(rule['contribution'] for rule in rules) / denominator
    weight_n = inputs['mass_g'] / 1000 * GRAVITY
    baseline_n = weight_n / inputs['mu']
    total_n = baseline_n * multiplier
    exact_bound_n = baseline_n * (1+p) / (1-q)
    limit = inputs['limit_n']
    tolerance = 1e-12 * max(1.0, total_n)
    if limit is None:
        status = 'unknown_limit'
        text = 'Damage limit unknown. This is a holding-force estimate, not a damage assessment.'
    elif limit < baseline_n - tolerance:
        status = 'below_baseline'
        text = 'The supplied limit is below the nominal static holding threshold. No force within that limit meets this model’s nominal holding requirement.'
    elif limit < total_n - tolerance:
        status = 'below_estimate'
        text = 'The supplied limit covers the nominal threshold but not the full fuzzy estimate with the assumed uncertainty allowance.'
    else:
        status = 'within_limit'
        text = 'The estimate is within the supplied limit. This comparison does not establish safe or damage-free real-world handling.'
    return {
        'inputs': inputs,
        'weight_n': weight_n,
        'baseline_n': baseline_n,
        'multiplier': multiplier,
        'total_n': total_n,
        'per_finger_n': total_n / 2,
        'exact_bound_n': exact_bound_n,
        'margin_pct': (multiplier - 1) * 100,
        'mass_memberships': mass_degrees,
        'friction_memberships': friction_degrees,
        'rules': rules,
        'status': status,
        'status_text': text,
        'rule_matrix': [list(row) for row in RULE_MATRIX],
    }
