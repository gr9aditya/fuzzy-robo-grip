"""Analytical and property checks for the grip estimator, not hardware tests."""
import json
import math
import unittest
from model import ANCHORS, GRAVITY, RULE_MATRIX, calculate, memberships, profiles


def example(**changes):
    values = dict(name='Example glass', mass_g=250, mu=.4,
                  mass_uncertainty_pct=10, friction_uncertainty_pct=25,
                  contact='Glass / pads (assumed)', limit_n=None)
    values.update(changes)
    return values


class ModelTests(unittest.TestCase):
    def test_analytical_rule_anchors(self):
        for i, p in enumerate(ANCHORS):
            for j, q in enumerate(ANCHORS):
                r = calculate(example(mass_uncertainty_pct=100*p, friction_uncertainty_pct=100*q))
                self.assertAlmostEqual(r['multiplier'], (1+p)/(1-q))
                self.assertAlmostEqual(r['total_n'], r['exact_bound_n'])
                self.assertEqual(sum(rule['activation'] > 0 for rule in r['rules']), 1)
                self.assertAlmostEqual(r['rules'][i*3+j]['consequent'], RULE_MATRIX[i][j])

    def test_hand_worked_default(self):
        r = calculate(example())
        self.assertAlmostEqual(r['weight_n'], 2.4525)
        self.assertAlmostEqual(r['baseline_n'], 6.13125)
        self.assertAlmostEqual(r['multiplier'], 1.1/.75)
        self.assertAlmostEqual(r['total_n'], 8.9925)
        self.assertAlmostEqual(r['per_finger_n'], 4.49625)
        self.assertAlmostEqual(sum(rule['activation'] for rule in r['rules']), 1)

    def test_partition_and_interior_conservativeness(self):
        for k in range(101):
            p = .5*k/100
            self.assertAlmostEqual(sum(memberships(p)), 1)
            self.assertTrue(all(0 <= degree <= 1 for degree in memberships(p)))
            for j in range(101):
                q = .5*j/100
                r = calculate(example(mass_uncertainty_pct=p*100, friction_uncertainty_pct=q*100))
                self.assertGreaterEqual(r['total_n']+1e-10, r['exact_bound_n'])
                self.assertGreaterEqual(r['multiplier'], 1-1e-12)
                self.assertLessEqual(r['multiplier'], 3+1e-12)

    def test_monotonicity(self):
        for fixed in range(0, 51, 2):
            by_mass = [calculate(example(mass_uncertainty_pct=x, friction_uncertainty_pct=fixed))['total_n'] for x in range(51)]
            by_friction = [calculate(example(mass_uncertainty_pct=fixed, friction_uncertainty_pct=x))['total_n'] for x in range(51)]
            for values in (by_mass, by_friction):
                self.assertTrue(all(b >= a-1e-10 for a, b in zip(values, values[1:])))

    def test_mass_and_friction_scaling(self):
        r = calculate(example())
        self.assertAlmostEqual(calculate(example(mass_g=500))['total_n'], 2*r['total_n'])
        self.assertAlmostEqual(calculate(example(mu=.8))['total_n'], r['total_n']/2)
        self.assertAlmostEqual(2*r['per_finger_n'], r['total_n'])
        # Object labels and limits do not secretly alter the physical estimate.
        self.assertEqual(calculate(example(name='Different object', limit_n=1))['total_n'], r['total_n'])

    def test_limit_boundaries(self):
        r = calculate(example())
        self.assertEqual(r['status'], 'unknown_limit')
        self.assertEqual(calculate(example(limit_n=r['baseline_n']-1e-5))['status'], 'below_baseline')
        self.assertEqual(calculate(example(limit_n=r['baseline_n']))['status'], 'below_estimate')
        self.assertEqual(calculate(example(limit_n=r['total_n']-1e-5))['status'], 'below_estimate')
        self.assertEqual(calculate(example(limit_n=r['total_n']))['status'], 'within_limit')
        zero_margin = calculate(example(mass_uncertainty_pct=0, friction_uncertainty_pct=0))
        self.assertEqual(calculate(example(mass_uncertainty_pct=0, friction_uncertainty_pct=0,
                                          limit_n=zero_margin['baseline_n']))['status'], 'within_limit')

    def test_invalid_values(self):
        invalid = {
            'name': ['', '   ', 'x'*81, 12, None],
            'contact': ['x'*121, None, True],
            'mass_g': [0, .99, -1, 20001, True, '250', None, math.nan, math.inf],
            'mu': [0, .049, 1.501, True, '.4', math.nan, math.inf],
            'mass_uncertainty_pct': [-1, 51, True, '10', None, math.nan],
            'friction_uncertainty_pct': [-1, 51, False, '25', math.inf],
            'limit_n': [0, -1, 100001, True, '10', math.nan, math.inf],
        }
        for key, values in invalid.items():
            for value in values:
                with self.subTest(key=key, value=value):
                    with self.assertRaises(ValueError):
                        calculate(example(**{key: value}))
        with self.assertRaises(ValueError):
            calculate([])
        for bad in (True, math.nan, math.inf, -.01, .501, '0.2'):
            with self.assertRaises(ValueError):
                memberships(bad)

    def test_json_profiles_and_sanitization(self):
        self.assertEqual(len(profiles()), 4)
        changed = profiles()
        changed[0]['mass_g'] = 999
        self.assertEqual(profiles()[0]['mass_g'], 250)
        for profile in profiles():
            r = calculate(example(name=profile['name'], mass_g=profile['mass_g'], mu=profile['mu'], contact=profile['contact']))
            json.dumps(r, allow_nan=False)
        r = calculate(example(name='  My\n glass  ', contact='  Glass / pads '))
        self.assertEqual(r['inputs']['name'], 'My glass')
        self.assertEqual(r['inputs']['contact'], 'Glass / pads')


if __name__ == '__main__':
    unittest.main()
