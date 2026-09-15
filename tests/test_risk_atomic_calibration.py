import unittest

from kinesync.guard.schema import UpdateEvidence
from kinesync.external_gs.risk_atomic import RiskRecord, calibrate, commit


def evidence(gain, disagreement):
    return UpdateEvidence(2, gain, 0.5, 0.8, disagreement, 0.5, True)


class RiskAtomicTests(unittest.TestCase):
    def test_rejects_harmful_update_instead_of_maximizing_coverage(self):
        good = RiskRecord('dev_a', 'development', evidence(.99, .05), 1., .1)
        bad = RiskRecord('dev_b', 'development', evidence(.4, .8), 1., 4.)
        result = calibrate([good, bad])
        self.assertEqual(result['harmful_accepted'], 0)
        self.assertEqual(result['beneficial_accepted'], 1)
        self.assertEqual(commit([1., 2.], [.1, .2], good.evidence, result['thresholds'])[0], [.1, .2])
        self.assertEqual(commit([1., 2.], [4., 5.], bad.evidence, result['thresholds'])[0], [1., 2.])

    def test_refuses_heldout_calibration(self):
        with self.assertRaisesRegex(ValueError, 'development'):
            calibrate([RiskRecord('test_a', 'heldout', evidence(.9, .1), 1., .1)])

    def test_refuses_duplicate_states(self):
        row = RiskRecord('dev_a', 'development', evidence(.9, .1), 1., .1)
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            calibrate([row, row])

    def test_commit_never_partially_splices_joint_coordinates(self):
        result = calibrate([RiskRecord('dev_a', 'development', evidence(.9, .1), 1., .1)])
        final, _ = commit([1., 2., 3.], [.1, .2, .3], evidence(.9, .1), result['thresholds'])
        self.assertEqual(final, [.1, .2, .3])


if __name__ == '__main__':
    unittest.main()
