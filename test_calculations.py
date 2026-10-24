import unittest

from tools import calculate_total_changes


class TestCalculations(unittest.TestCase):

    def test_normal_changes(self):
        result = calculate_total_changes(20, 5)
        self.assertEqual(result, 25)

    def test_zero_changes(self):
        result = calculate_total_changes(0, 0)
        self.assertEqual(result, 0)

    def test_only_additions(self):
        result = calculate_total_changes(100, 0)
        self.assertEqual(result, 100)

    def test_only_deletions(self):
        result = calculate_total_changes(0, 50)
        self.assertEqual(result, 50)


if __name__ == "__main__":
    unittest.main(verbosity=2)
