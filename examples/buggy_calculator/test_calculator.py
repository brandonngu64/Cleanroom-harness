import unittest

from calculator import add


class AdditionTests(unittest.TestCase):
    def test_positive_numbers(self):
        self.assertEqual(add(2, 3), 5)

    def test_negative_numbers(self):
        self.assertEqual(add(-2, -3), -5)

    def test_zero(self):
        self.assertEqual(add(0, 0), 0)

    def test_fractional_numbers(self):
        self.assertAlmostEqual(add(1.5, 2.25), 3.75)


if __name__ == "__main__":
    unittest.main()
