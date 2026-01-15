import unittest


class TestPackage(unittest.TestCase):
    def test_import(self):
        import tiamat

        assert tiamat is not None

    def test_version(self):
        import tiamat

        self.assertTrue(hasattr(tiamat, "__version__"))
