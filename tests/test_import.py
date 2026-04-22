import unittest


class CasbenchImportTest(unittest.TestCase):
    def test_import_and_version(self) -> None:
        import casbench

        self.assertEqual(casbench.__version__, "0.1.0")


if __name__ == "__main__":
    unittest.main()
