import unittest

from OpenMuse.muse import MuseS


class EncodeCommandTests(unittest.TestCase):
    def test_basic_encoding(self):
        self.assertEqual(MuseS.encode_command("h"), b"\x02h\n")

    def test_rejects_empty_token(self):
        with self.assertRaises(ValueError):
            MuseS.encode_command("")

    def test_rejects_non_ascii(self):
        with self.assertRaises(ValueError):
            MuseS.encode_command("é")

    def test_rejects_too_long(self):
        long_token = "a" * 255
        with self.assertRaises(ValueError):
            MuseS.encode_command(long_token)


if __name__ == "__main__":
    unittest.main()
