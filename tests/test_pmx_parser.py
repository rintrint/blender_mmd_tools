# Copyright 2026 MMD Tools authors
# This file is part of MMD Tools.

import os
import shutil
import struct
import unittest

from bl_ext.blender_org.mmd_tools.core import pmx

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))


class TestPmxParserRobustness(unittest.TestCase):
    """The parser must reject malformed files (negative lengths/counts, truncation) with InvalidFileError instead of crashing with struct.error."""

    @classmethod
    def setUpClass(cls):
        cls.output_dir = os.path.join(TESTS_DIR, "output", "pmx_parser")
        os.makedirs(cls.output_dir, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.output_dir, ignore_errors=True)

    # ********************************************
    # Utils
    # ********************************************

    @staticmethod
    def __header():
        # sign, version 2.0, globals count 8, utf-16-le, 0 additional uvs, all index sizes 1 byte
        return b"PMX " + struct.pack("<f", 2.0) + bytes([8, 0, 0, 1, 1, 1, 1, 1, 1])

    @staticmethod
    def __str(text):
        data = text.encode("utf-16-le")
        return struct.pack("<i", len(data)) + data

    def __write_file(self, name, data):
        filepath = os.path.join(self.output_dir, name)
        with open(filepath, "wb") as f:
            f.write(data)
        return filepath

    # ********************************************
    # Test Cases
    # ********************************************

    def test_valid_minimal_model(self):
        body = self.__str("name") + self.__str("name_e") + self.__str("comment") + self.__str("comment_e")
        body += struct.pack("<i", 0) * 2  # vertices, faces
        body += struct.pack("<i", 1) + self.__str("tex\\a.png")  # textures
        body += struct.pack("<i", 0) * 6  # materials, bones, morphs, display, rigids, joints
        model = pmx.load(self.__write_file("valid_minimal.pmx", self.__header() + body))
        self.assertEqual(model.name, "name")
        self.assertEqual(len(model.textures), 1)

    def test_negative_string_length(self):
        data = self.__header() + struct.pack("<i", -1)
        with self.assertRaises(pmx.InvalidFileError):
            pmx.load(self.__write_file("negative_string_length.pmx", data))

    def test_string_longer_than_file(self):
        data = self.__header() + struct.pack("<i", 100) + b"abcd"
        with self.assertRaises(pmx.InvalidFileError):
            pmx.load(self.__write_file("string_longer_than_file.pmx", data))

    def test_truncated_file(self):
        data = self.__header() + b"\x01\x00"  # file ends inside the first length field
        with self.assertRaises(pmx.InvalidFileError):
            pmx.load(self.__write_file("truncated_file.pmx", data))

    def test_negative_count(self):
        data = self.__header() + self.__str("") * 4 + struct.pack("<i", -5)  # num_vertices = -5
        with self.assertRaises(pmx.InvalidFileError):
            pmx.load(self.__write_file("negative_count.pmx", data))

    def test_count_exceeding_file_size(self):
        data = self.__header() + self.__str("") * 4 + struct.pack("<i", 0x7FFFFFFF)
        with self.assertRaises(pmx.InvalidFileError):
            pmx.load(self.__write_file("count_exceeding_file_size.pmx", data))


if __name__ == "__main__":
    import sys

    sys.argv = [__file__] + (sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else [])
    unittest.main(verbosity=1, exit=True)
