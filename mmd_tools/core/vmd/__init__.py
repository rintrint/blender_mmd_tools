# Copyright 2014 MMD Tools authors
# This file is part of MMD Tools.

import collections
import logging
import struct

import numpy as np


class InvalidFileError(Exception):
    pass


def _decodeCp932String(byteString):
    """Convert a VMD format byte string to a regular string."""
    # If the first byte is replaced with b"\x00" during encoding, add � at the beginning during decoding
    # and replace ? with � to ensure replacement character consistency between UnicodeEncodeError and UnicodeDecodeError.
    # UnicodeEncodeError: Bone/Morph name has characters not supported by cp932 encoding. Default replacement character: ?
    # UnicodeDecodeError: Bone/Morph name was truncated at 15 bytes, breaking character boundaries. Default replacement character: �(U+FFFD)
    decoded = byteString.replace(b"\x00", b"").decode("cp932", errors="replace")
    if byteString[:1] == b"\x00":
        decoded = "\ufffd" + decoded.replace("?", "\ufffd")
    return decoded


def _encodeCp932String(string):
    """Convert a regular string to a VMD format byte string."""
    try:
        return string.encode("cp932")
    except UnicodeEncodeError:
        # Match MikuMikuDance's behavior: replace first byte with b"\x00" to indicate encoding failures
        return b"\x00" + string.encode("cp932", errors="replace")[1:]


class Header:
    VMD_SIGN = b"Vocaloid Motion Data 0002"

    def __init__(self):
        self.signature = None
        self.model_name = ""

    def load(self, fin):
        (self.signature,) = struct.unpack("<30s", fin.read(30))
        if self.signature[: len(self.VMD_SIGN)] != self.VMD_SIGN:
            raise InvalidFileError('File signature "%s" is invalid.' % self.signature)
        self.model_name = _decodeCp932String(struct.unpack("<20s", fin.read(20))[0])

    def save(self, fin):
        fin.write(struct.pack("<30s", self.VMD_SIGN))
        fin.write(struct.pack("<20s", _encodeCp932String(self.model_name)))

    def __repr__(self):
        return "<Header model_name %s>" % (self.model_name)


class BoneFrameKey:
    def __init__(self):
        self.frame_number = 0
        self.location = []
        self.rotation = []
        self.interp = []

    def load(self, fin):
        (self.frame_number,) = struct.unpack("<L", fin.read(4))
        self.location = list(struct.unpack("<fff", fin.read(4 * 3)))
        self.rotation = list(struct.unpack("<ffff", fin.read(4 * 4)))
        if not any(self.rotation):
            self.rotation = (0, 0, 0, 1)
        self.interp = list(struct.unpack("<64b", fin.read(64)))

    def save(self, fin):
        fin.write(struct.pack("<L", self.frame_number))
        fin.write(struct.pack("<fff", *self.location))
        fin.write(struct.pack("<ffff", *self.rotation))
        fin.write(struct.pack("<64b", *self.interp))

    def __repr__(self):
        return "<BoneFrameKey frame %s, loa %s, rot %s>" % (
            str(self.frame_number),
            str(self.location),
            str(self.rotation),
        )


class ShapeKeyFrameKey:
    def __init__(self):
        self.frame_number = 0
        self.weight = 0.0

    def load(self, fin):
        (self.frame_number,) = struct.unpack("<L", fin.read(4))
        (self.weight,) = struct.unpack("<f", fin.read(4))

    def save(self, fin):
        fin.write(struct.pack("<L", self.frame_number))
        fin.write(struct.pack("<f", self.weight))

    def __repr__(self):
        return "<ShapeKeyFrameKey frame %s, weight %s>" % (
            str(self.frame_number),
            str(self.weight),
        )


class CameraKeyFrameKey:
    def __init__(self):
        self.frame_number = 0
        self.distance = 0.0
        self.location = []
        self.rotation = []
        self.interp = []
        self.angle = 0.0
        self.persp = True

    def load(self, fin):
        (self.frame_number,) = struct.unpack("<L", fin.read(4))
        (self.distance,) = struct.unpack("<f", fin.read(4))
        self.location = list(struct.unpack("<fff", fin.read(4 * 3)))
        self.rotation = list(struct.unpack("<fff", fin.read(4 * 3)))
        self.interp = list(struct.unpack("<24b", fin.read(24)))
        (self.angle,) = struct.unpack("<L", fin.read(4))
        (self.persp,) = struct.unpack("<b", fin.read(1))
        self.persp = self.persp == 0

    def save(self, fin):
        fin.write(struct.pack("<L", self.frame_number))
        fin.write(struct.pack("<f", self.distance))
        fin.write(struct.pack("<fff", *self.location))
        fin.write(struct.pack("<fff", *self.rotation))
        fin.write(struct.pack("<24b", *self.interp))
        fin.write(struct.pack("<L", self.angle))
        fin.write(struct.pack("<b", 0 if self.persp else 1))

    def __repr__(self):
        return "<CameraKeyFrameKey frame %s, distance %s, loc %s, rot %s, angle %s, persp %s>" % (
            str(self.frame_number),
            str(self.distance),
            str(self.location),
            str(self.rotation),
            str(self.angle),
            str(self.persp),
        )


class LampKeyFrameKey:
    def __init__(self):
        self.frame_number = 0
        self.color = []
        self.direction = []

    def load(self, fin):
        (self.frame_number,) = struct.unpack("<L", fin.read(4))
        self.color = list(struct.unpack("<fff", fin.read(4 * 3)))
        self.direction = list(struct.unpack("<fff", fin.read(4 * 3)))

    def save(self, fin):
        fin.write(struct.pack("<L", self.frame_number))
        fin.write(struct.pack("<fff", *self.color))
        fin.write(struct.pack("<fff", *self.direction))

    def __repr__(self):
        return "<LampKeyFrameKey frame %s, color %s, direction %s>" % (
            str(self.frame_number),
            str(self.color),
            str(self.direction),
        )


class SelfShadowFrameKey:
    def __init__(self):
        self.frame_number = 0
        self.mode = 0  # 0: none, 1: mode1, 2: mode2
        self.distance = 0.0

    def load(self, fin):
        (self.frame_number,) = struct.unpack("<L", fin.read(4))
        (self.mode,) = struct.unpack("<b", fin.read(1))
        if self.mode not in range(3):
            logging.warning(" * Invalid self shadow mode %d at frame %d", self.mode, self.frame_number)
            raise struct.error
        (distance,) = struct.unpack("<f", fin.read(4))
        self.distance = 10000 - distance * 100000
        logging.info("    %s", self)

    def save(self, fin):
        fin.write(struct.pack("<L", self.frame_number))
        fin.write(struct.pack("<b", self.mode))
        distance = (10000 - self.distance) / 100000
        fin.write(struct.pack("<f", distance))

    def __repr__(self):
        return "<SelfShadowFrameKey frame %s, mode %s, distance %s>" % (
            str(self.frame_number),
            str(self.mode),
            str(self.distance),
        )


class PropertyFrameKey:
    def __init__(self):
        self.frame_number = 0
        self.visible = True
        self.ik_states = []  # list of (ik_name, enable/disable)

    def load(self, fin):
        (self.frame_number,) = struct.unpack("<L", fin.read(4))
        (self.visible,) = struct.unpack("<b", fin.read(1))
        (count,) = struct.unpack("<L", fin.read(4))
        for i in range(count):
            ik_name = _decodeCp932String(struct.unpack("<20s", fin.read(20))[0])
            (state,) = struct.unpack("<b", fin.read(1))
            self.ik_states.append((ik_name, state))

    def save(self, fin):
        fin.write(struct.pack("<L", self.frame_number))
        fin.write(struct.pack("<b", 1 if self.visible else 0))
        fin.write(struct.pack("<L", len(self.ik_states)))
        for ik_name, state in self.ik_states:
            fin.write(struct.pack("<20s", _encodeCp932String(ik_name)))
            fin.write(struct.pack("<b", 1 if state else 0))

    def __repr__(self):
        return "<PropertyFrameKey frame %s, visible %s, ik_states %s>" % (
            str(self.frame_number),
            str(self.visible),
            str(self.ik_states),
        )


class _AnimationBase(collections.defaultdict):
    def __init__(self):
        collections.defaultdict.__init__(self, list)

    @staticmethod
    def frameClass():
        raise NotImplementedError

    def load(self, fin):
        (count,) = struct.unpack("<L", fin.read(4))
        logging.info("loading %s... %d", self.__class__.__name__, count)
        if count == 0:
            return

        # Load data directly as numpy arrays without creating Python objects
        optimized_dict = self._load_optimized_numpy(fin, count)

        # Use optimized numpy data
        for name, numpy_data in optimized_dict.items():
            self[name] = numpy_data

    def _load_optimized_numpy(self, fin, count):
        """Optimized loading: store numpy arrays directly without creating Python objects"""
        optimized_dict = {}

        # Determine data structure based on frame class
        frame_class = self.frameClass()

        if frame_class.__name__ == "BoneFrameKey":
            # VMD bone frame structure: name(15) + frame(4) + location(12) + rotation(16) + interp(64) = 111 bytes
            record_size = 111
            dtype = np.dtype(
                [
                    ("name", "S15"),  # 15 bytes: bone name
                    ("frame", "<u4"),  # 4 bytes: frame number
                    ("location", "<f4", (3,)),  # 12 bytes: x,y,z location
                    ("rotation", "<f4", (4,)),  # 16 bytes: x,y,z,w quaternion
                    ("interp", "u1", (64,)),  # 64 bytes: interpolation data
                ]
            )
        elif frame_class.__name__ == "ShapeKeyFrameKey":
            # VMD shape key frame structure: name(15) + frame(4) + weight(4) = 23 bytes
            record_size = 23
            dtype = np.dtype(
                [
                    ("name", "S15"),  # 15 bytes: shape key name
                    ("frame", "<u4"),  # 4 bytes: frame number
                    ("weight", "<f4"),  # 4 bytes: weight value
                ]
            )
        else:
            # Fallback for unknown frame types
            raise NotImplementedError(f"Unknown frame class: {frame_class.__name__}")

        # Calculate total bytes needed
        total_bytes = count * record_size

        # Read all raw data at once
        raw_bytes = fin.read(total_bytes)

        # Verify we read the expected amount of data
        if len(raw_bytes) != total_bytes:
            raise ValueError(f"Expected to read {total_bytes} bytes, but only got {len(raw_bytes)} bytes")

        # Convert raw bytes to structured numpy array
        structured_data = np.frombuffer(raw_bytes, dtype=dtype)

        # Verify the array has the expected number of records
        if len(structured_data) != count:
            raise ValueError(f"Expected {count} records, but got {len(structured_data)} records")

        # Extract names for grouping
        names_view = structured_data["name"]

        # Sort by name for efficient grouping
        sort_indices = np.lexsort((names_view,))
        sorted_names = names_view[sort_indices]

        # Find group boundaries efficiently
        name_changes = np.concatenate(([True], sorted_names[1:] != sorted_names[:-1]))
        boundaries = np.where(name_changes)[0]
        boundaries = np.append(boundaries, len(sorted_names))

        # Build name decoding cache to avoid repeated decoding
        unique_name_indices = boundaries[:-1]
        name_decode_cache = {}
        for boundary_idx in unique_name_indices:
            name_bytes = sorted_names[boundary_idx]
            if name_bytes not in name_decode_cache:
                decoded_name = _decodeCp932String(name_bytes)
                name_decode_cache[name_bytes] = decoded_name

        # Process each group and store numpy arrays directly
        for i in range(len(boundaries) - 1):
            start_idx = boundaries[i]
            end_idx = boundaries[i + 1]

            # Get all original indices for this bone/shape key
            group_indices = sort_indices[start_idx:end_idx]
            name_bytes = sorted_names[start_idx]
            decoded_name = name_decode_cache[name_bytes]

            # Store numpy array directly without creating Python objects
            group_numpy_data = structured_data[group_indices]
            optimized_dict[decoded_name] = group_numpy_data

        return optimized_dict

    def save(self, fin):
        """Save animation data, handling both numpy and traditional formats"""
        # Calculate total count
        count = 0
        for name, data in self.items():
            count += len(data)

        # Write count
        fin.write(struct.pack("<L", count))

        # Write data
        for name, frameKeys in self.items():
            name_data = struct.pack("<15s", _encodeCp932String(name))

            if isinstance(frameKeys, np.ndarray):
                # Handle numpy array format
                for i in range(len(frameKeys)):
                    fin.write(name_data)
                    frame_data = frameKeys[i]

                    # Write frame data based on type
                    if self.frameClass().__name__ == "BoneFrameKey":
                        # Write bone frame data
                        fin.write(struct.pack("<L", int(frame_data["frame"])))
                        fin.write(struct.pack("<fff", *frame_data["location"]))
                        fin.write(struct.pack("<ffff", *frame_data["rotation"]))
                        fin.write(struct.pack("<64b", *frame_data["interp"]))
                    elif self.frameClass().__name__ == "ShapeKeyFrameKey":
                        # Write shape key frame data
                        fin.write(struct.pack("<L", int(frame_data["frame"])))
                        fin.write(struct.pack("<f", float(frame_data["weight"])))
            else:
                # Handle traditional format
                for frameKey in frameKeys:
                    fin.write(name_data)
                    frameKey.save(fin)


class _AnimationListBase(list):
    def __init__(self):
        list.__init__(self)

    @staticmethod
    def frameClass():
        raise NotImplementedError

    def load(self, fin):
        (count,) = struct.unpack("<L", fin.read(4))
        logging.info("loading %s... %d", self.__class__.__name__, count)
        for i in range(count):
            cls = self.frameClass()
            frameKey = cls()
            frameKey.load(fin)
            self.append(frameKey)

    def save(self, fin):
        fin.write(struct.pack("<L", len(self)))
        for frameKey in self:
            frameKey.save(fin)


class BoneAnimation(_AnimationBase):
    def __init__(self):
        _AnimationBase.__init__(self)

    @staticmethod
    def frameClass():
        return BoneFrameKey


class ShapeKeyAnimation(_AnimationBase):
    def __init__(self):
        _AnimationBase.__init__(self)

    @staticmethod
    def frameClass():
        return ShapeKeyFrameKey


class CameraAnimation(_AnimationListBase):
    def __init__(self):
        _AnimationListBase.__init__(self)

    @staticmethod
    def frameClass():
        return CameraKeyFrameKey


class LampAnimation(_AnimationListBase):
    def __init__(self):
        _AnimationListBase.__init__(self)

    @staticmethod
    def frameClass():
        return LampKeyFrameKey


class SelfShadowAnimation(_AnimationListBase):
    def __init__(self):
        _AnimationListBase.__init__(self)

    @staticmethod
    def frameClass():
        return SelfShadowFrameKey


class PropertyAnimation(_AnimationListBase):
    def __init__(self):
        _AnimationListBase.__init__(self)

    @staticmethod
    def frameClass():
        return PropertyFrameKey


class File:
    def __init__(self):
        self.filepath = None
        self.header = None
        self.boneAnimation = None
        self.shapeKeyAnimation = None
        self.cameraAnimation = None
        self.lampAnimation = None
        self.selfShadowAnimation = None
        self.propertyAnimation = None

    def load(self, **args):
        path = args["filepath"]

        with open(path, "rb") as fin:
            self.filepath = path
            self.header = Header()
            self.boneAnimation = BoneAnimation()
            self.shapeKeyAnimation = ShapeKeyAnimation()
            self.cameraAnimation = CameraAnimation()
            self.lampAnimation = LampAnimation()
            self.selfShadowAnimation = SelfShadowAnimation()
            self.propertyAnimation = PropertyAnimation()

            self.header.load(fin)
            try:
                self.boneAnimation.load(fin)
                self.shapeKeyAnimation.load(fin)
                self.cameraAnimation.load(fin)
                self.lampAnimation.load(fin)
                self.selfShadowAnimation.load(fin)
                self.propertyAnimation.load(fin)
            except struct.error:
                pass  # no valid camera/lamp data

    def save(self, **args):
        path = args.get("filepath", self.filepath)

        header = self.header or Header()
        boneAnimation = self.boneAnimation or BoneAnimation()
        shapeKeyAnimation = self.shapeKeyAnimation or ShapeKeyAnimation()
        cameraAnimation = self.cameraAnimation or CameraAnimation()
        lampAnimation = self.lampAnimation or LampAnimation()
        selfShadowAnimation = self.selfShadowAnimation or SelfShadowAnimation()
        propertyAnimation = self.propertyAnimation or PropertyAnimation()

        with open(path, "wb") as fin:
            header.save(fin)
            boneAnimation.save(fin)
            shapeKeyAnimation.save(fin)
            cameraAnimation.save(fin)
            lampAnimation.save(fin)
            selfShadowAnimation.save(fin)
            propertyAnimation.save(fin)
