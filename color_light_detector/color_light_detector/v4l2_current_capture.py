"""Native V4L2 capture with optional maximum mode negotiation, never controls."""

import ctypes as ct
import errno
import fcntl
import mmap
import os
import platform
import select
import stat
import sys
from dataclasses import dataclass
from fractions import Fraction

import cv2
import numpy as np


class Capability(ct.Structure):
    _fields_ = [("driver", ct.c_uint8 * 16), ("card", ct.c_uint8 * 32),
                ("bus_info", ct.c_uint8 * 32), ("version", ct.c_uint32),
                ("capabilities", ct.c_uint32), ("device_caps", ct.c_uint32),
                ("reserved", ct.c_uint32 * 3)]


class PixelFormat(ct.Structure):
    _fields_ = [(name, ct.c_uint32) for name in (
        "width", "height", "pixelformat", "field", "bytesperline", "sizeimage",
        "colorspace", "priv", "flags", "ycbcr_enc", "quantization", "xfer_func")]


class FormatUnion(ct.Union):
    # The kernel union includes pointers and therefore needs native pointer alignment.
    _fields_ = [("pix", PixelFormat), ("raw", ct.c_uint8 * 200),
                ("alignment", ct.c_void_p)]


class Format(ct.Structure):
    _fields_ = [("type", ct.c_uint32), ("fmt", FormatUnion)]


class RequestBuffers(ct.Structure):
    _fields_ = [("count", ct.c_uint32), ("type", ct.c_uint32),
                ("memory", ct.c_uint32), ("capabilities", ct.c_uint32),
                ("flags", ct.c_uint8), ("reserved", ct.c_uint8 * 3)]


class TimeVal(ct.Structure):
    _fields_ = [("seconds", ct.c_long), ("microseconds", ct.c_long)]


class TimeCode(ct.Structure):
    _fields_ = [("type", ct.c_uint32), ("flags", ct.c_uint32),
                ("frames", ct.c_uint8), ("seconds", ct.c_uint8),
                ("minutes", ct.c_uint8), ("hours", ct.c_uint8),
                ("userbits", ct.c_uint8 * 4)]


class BufferMemory(ct.Union):
    _fields_ = [("offset", ct.c_uint32), ("userptr", ct.c_ulong),
                ("planes", ct.c_void_p), ("fd", ct.c_int32)]


class Buffer(ct.Structure):
    _fields_ = [("index", ct.c_uint32), ("type", ct.c_uint32),
                ("bytesused", ct.c_uint32), ("flags", ct.c_uint32),
                ("field", ct.c_uint32), ("timestamp", TimeVal),
                ("timecode", TimeCode), ("sequence", ct.c_uint32),
                ("memory", ct.c_uint32), ("m", BufferMemory),
                ("length", ct.c_uint32), ("reserved2", ct.c_uint32),
                ("request_fd", ct.c_int32)]


class CaptureParameters(ct.Structure):
    _fields_ = [(name, ct.c_uint32) for name in (
        "capability", "capturemode", "numerator", "denominator", "extendedmode",
        "readbuffers")] + [("reserved", ct.c_uint32 * 4)]


class ParameterUnion(ct.Union):
    _fields_ = [("capture", CaptureParameters), ("raw", ct.c_uint8 * 200)]


class StreamParameters(ct.Structure):
    _fields_ = [("type", ct.c_uint32), ("parm", ParameterUnion)]


class FormatDescription(ct.Structure):
    _fields_ = [("index", ct.c_uint32), ("type", ct.c_uint32), ("flags", ct.c_uint32),
                ("description", ct.c_uint8 * 32), ("pixelformat", ct.c_uint32),
                ("mbus_code", ct.c_uint32), ("reserved", ct.c_uint32 * 3)]


class DiscreteSize(ct.Structure):
    _fields_ = [("width", ct.c_uint32), ("height", ct.c_uint32)]


class StepwiseSize(ct.Structure):
    _fields_ = [(name, ct.c_uint32) for name in (
        "min_width", "max_width", "step_width", "min_height", "max_height", "step_height")]


class SizeUnion(ct.Union):
    _fields_ = [("discrete", DiscreteSize), ("stepwise", StepwiseSize)]


class FrameSize(ct.Structure):
    _fields_ = [("index", ct.c_uint32), ("pixel_format", ct.c_uint32),
                ("type", ct.c_uint32), ("size", SizeUnion), ("reserved", ct.c_uint32 * 2)]


class FramePeriod(ct.Structure):
    _fields_ = [("numerator", ct.c_uint32), ("denominator", ct.c_uint32)]


class StepwisePeriod(ct.Structure):
    _fields_ = [("min", FramePeriod), ("max", FramePeriod), ("step", FramePeriod)]


class PeriodUnion(ct.Union):
    _fields_ = [("discrete", FramePeriod), ("stepwise", StepwisePeriod)]


class FrameInterval(ct.Structure):
    _fields_ = [("index", ct.c_uint32), ("pixel_format", ct.c_uint32),
                ("width", ct.c_uint32), ("height", ct.c_uint32), ("type", ct.c_uint32),
                ("interval", PeriodUnion), ("reserved", ct.c_uint32 * 2)]


@dataclass(frozen=True)
class CameraMode:
    width: int
    height: int
    pixel_format: str
    numerator: int
    denominator: int

    @property
    def fps(self):
        return Fraction(self.denominator, self.numerator)

    @property
    def rank(self):
        # Frame rate outranks compression preference. YUYV only breaks a tie.
        preference = {"YUYV": 3, "UYVY": 2, "RGB3": 1, "BGR3": 1}
        return self.width * self.height, self.fps, preference.get(self.pixel_format, 0)


def _request(direction, number, structure):
    return (direction << 30) | (ct.sizeof(structure) << 16) | (ord("V") << 8) | number


QUERYCAP = _request(2, 0, Capability)
ENUM_FMT = _request(3, 2, FormatDescription)
G_FMT = _request(3, 4, Format)
S_FMT = _request(3, 5, Format)
REQBUFS = _request(3, 8, RequestBuffers)
QUERYBUF = _request(3, 9, Buffer)
QBUF = _request(3, 15, Buffer)
DQBUF = _request(3, 17, Buffer)
STREAMON = _request(1, 18, ct.c_uint32)
STREAMOFF = _request(1, 19, ct.c_uint32)
G_PARM = _request(3, 21, StreamParameters)
S_PARM = _request(3, 22, StreamParameters)
ENUM_FRAMESIZES = _request(3, 74, FrameSize)
ENUM_FRAMEINTERVALS = _request(3, 75, FrameInterval)
ALLOWED_REQUESTS = frozenset((QUERYCAP, G_FMT, G_PARM, REQBUFS, QUERYBUF,
                              QBUF, DQBUF, STREAMON, STREAMOFF, ENUM_FMT,
                              ENUM_FRAMESIZES, ENUM_FRAMEINTERVALS, S_FMT, S_PARM))
CAPTURE = 1
MMAP_MEMORY = 1
SUPPORTED_FORMATS = frozenset(("YUYV", "UYVY", "MJPG", "JPEG", "RGB3", "BGR3"))


def _ioctl(fd, request, value):
    if request not in ALLOWED_REQUESTS:
        raise ValueError("Only capture I/O and image-mode requests are allowed; controls are forbidden")
    while True:
        try:
            result = fcntl.ioctl(fd, request, bytes(value))
            ct.memmove(ct.addressof(value), result, ct.sizeof(value))
            return
        except OSError as error:
            if error.errno != errno.EINTR:
                raise


def _enumerate(fd, request, factory):
    index = 0
    while True:
        value = factory(index)
        try:
            _ioctl(fd, request, value)
        except OSError as error:
            if error.errno == errno.EINVAL:
                return
            raise
        yield value
        index += 1


def enumerate_modes(fd):
    """Enumerate decodable native sizes and advertised fastest valid periods."""
    modes = set()
    unknown_period_sizes = []
    for description in _enumerate(fd, ENUM_FMT,
                                  lambda index: FormatDescription(index=index, type=CAPTURE)):
        code = int(description.pixelformat)
        pixel_format = code.to_bytes(4, "little").decode("ascii", "replace")
        if pixel_format not in SUPPORTED_FORMATS:
            continue
        for size in _enumerate(fd, ENUM_FRAMESIZES,
                               lambda index: FrameSize(index=index, pixel_format=code)):
            if size.type == 1:
                width, height = size.size.discrete.width, size.size.discrete.height
            elif size.type in (2, 3):
                step = size.size.stepwise
                if (not step.min_width or not step.min_height
                        or step.max_width < step.min_width or step.max_height < step.min_height):
                    continue
                if size.type == 3 and (not step.step_width or not step.step_height):
                    continue
                width_step = step.step_width if size.type == 3 else 1
                height_step = step.step_height if size.type == 3 else 1
                width = step.min_width + (step.max_width - step.min_width) // width_step * width_step
                height = step.min_height + (step.max_height - step.min_height) // height_step * height_step
            else:
                continue
            if not width or not height or (pixel_format in ("YUYV", "UYVY") and width % 2):
                continue
            known_period = False
            for interval in _enumerate(fd, ENUM_FRAMEINTERVALS,
                                       lambda index: FrameInterval(index=index, pixel_format=code,
                                                                   width=width, height=height)):
                if interval.type == 1:
                    period = interval.interval.discrete
                elif interval.type in (2, 3):
                    period = interval.interval.stepwise.min
                else:
                    continue
                if period.numerator and period.denominator:
                    known_period = True
                    modes.add(CameraMode(int(width), int(height), pixel_format,
                                         int(period.numerator), int(period.denominator)))
                if interval.type in (2, 3):
                    break
            if not known_period:
                unknown_period_sizes.append(int(width) * int(height))
            if size.type in (2, 3):
                break
    if unknown_period_sizes and max(unknown_period_sizes) >= max(
            (mode.width * mode.height for mode in modes), default=0):
        raise RuntimeError("Highest advertised camera size has unknown FPS; "
                           "cannot verify maximum mode. Use --camera-current explicitly")
    return sorted(modes, key=lambda mode: mode.rank, reverse=True)


def _read_period(fd):
    parameters = StreamParameters(type=CAPTURE)
    _ioctl(fd, G_PARM, parameters)
    period = parameters.parm.capture
    if not period.numerator or not period.denominator:
        raise RuntimeError("Driver did not report a usable nominal frame interval")
    return parameters


def _apply_mode(fd, candidate):
    mode = Format(type=CAPTURE)
    pixel = mode.fmt.pix
    pixel.width, pixel.height, pixel.field = candidate.width, candidate.height, 1
    pixel.pixelformat = int.from_bytes(candidate.pixel_format.encode("ascii"), "little")
    _ioctl(fd, S_FMT, mode)
    _ioctl(fd, G_FMT, mode)
    pixel = mode.fmt.pix
    actual_format = int(pixel.pixelformat).to_bytes(4, "little").decode("ascii", "replace")
    if ((int(pixel.width), int(pixel.height), actual_format)
            != (candidate.width, candidate.height, candidate.pixel_format)
            or pixel.field not in (0, 1)):
        raise RuntimeError("Driver changed the requested native dimensions/format")
    # Preserve capturemode and other driver fields: only request a different period.
    parameters = _read_period(fd)
    parameters.parm.capture.numerator = candidate.numerator
    parameters.parm.capture.denominator = candidate.denominator
    _ioctl(fd, S_PARM, parameters)
    actual = _read_period(fd).parm.capture
    return CameraMode(candidate.width, candidate.height, candidate.pixel_format,
                      int(actual.numerator), int(actual.denominator))


def negotiate_max_mode(fd):
    """Choose maximum resolution, then maximum accepted nominal FPS at that size."""
    try:
        modes = enumerate_modes(fd)
    except OSError as error:
        raise RuntimeError("Cannot enumerate highest camera mode; use --camera-current "
                           "only to explicitly retain the current mode") from error
    if not modes:
        raise RuntimeError("No supported camera sizes with known FPS were advertised; "
                           "use --camera-current to explicitly retain the current mode")
    best = None
    selected = None
    active = None
    failures = []
    for candidate in modes:
        if best is not None and candidate.width * candidate.height < best.width * best.height:
            break
        # A lower advertised rate cannot improve an already accepted rate at this size.
        if best is not None and candidate.fps < best.fps:
            break
        try:
            actual = _apply_mode(fd, candidate)
        except (OSError, RuntimeError) as error:
            failures.append("%dx%d %s %.3f FPS: %s" % (
                candidate.width, candidate.height, candidate.pixel_format, float(candidate.fps), error))
            active = None
            continue
        active = actual
        if best is None or actual.rank > best.rank:
            best, selected = actual, candidate
        if actual.fps >= candidate.fps:
            break  # Candidates are already ordered by maximum advertised mode.
        failures.append("%dx%d %s: requested %.3f FPS, driver accepted %.3f FPS" % (
            candidate.width, candidate.height, candidate.pixel_format,
            float(candidate.fps), float(actual.fps)))
    if best is None:
        raise RuntimeError("Driver rejected every advertised native mode; use --camera-current. "
                           + "; ".join(failures))
    if active != best:
        restored = _apply_mode(fd, selected)
        if restored != best:
            raise RuntimeError("Driver could not restore the best verified camera mode")
    return modes[0], selected, best, failures


def decode_frame(payload, width, height, pixel_format, stride):
    """Convert the complete native frame to BGR, without cropping or resizing."""
    if pixel_format in ("MJPG", "JPEG"):
        image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.shape != (height, width, 3):
            raise RuntimeError("Invalid JPEG frame or native frame size changed")
        return image
    if pixel_format not in SUPPORTED_FORMATS:
        raise RuntimeError("Unsupported existing camera format: " + pixel_format)
    channels = 2 if pixel_format in ("YUYV", "UYVY") else 3
    if channels == 2 and width % 2:
        raise RuntimeError("Packed YUV frame width must be even")
    row_bytes = width * channels
    stride = stride or row_bytes
    if width <= 0 or height <= 0 or stride < row_bytes:
        raise RuntimeError("Invalid native frame dimensions or row stride")
    if len(payload) < (height - 1) * stride + row_bytes:
        raise RuntimeError("Truncated native camera frame")
    image = np.ndarray((height, width, channels), np.uint8, buffer=payload,
                       strides=(stride, channels, 1)).copy()
    if pixel_format == "YUYV":
        return cv2.cvtColor(image, cv2.COLOR_YUV2BGR_YUY2)
    if pixel_format == "UYVY":
        return cv2.cvtColor(image, cv2.COLOR_YUV2BGR_UYVY)
    if pixel_format == "RGB3":
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


class CurrentModeCapture:
    """Own a native capture queue, with mode-only negotiation when explicitly enabled."""

    def __init__(self, device, camera_mode="max"):
        if (sys.platform != "linux" or platform.machine().lower() not in
                ("x86_64", "amd64", "aarch64", "arm64")
                or ct.sizeof(ct.c_long) != 8 or sys.byteorder != "little"):
            raise RuntimeError("Native capture requires Linux x86_64 or ARM64")
        if camera_mode not in ("max", "current"):
            raise ValueError("camera_mode must be max or current")
        self.camera_mode = camera_mode
        self.fd = None
        self.buffers = []
        self.streaming = False
        self.dequeued_frame_count = 0
        self.requested_mode = self.selected_mode = self.negotiated_mode = None
        self.mode_warnings = []
        try:
            self.fd = os.open(device, os.O_RDWR | os.O_NONBLOCK | os.O_CLOEXEC)
            if not stat.S_ISCHR(os.fstat(self.fd).st_mode):
                raise RuntimeError("Camera path is not a character device: " + device)
            capability = Capability()
            _ioctl(self.fd, QUERYCAP, capability)
            caps = (capability.device_caps if capability.capabilities & 0x80000000
                    else capability.capabilities)
            if not caps & 0x00000001 or not caps & 0x04000000:
                raise RuntimeError("Camera must support single-plane video capture and streaming")
            if camera_mode == "max":
                (self.requested_mode, self.selected_mode, self.negotiated_mode,
                 self.mode_warnings) = negotiate_max_mode(self.fd)
            mode = Format(type=CAPTURE)
            _ioctl(self.fd, G_FMT, mode)
            pixel = mode.fmt.pix
            self.width, self.height = int(pixel.width), int(pixel.height)
            self.pixel_format = int(pixel.pixelformat).to_bytes(4, "little").decode("ascii", "replace")
            self.stride = int(pixel.bytesperline)
            if self.width <= 0 or self.height <= 0 or pixel.field not in (0, 1):
                raise RuntimeError("Invalid or interlaced existing camera mode")
            if self.pixel_format not in SUPPORTED_FORMATS:
                raise RuntimeError("Unsupported existing camera format: " + self.pixel_format
                                   + "; settings were not changed")
            parameters = StreamParameters(type=CAPTURE)
            self.fps = None
            try:
                _ioctl(self.fd, G_PARM, parameters)
                period = parameters.parm.capture
                if period.numerator and period.denominator:
                    self.fps = period.denominator / period.numerator
            except OSError as error:
                if error.errno not in (errno.EINVAL, errno.ENOTTY):
                    raise
            if self.negotiated_mode is not None:
                verified = self.negotiated_mode
                if ((self.width, self.height, self.pixel_format)
                        != (verified.width, verified.height, verified.pixel_format)
                        or self.fps is None
                        or abs(self.fps - float(verified.fps)) > 1e-6):
                    raise RuntimeError("Camera mode changed after negotiation; refusing to misreport input")
            request = RequestBuffers(count=2, type=CAPTURE, memory=MMAP_MEMORY)
            _ioctl(self.fd, REQBUFS, request)
            if not request.count:
                raise RuntimeError("Camera allocated no mmap capture buffers")
            for index in range(request.count):
                buffer = Buffer(index=index, type=CAPTURE, memory=MMAP_MEMORY)
                _ioctl(self.fd, QUERYBUF, buffer)
                self.buffers.append(mmap.mmap(self.fd, buffer.length, flags=mmap.MAP_SHARED,
                                              prot=mmap.PROT_READ | mmap.PROT_WRITE,
                                              offset=buffer.m.offset))
                _ioctl(self.fd, QBUF, buffer)
            _ioctl(self.fd, STREAMON, ct.c_uint32(CAPTURE))
            self.streaming = True
        except BaseException:
            self.close()
            raise

    def read(self, timeout=2.0):
        if not select.select([self.fd], [], [], timeout)[0]:
            raise TimeoutError("Camera produced no frame for %.1f seconds" % timeout)
        payload = None
        # Drain only this queue's maximum depth to keep the newest available frame.
        for _ in self.buffers:
            buffer = Buffer(type=CAPTURE, memory=MMAP_MEMORY)
            try:
                _ioctl(self.fd, DQBUF, buffer)
            except OSError as error:
                if error.errno == errno.EAGAIN:
                    break
                raise
            try:
                if buffer.index >= len(self.buffers) or buffer.flags & 0x0040:
                    raise RuntimeError("Invalid or corrupted camera capture buffer")
                mapping = self.buffers[buffer.index]
                if not 0 < buffer.bytesused <= len(mapping):
                    raise RuntimeError("Invalid camera frame payload length")
                self.dequeued_frame_count += 1
                payload = mapping[:buffer.bytesused]
            finally:
                _ioctl(self.fd, QBUF, buffer)
        if payload is None:
            return None
        return decode_frame(payload, self.width, self.height, self.pixel_format, self.stride)

    def close(self):
        if self.fd is None:
            return
        try:
            if self.streaming:
                try:
                    _ioctl(self.fd, STREAMOFF, ct.c_uint32(CAPTURE))
                except OSError:
                    pass  # A disconnected USB device still needs local cleanup.
            for mapping in self.buffers:
                mapping.close()
        finally:
            self.buffers = []
            self.streaming = False
            os.close(self.fd)
            self.fd = None
