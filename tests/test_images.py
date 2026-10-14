"""Tests for EXIF-orientation normalisation (turnstone.core.images)."""

from __future__ import annotations

import ast
import struct
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest

from tests._exif_helpers import (
    exif_block,
    halves,
    jpeg_with_exif,
    orientation_entry,
    png_with_exif,
)
from turnstone.core import images
from turnstone.core.images import exif_orientation, normalize_image_orientation

Image = pytest.importorskip("PIL.Image")

_ORIENTATION_TAG = 0x0112  # EXIF orientation (standard tag id)


@pytest.fixture(autouse=True)
def _nothing_kept() -> None:
    images._normalized.clear()


def _oriented_jpeg(orientation: int, size: tuple[int, int] = (4, 2)) -> bytes:
    img = Image.new("RGB", size, "red")
    exif = img.getexif()
    exif[_ORIENTATION_TAG] = orientation
    buf = BytesIO()
    img.save(buf, format="JPEG", exif=exif)
    return buf.getvalue()


def test_applies_rotation_and_strips_tag() -> None:
    # Orientation 6 = "rotate 90° for display": a 4×2 landscape becomes 2×4.
    data = _oriented_jpeg(6, size=(4, 2))
    out = normalize_image_orientation(data)
    assert out != data, "a rotated image must be re-encoded upright"
    img = Image.open(BytesIO(out))
    assert img.size == (2, 4), "the 90° rotation must be baked into the pixels"
    assert img.getexif().get(_ORIENTATION_TAG) in (None, 1), "the orientation tag must be cleared"


def test_passthrough_when_upright() -> None:
    data = _oriented_jpeg(1, size=(4, 2))
    assert normalize_image_orientation(data) == data, "identity orientation must not re-encode"


def test_passthrough_when_no_exif() -> None:
    buf = BytesIO()
    Image.new("RGB", (3, 3), "blue").save(buf, format="PNG")
    data = buf.getvalue()
    assert normalize_image_orientation(data) == data, "a tag-less image must pass through verbatim"


def test_never_raises_on_garbage() -> None:
    assert normalize_image_orientation(b"not an image") == b"not an image"
    assert normalize_image_orientation(b"") == b""


def test_scan_limit() -> None:
    from tests._jpeg_helpers import jpeg_with_scans
    from turnstone.core.images import MAX_JPEG_SCANS, too_many_jpeg_scans

    own = jpeg_with_scans(0).count(b"\xff\xda")
    assert not too_many_jpeg_scans(jpeg_with_scans(MAX_JPEG_SCANS - own))
    assert too_many_jpeg_scans(jpeg_with_scans(MAX_JPEG_SCANS - own + 1))
    buf = BytesIO()
    Image.new("RGB", (3, 3), "blue").save(buf, format="PNG")
    assert not too_many_jpeg_scans(buf.getvalue() + b"\xff\xda" * (MAX_JPEG_SCANS + 1))


def test_a_jpeg_with_too_many_scans_is_passed_through_undecoded() -> None:
    # Each scan costs a decode pass over the whole picture.
    from tests._jpeg_helpers import jpeg_with_scans

    turned = jpeg_with_scans(0, orientation=6)
    assert normalize_image_orientation(turned) != turned
    slow = jpeg_with_scans(150, orientation=6)
    assert normalize_image_orientation(slow) == slow


def _webp(exif: bytes) -> bytes:
    buf = BytesIO()
    halves().save(buf, format="WEBP", exif=exif)
    return buf.getvalue()


def _png_with_text(key: str, text: str) -> bytes:
    from PIL import PngImagePlugin

    info = PngImagePlugin.PngInfo()
    info.add_itxt(key, text)
    buf = BytesIO()
    halves().save(buf, format="PNG", pnginfo=info)
    return buf.getvalue()


def _raw_profile(exif: bytes) -> str:
    # The text form some tools write: a name, the length, then hex lines.
    hexed = exif.hex()
    lines = [hexed[i : i + 72] for i in range(0, len(hexed), 72)]
    return "\nexif\n" + f"{len(exif):8d}\n" + "\n".join(lines)


@pytest.mark.parametrize(
    ("source", "data", "expected"),
    [
        ("jpeg", lambda: jpeg_with_exif(exif_block([orientation_entry(6)])), 6),
        (
            "big-endian jpeg",
            lambda: jpeg_with_exif(exif_block([orientation_entry(3, ">")], ">")),
            3,
        ),
        ("png", lambda: png_with_exif(exif_block([orientation_entry(8)])), 8),
        ("webp", lambda: _webp(exif_block([orientation_entry(2)])), 2),
        ("none", lambda: png_with_exif(b""), 1),
    ],
)
def test_orientation_is_read_where_browsers_read_it(source: str, data: Any, expected: int) -> None:
    img = Image.open(BytesIO(data()))
    assert exif_orientation(img) == expected, "read at open, before any decode"
    assert img.getexif().get(_ORIENTATION_TAG, 1) == expected, "Pillow reads the same"


@pytest.mark.parametrize(
    ("source", "data"),
    [
        (
            "png, after its pixels",
            lambda: png_with_exif(exif_block([orientation_entry(5)]), after_pixels=True),
        ),
        (
            "png text",
            lambda: _png_with_text(
                "Raw profile type exif", _raw_profile(exif_block([orientation_entry(4)]))
            ),
        ),
        (
            "xmp",
            lambda: _png_with_text(
                "XML:com.adobe.xmp",
                '<x:xmpmeta><rdf:Description tiff:Orientation="6"/></x:xmpmeta>',
            ),
        ),
    ],
)
def test_orientation_stored_elsewhere_is_ignored(source: str, data: Any) -> None:
    # Browsers don't turn these, so neither do the thumbnail and the model's copy.
    picture = data()
    assert exif_orientation(Image.open(BytesIO(picture))) == 1
    assert normalize_image_orientation(picture) is picture


@pytest.mark.parametrize(
    ("entry", "expected"),
    [
        ((0x0112, 3, 1, struct.pack("<HH", 6, 0)), 6),  # SHORT, as the standard says
        ((0x0112, 4, 1, struct.pack("<L", 6)), 6),  # LONG
        ((0x0112, 8, 1, struct.pack("<hh", 6, 0)), 6),  # SSHORT
        ((0x0112, 9, 1, struct.pack("<l", 6)), 6),  # SLONG
        ((0x0112, 1, 1, b"\x06\x00\x00\x00"), 1),  # BYTE: ignored
        ((0x0112, 3, 2, struct.pack("<HH", 6, 6)), 1),  # more than one value: ignored
    ],
)
def test_orientation_entry_types(entry: tuple[int, int, int, bytes], expected: int) -> None:
    img = Image.open(BytesIO(jpeg_with_exif(exif_block([entry]))))
    assert exif_orientation(img) == expected


@pytest.mark.parametrize(
    "exif",
    [
        b"II*\x00",  # stops after the byte order
        b"II*\x00" + struct.pack("<L", 4000) + bytes(8),  # directory past the end
    ],
)
@pytest.mark.parametrize("container", [jpeg_with_exif, png_with_exif])
@pytest.mark.filterwarnings("ignore:Corrupt EXIF data")
def test_malformed_exif_reads_as_upright(exif: bytes, container: Any) -> None:
    # A photo whose EXIF stops short is still sent, thumbnailed and accepted.
    from turnstone.core.avatars import normalize_picture
    from turnstone.core.thumbnails import make_thumbnail

    picture = container(exif)
    assert exif_orientation(Image.open(BytesIO(picture))) == 1
    assert normalize_image_orientation(picture) is picture
    assert make_thumbnail(picture, "image") is not None
    assert normalize_picture(picture)[0]


def test_a_rotated_photo_with_oddly_typed_tags_is_still_turned() -> None:
    # Pillow's own transpose rewrites the remaining EXIF, and fails on a tag
    # stored in a type other than the one it expects.
    resolution_unit = (0x0128, 4, 1, struct.pack("<L", 70000))
    data = jpeg_with_exif(exif_block([orientation_entry(6), resolution_unit]))
    assert Image.open(BytesIO(normalize_image_orientation(data))).size == (20, 40)


def test_a_rotated_photo_is_turned_once(monkeypatch: pytest.MonkeyPatch) -> None:
    # Every send rebuilds the request from the stored originals.
    turns: list[int] = []
    real = images.turn_upright

    def counting(img: Any, orientation: int) -> Any:
        turns.append(orientation)
        return real(img, orientation)

    monkeypatch.setattr(images, "turn_upright", counting)
    data = _oriented_jpeg(6, size=(4, 2))
    first = normalize_image_orientation(data)
    assert normalize_image_orientation(data) is first
    assert turns == [6]


def test_an_upright_png_is_never_decoded(monkeypatch: pytest.MonkeyPatch) -> None:
    from PIL import PngImagePlugin

    loads: list[int] = []
    real = PngImagePlugin.PngImageFile.load

    def counting(self: Any) -> Any:
        loads.append(1)
        return real(self)

    monkeypatch.setattr(PngImagePlugin.PngImageFile, "load", counting)
    buf = BytesIO()
    Image.new("RGB", (3, 3), "blue").save(buf, format="PNG")
    data = buf.getvalue()
    assert normalize_image_orientation(data) is data
    assert loads == []


def test_kept_images_are_bounded() -> None:
    cache = images._NormalizedCache(max_bytes=10, max_entries=3)
    cache.put(b"a", b"x" * 6)
    cache.put(b"b", b"y")
    cache.put(b"c", b"x" * 6)  # 13 bytes: the least recently used goes
    assert cache.get(b"a") is None
    assert cache.get(b"b") == b"y"
    cache.put(b"d", b"z")
    cache.put(b"e", b"w")  # four entries: c goes, since b was just used
    assert cache.get(b"c") is None
    assert cache.get(b"b") == b"y"
    cache.put(b"f", b"x" * 11)  # more than the whole budget: not kept
    assert cache.get(b"f") is None


def test_every_image_decoder_checks_scans_and_reads_exif_itself() -> None:
    # A decoder that skips the scan check, or reads EXIF through Pillow, still
    # works on every ordinary file, so nothing else would notice.
    root = Path(__file__).resolve().parent.parent / "turnstone"
    checked: dict[str, bool] = {}
    pillow_exif: list[str] = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        where = path.relative_to(root.parent)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("getexif", "exif_transpose")
            ):
                pillow_exif.append(f"{where}:{node.lineno}")
        for fn in ast.walk(tree):
            if not isinstance(fn, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            nodes = list(ast.walk(fn))
            opens = any(
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr == "open"
                and isinstance(n.func.value, ast.Name)
                and n.func.value.id == "Image"
                for n in nodes
            )
            if opens:
                names = {n.id for n in nodes if isinstance(n, ast.Name)}
                checked[f"{where}:{fn.name}"] = "too_many_jpeg_scans" in names
    assert "turnstone/core/images.py:normalize_image_orientation" in checked, checked
    assert all(checked.values()), checked
    assert pillow_exif == []
