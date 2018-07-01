import tempfile

from galaxy.util.compression_utils import CompressedFile


def test_compression_safety():
    assert_safety_needed("test-data/unsafe.tar")
    assert_safety_needed("test-data/unsafe.zip")


def assert_safety_needed(path):
    d = tempfile.mkdtemp()
    exception = None
    try:
        CompressedFile(path).extract(d)
    except Exception as e:
        exception = e

    assert exception is not None

    exception = None
    try:
        CompressedFile(path, safe=False).extract(d)
    except Exception as e:
        exception = e

    assert exception is None, str(exception)
