import ctypes
from types import SimpleNamespace

import pytest

from mojonumagg import _lib


def test_ffi_signatures_distinguish_pointers_from_lengths():
    library = _lib.lib()
    assert library.mna_move_sum.argtypes == [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
    ]
    assert library.mna_group_reduce.argtypes[:5] == [ctypes.c_void_p] * 5
    assert library.mna_group_reduce.argtypes[5:] == [ctypes.c_int64] * 5


def test_addr_rejects_null_numpy_buffer():
    with pytest.raises(ValueError, match="null array pointer"):
        _lib.addr(SimpleNamespace(ctypes=SimpleNamespace(data=0)))


def test_missing_library_override_fails_clearly(monkeypatch, tmp_path):
    missing = tmp_path / "missing.so"
    monkeypatch.setenv("MOJONUMAGG_LIB", str(missing))
    monkeypatch.setattr(_lib, "LIB", str(missing))
    with pytest.raises(_lib.BuildError, match="does not exist or is not a file"):
        _lib.build()
