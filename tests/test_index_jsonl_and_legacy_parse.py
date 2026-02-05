import pytest
import datetime
from cleo.classes import _parse_index_line, _timestamp_key


def test_parse_index_jsonl_roundtrip():
    line = '{"subclass":"WindAtlas","country":"AUT","region":"None","scenario":"default","path":"C:\\\\data\\\\x.nc","timestamp":"20260101T000000"}'
    e = _parse_index_line(line)
    assert e[4].startswith("C:\\")
    assert e[5] == "20260101T000000"


def test_parse_legacy_colon_with_windows_path():
    legacy = "WindAtlas:AUT:None:default:C:\\data\\WindAtlas_AUT.nc:20260101T000000"
    e = _parse_index_line(legacy)
    assert e[0] == "WindAtlas"
    assert e[4] == "C:\\data\\WindAtlas_AUT.nc"
    assert e[5] == "20260101T000000"


def test_timestamp_key_legacy_is_oldest():
    assert _timestamp_key("legacy") < _timestamp_key("20200101T000000")
    with pytest.raises(ValueError):
        _timestamp_key("not-a-timestamp")
