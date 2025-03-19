import pytest
from unittest.mock import patch, call, Mock
from tiamat.readers import factory
from tiamat.errors import UnknownFileError


expected_args = [
    (
        "foo/bar/a.b.c.d.e",
        ["a.b.c.d.e", "b.c.d.e", "c.d.e", "d.e", "e"]
    ),
    (
        "a.b.c.d.e",
        ["a.b.c.d.e", "b.c.d.e", "c.d.e", "d.e", "e"]
    ),
]

@pytest.mark.parametrize("fname, expected_calls", expected_args)
def test_get_reader(fname, expected_calls):
    with patch.object(factory, "get_reader_for_file_type") as get_reader_for_file_type_mock:
        get_reader_for_file_type_mock.side_effect = UnknownFileError("foo")
        with pytest.raises(UnknownFileError):
            factory.get_reader(fname)
        assert get_reader_for_file_type_mock.call_args_list == list(map(call, expected_calls))


@pytest.mark.parametrize("fname, expected_calls", expected_args)
def test_get_reader_success(fname, expected_calls):
    with patch.object(factory, "get_reader_for_file_type") as get_reader_for_file_type_mock:
        mock_cls = Mock()
        mock_cls.return_value = Mock()
        get_reader_for_file_type_mock.return_value = mock_cls
        reader = factory.get_reader(fname)

        get_reader_for_file_type_mock.assert_called_once_with(expected_calls[0])
        assert get_reader_for_file_type_mock.call_count == 1
        
        mock_cls.assert_called_once_with(fname)
        assert reader is mock_cls.return_value
