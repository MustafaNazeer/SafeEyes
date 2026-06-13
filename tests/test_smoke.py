import safeeyes


def test_version_is_non_empty_string() -> None:
    assert isinstance(safeeyes.__version__, str)
    assert safeeyes.__version__
