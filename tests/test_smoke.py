"""The package imports and reports a version."""

import torchdenoise


def test_the_package_reports_a_version():
    assert isinstance(torchdenoise.__version__, str)
    assert torchdenoise.__version__
