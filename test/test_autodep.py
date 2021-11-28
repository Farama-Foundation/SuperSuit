import pytest
import supersuit


def test_bad_import():
    with pytest.raises(supersuit.DeprecatedWrapper):
        from supersuit import action_lambda_v0
    with pytest.raises(supersuit.DeprecatedWrapper):
        supersuit.action_lambda_v0
