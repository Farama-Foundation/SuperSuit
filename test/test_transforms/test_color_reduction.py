from supersuit.transforms.color_reduction import check_param
def test_param_check():
    check_param("jon")
    with pytest.raises(AssertionError):
        check_param("bob")
