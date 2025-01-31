from cm_tpm import CMImputer

def test_module():
    imputer = CMImputer()
    assert imputer.test() == 1
    assert imputer.add(1, 2) == 3
    assert imputer.multiply(2, 3) == 6
    assert imputer.subtract(3, 2) == 1
    assert imputer.divide(6, 3) == 2