from cm_tpm import CMImputer

def test_module():
    imputer = CMImputer()
    assert imputer.test() == 1
    