from cm_tpm._add import add
from cm_tpm._multiply import multiply
#import cm_tpm._add

class CMImputer():
    """
    
    """
    def __init__(self):
        super().__init__()

    def test(self):
        """
        Test the module
        """
        return 1
    
    def add(self, x, y):
        """
        Add two numbers
        """
        return add(x, y)
    
    def multiply(self, x, y):
        """
        Multiply two numbers
        """
        return multiply(x, y)