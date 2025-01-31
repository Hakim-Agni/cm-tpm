from cm_tpm.cpp._add import add
from cm_tpm.cpp._multiply import multiply
from cm_tpm.cpp._subtract import subtract
from cm_tpm.cpp._divide import divide

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
    
    def subtract(self, x, y):
        """
        Subtract two numbers
        """
        return subtract(x, y)
    
    def divide(self, x, y):
        """
        Divide two numbers
        """
        return divide(x, y)