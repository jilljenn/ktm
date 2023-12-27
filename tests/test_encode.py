import unittest
from subprocess import check_output


class TestDummy(unittest.TestCase):
    def test_encode(self):
        commands = [
            ['python', 'encode.py', '--users', '--items'],
            ['python', 'lr.py', 'data/dummy/X-ui.npz'],
            ['python', 'lr.py', '--folds', 'strong', 'data/dummy/X-ui.npz'],
            ['python', 'fm.py', 'data/dummy/X-ui.npz'],
            ['python', 'fm.py', '--folds', 'weak', 'data/dummy/X-ui.npz'],
            ['python', 'sktm.py', '--model', 'irt'],
            ['python', 'sktm.py', '--model', 'pfa'],
            ['python', 'sktm.py', '--model', 'iswf']
        ]
        for command in commands:
            p = check_output(command)
