import unittest
from utils.this_queue import OurQueue


class TestOurQueue(unittest.TestCase):

    def test_simple(self):
        q = OurQueue()
        q.push(0)
        q.push(0.8 * 3600 * 24)
        q.push(5 * 3600 * 24)
        q.push(40 * 3600 * 24)
        self.assertEqual(q.get_counters(40 * 3600 * 24), [4, 1, 1, 1, 1])
        
    def test_complex(self):
        q = OurQueue()
        q.push(0)
        q.push(10)
        q.push(3599)
        q.push(3600)
        q.push(3601)
        q.push(3600 * 24)
        q.push(3600 * 24 + 1)
        q.push(3600 * 24 * 7)
        q.push(3600 * 24 * 7 + 1)
        q.push(3600 * 24 * 7 * 30)
        q.push(3600 * 24 * 7 * 30 + 1)
        self.assertEqual(q.get_counters(3600 * 24 * 7 * 30 + 1), [11, 2, 2, 2, 2])
