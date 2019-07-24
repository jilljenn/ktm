class OurQueue:
    """A queue for counting efficiently the number of events within different time windows.
    Complexity:
        all operators in amortized O(B) time,
    """
    def __init__(self):
        self.queue = []        # tail
        self.window_lengths = [3600 * 24 * 7 * 30, 3600 * 24 * 7, 3600 * 24, 3600]
        self.cursors = [0] * len(self.window_lengths) 

    def __len__(self):
        return len(self.queue)

    def get_counters(self):
        return [len(self.queue)] + [len(self.queue) - cursor for cursor in self.cursors]

    def push(self, time):
        self.queue.append(time)
        self.now = time
        return self.update_cursors()

    def update_cursors(self):
        for pos, length in enumerate(self.window_lengths):
            while self.now - self.queue[self.cursors[pos]] > length:
                self.cursors[pos] += 1
        # print(self.now, self.queue[:self.cursors[0]], [self.queue[self.cursors[i]:self.cursors[i + 1]] for i in range(len(self.cursors) - 1)], self.queue[self.cursors[-1]:])


# Make this a test
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
