class OurQueue:
    """
    A queue for counting efficiently the number of events within time windows.
    Complexity:
        All operators in amortized O(W) time where W is the number of windows.
    """
    def __init__(self, only_forever=False):
        self.now = None
        self.queue = []
        self.window_lengths = [] if only_forever else [3600 * 24 * 30, 3600 * 24 * 7, 3600 * 24, 3600]
        self.cursors = [0] * len(self.window_lengths)

    def __len__(self):
        return len(self.queue)

    def get_counters(self, t):
        self.update_cursors(t)
        return [len(self.queue)] + [len(self.queue) - cursor
                                    for cursor in self.cursors]

    def push(self, time):
        self.queue.append(time)

    def update_cursors(self, t):
        for pos, length in enumerate(self.window_lengths):
            while (self.cursors[pos] < len(self.queue) and
                   t - self.queue[self.cursors[pos]] >= length):
                self.cursors[pos] += 1
        # print(t, self.queue[:self.cursors[0]],  # For debug purposes
        #       [self.queue[self.cursors[i]:self.cursors[i + 1]]
        #        for i in range(len(self.cursors) - 1)],
        #       self.queue[self.cursors[-1]:])
