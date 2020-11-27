
import math


def set_seq(number_needed, num_range):
    sequence = []
    for i in range(number_needed):
        sequence.append(hash(i*(math.pi))%num_range)
    return sequence

print(set_seq(1000, 100))

