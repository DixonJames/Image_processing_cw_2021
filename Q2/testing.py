import random
import math

class masks:

    def normalDis(self, x, sigma):
        return ((1 / (sigma * math.sqrt(2 * math.pi))) * (math.e ** (-(x ** 2 / (2 * (sigma ** 2))))))



    def gaussianDegradation(self, number_Points, sigma):
        diff = (4 * sigma) / number_Points
        out = []
        for i in range(number_Points):
            out.append(self.normalDis(i * diff, sigma) * 1 )
        return out

    def gausssianNoise(self, deviation, width):
        #dropoff = self.gaussianDegradation(20, deviation)
        step = deviation * 4 / width
        distribution = [self.normalDis(0, deviation)]
        effect = [1]
        for i in range(width):
            distribution = [self.normalDis(i*step, deviation)] + distribution + [self.normalDis(i*step, deviation)]
            effect = [self.normalDis(i*step, deviation)] + effect + [self.normalDis(i, deviation) ]
        return random.choices(effect, distribution, k = 1)

print(masks().gausssianNoise(1, 10))