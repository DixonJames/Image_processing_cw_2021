import math

def addAngle(angleA, angleB):
    if angleA + angleB - (2 * math.pi) == 0:
        return 2 * math.pi
    if angleA + angleB - (2 * math.pi) < 0:
        return angleA + angleB
def subAngle(angleA, angleB):
    #result is A - B
    if angleA >= angleB:
        return addAngle(angleA, - angleB)
    return ((angleA - angleB)%(2*math.pi))



print(addAngle(math.pi, math.pi))
print(addAngle(math.pi, - math.pi))
print(subAngle(math.pi,  4.5 * math.pi))