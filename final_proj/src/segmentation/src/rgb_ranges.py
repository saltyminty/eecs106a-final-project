
from typing import Tuple


class RGBRange:
    def __init__(self, rgb_min: Tuple[int], rgb_max: Tuple[int]):
        self.rgb_min = rgb_min
        self.rgb_max = rgb_max

PURPLE_CHAIR = RGBRange((100, 100, 100),
                        (200, 200, 200))

GREEN_CHAIR = RGBRange((100, 100, 100),
                        (200, 200, 200))

BROWN_TABLE = RGBRange((100, 100, 100),
                        (200, 200, 200)) 