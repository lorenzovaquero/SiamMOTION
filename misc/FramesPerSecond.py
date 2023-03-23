"""FramesPerSecond.py: Easy calculation of FPS"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

__author__ = "Lorenzo Vaquero Otal"
__credits__ = ["Lorenzo Vaquero Otal"]
__date__ = "2023"
__version__ = "1.0.0"
__maintainer__ = "Lorenzo Vaquero Otal"
__contact__ = "lorenzo.vaquero.otal@usc.es"
__status__ = "Prototype"


class FramesPerSecond(object):
    """Allows an easy calculation of FPS"""

    def __init__(self, max_stored_ticks=20):
        self.max_stored_ticks = max_stored_ticks
        self.tick_list = [0.0] * self.max_stored_ticks
        self.current_ticks = 0
        self.tick_index = 0
        self.ticks_sum = 0.0
        self.start_time = 0.0
        self.stop_time = 0.0
        self.last_time = self.start_time

        self.slowest_tick = float('inf')
        self.fastest_tick = -float('inf')
        self.average_tick = 0.0

        self.first_tick = None
        self.last_tick = None

        self.total_ticks = 0
        self.total_tick_sum = 0.0

    def start(self):
        """Starts the FPS counter"""

        self.start_time = time.time()
        self.last_time = self.start_time

    def tick(self):
        """Indicates the calculation of a new frame, and retrieves the current amount of FPS"""

        new_time = time.time()
        tick_time = new_time - self.last_time
        tick_fps_raw = 1.0 / tick_time

        fps = self.__new_tick(tick_time)
        self.last_time = new_time

        if tick_fps_raw > self.fastest_tick:
            self.fastest_tick = tick_fps_raw

        if tick_fps_raw < self.slowest_tick:
            self.slowest_tick = tick_fps_raw

        return fps

    def stop(self):
        """Stops the FPS counter. (doesn't count as a tick)"""

        self.stop_time = time.time()

        if self.total_ticks == 0:
            total_time = (self.stop_time - self.start_time)
            return total_time

        else:
            if self.total_ticks > 2:
                total_ticks = self.total_ticks - 2
                start_time = self.start_time + self.first_tick
                stop_time = self.stop_time - self.last_tick
            else:
                total_ticks = self.total_ticks
                start_time = self.start_time
                stop_time = self.stop_time

            self.average_tick = total_ticks / (stop_time - start_time)
            return self.average_tick

    def __new_tick(self, tick):
        if self.first_tick is None:
            self.first_tick = tick
        self.last_tick = tick

        self.ticks_sum = self.ticks_sum - self.tick_list[self.tick_index]

        self.ticks_sum = self.ticks_sum + tick

        self.tick_list[self.tick_index] = tick

        self.tick_index = self.tick_index + 1
        self.current_ticks = self.current_ticks + 1

        self.total_ticks = self.total_ticks + 1
        self.total_tick_sum = self.total_tick_sum + tick

        if self.current_ticks >= self.max_stored_ticks:
            self.tick_index = 0
            self.current_ticks = self.max_stored_ticks

        return self.current_ticks / self.ticks_sum
