import numpy as np
from scipy import ndimage
import cv2
from abc import ABC, abstractmethod


"""
Action[0] is accelerator, Action[1] is steering
"""


class AutoDriveGame:

    target_color = (255, 255, 0)

    def __init__(self, map_, passenger, traffic_signs, car, target, score_desc_rate):
        self.map = map_
        self.view = np.array(self.map, copy=True)
        self.passenger = passenger
        self.traffic_signs = traffic_signs
        self.car = car
        self.target = target
        self.time = 0
        self.score_desc_rate = score_desc_rate
        self.extra_score = 0

        # Make a fence
        for i in range(np.ceil(np.amax(car.shape)).astype(np.int64)):
            self.map[i] = 255
            self.map[-i - 1] = 255
            self.map[:, i] = 255
            self.map[:, -i - 1] = 255

        self.map = cv2.circle(self.map, (target[1], target[0]), 7, self.target_color, 3)

    def step(self, action):
        self.view = np.array(self.map, copy=True, dtype=np.float64)
        self.car.step(action)
        pos = np.round(self.car.pos)
        max_shape = self.car.cur_img.shape
        y_min, y_max = int(pos[1] - max_shape[1]//2), int(pos[1] + max_shape[1]//2+1)
        x_min, x_max = int(pos[0] - max_shape[0]//2), int(pos[0] + max_shape[0]//2+1)
        map_to_update = self.map[y_min:y_max, x_min:x_max]
        if map_to_update.shape != self.car.appearance.shape:
            self.time = 99999999
            return False
        if np.any(np.all(map_to_update == self.target_color, axis=2) &
                             np.any(self.car.cur_img != 0, axis=2)):
            return False
        if np.any(np.all(map_to_update == 255, axis=2) & np.any(self.car.cur_img != 0, axis=2)):
            self.time = 99999999
            return False
        self.view[y_min:y_max, x_min:x_max] = map_to_update + self.car.cur_img

        self.time += 1
        return True

    def reset(self):
        self.car.reset()
        self.time = 0
        self.extra_score = 0
        self.view = self.map

    def get_score(self):
        return np.power(self.score_desc_rate, self.time) + self.extra_score

    def get_img(self):
        return self.view


class Car:

    def __init__(self, head_direction, pos, shape, color, horsepower, steering_sensitivity,  max_steering, friction):
        assert shape[0] % 2 == 1 and shape[1] % 2 == 1
        self.head_direction = head_direction
        self.steering = steering_sensitivity
        self.steering_angle = 0
        self.max_steering_angle = max_steering
        self.shape = shape
        self.tier_len = shape[1]//4
        self.horsepower = horsepower

        # should be negative
        self.friction = friction
        self.velocity = np.zeros(2)
        self.pos = pos
        self.appearance = np.zeros((*([max(shape) * 2 + 1] * 2), 3))
        self.center = (self.appearance.shape[0]//2, self.appearance.shape[1]//2)
        start_point = self.center[1] - self.shape[1] // 2, self.center[0] - self.shape[0] // 2
        end_point = self.center[1] + self.shape[1] // 2, self.center[0] + self.shape[0] // 2
        # draw body
        self.appearance = cv2.rectangle(self.appearance, start_point, end_point, color, -1)
        # draw head
        self.appearance = cv2.circle(self.appearance, (end_point[0] - self.tier_len//2, self.center[0]), 6, (0, 250, 0), -1)
        # draw back tiers
        self.appearance = cv2.rectangle(self.appearance, (start_point[0], start_point[1] - 3),
                                        (start_point[0] + self.tier_len, start_point[1]), (255, 255, 255), -1)

        self.appearance = cv2.rectangle(self.appearance, (start_point[0], end_point[1] + 3),
                                         (start_point[0] + self.tier_len, end_point[1]), (255, 255, 255), -1)
        self.start_point = start_point
        self.end_point = end_point
        self._update_current_img()

        self.rotation_constant = 180/(np.pi * self.shape[1] * 0.75)
        self.init_pos = pos
        self.init_head = head_direction

    def step(self, action):
        # rotation: constant speed:
        dx_dy = np.array([np.cos(np.deg2rad(self.head_direction)), np.sin(np.deg2rad(self.head_direction))])
        net_force = action[0] * self.horsepower * dx_dy - self.friction * self.velocity
        self.velocity += net_force
        self.pos += self.velocity
        d_theta = self.rotation_constant * np.linalg.norm(self.velocity) * \
                               np.tan(np.deg2rad(self.steering_angle))
        self.head_direction += d_theta
        self.velocity = self.rotate_p(self.velocity, (0, 0), d_theta)
        self.head_direction %= 360
        self.steering_angle = min(max(action[1] * self.steering + self.steering_angle,
                                      -self.max_steering_angle), self.max_steering_angle)
        self._update_current_img()

    def rotate_p(self, p, origin, angle):
        theta = np.deg2rad(angle)
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        return np.dot(rot, p - origin) + origin

    def rotate_cnt(self, cnt, origin, angle):
        for p in cnt:
            p[:] = self.rotate_p(p, origin, angle)
        return cnt

    def reset(self):
        self.pos = self.init_pos
        self.head_direction = self.init_head
        self.steering_angle = 0
        self.velocity = np.zeros(2)
        self._update_current_img()

    def _update_current_img(self):
        self.cur_img = np.array(self.appearance, copy=True)
        left_tier_cnt = np.array([[self.end_point[0], self.end_point[1] + 3],
                                    [self.end_point[0], self.end_point[1]],
                                    [self.end_point[0] - self.tier_len, self.end_point[1]],
                                    [self.end_point[0] - self.tier_len, self.end_point[1] + 3]])

        right_tier_cnt = np.array([[self.end_point[0], self.start_point[1] - 3],
                                    [self.end_point[0], self.start_point[1]],
                                    [self.end_point[0] - self.tier_len, self.start_point[1]],
                                    [self.end_point[0] - self.tier_len, self.start_point[1] - 3]])

        axis_origin = [self.end_point[0] - self.tier_len//2, self.center[0]]

        self.cur_img = cv2.drawContours(self.cur_img,
                                        [self.rotate_cnt(left_tier_cnt, axis_origin, self.steering_angle)],
                                        0, (255, 255, 255), -1)

        self.cur_img = cv2.drawContours(self.cur_img,
                                         [self.rotate_cnt(right_tier_cnt, axis_origin, self.steering_angle)],
                                        0, (255, 255, 255), -1)

        self.cur_img = ndimage.rotate(self.cur_img, -self.head_direction, reshape=False, prefilter=False).astype(np.int8)


class Passenger(ABC):

    @abstractmethod
    def step(self, map_state):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def get_score(self):
        pass

    @abstractmethod
    def get_img(self):
        pass


class TypicalPassenger(Passenger):

    def step(self, map_state):
        pass

    def reset(self):
        pass

    def get_score(self):
        pass




