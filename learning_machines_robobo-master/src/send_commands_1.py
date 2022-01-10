#!/usr/bin/env python2
from __future__ import print_function
import time
import robobo
import cv2
import sys
import signal
import prey
import numpy as np
import argparse
import random
from numpy.random import rand


def terminate_program(signal_number, frame):
    print("Ctrl-C received, terminating program")
    sys.exit(1)
#
# def observe_state(prev_state, target_close_c, target_close_l, target_close_r, target_far_l, target_far_c, target_far_r, front, left, right):
#     best = max(front,left,right)
#     best_mean = max(target_close_c.mean(), target_close_l.mean(), target_close_r.mean(), target_far_l.mean(), target_far_c.mean(), target_far_r.mean())
#     if best_mean == target_close_l.mean() and target_close_l.mean() > 0.05:
#         state = 0
#         reward = 10
#         return state, reward
#     if best_mean == target_close_c.mean() and target_close_c.mean() > 0.05:
#         state = 1
#         reward = 20
#         return state, reward
#     if best_mean == target_close_r.mean() and target_close_r.mean() > 0.05:
#         state = 2
#         reward = 10
#         return state, reward
#     if best_mean == target_far_l.mean() and target_far_l.mean() > 0.05:
#         state = 3
#         reward = 5
#         return state, reward
#     if best_mean == target_far_c.mean() and target_far_c.mean() > 0.05:
#         state = 4
#         reward = 5
#         return state, reward
#     if best_mean == target_far_r.mean() and target_far_r.mean() > 0.05:
#         state = 5
#         reward = 5
#         return state, reward
#     if front==False and left==False and right == False:
#         if prev_state == 3:
#             state = 9
#             reward = -10
#             return state, reward
#         if prev_state == 5:
#             state = 11
#             reward = -10
#             return state, reward
#         else:
#             state = 10 #FREE
#             reward = -10
#             return state, reward
#     if best == left  and target_close_l.mean() == 0:
#         state = 6
#         reward = -10
#         return state, reward
#     if best == front and target_close_c.mean() == 0:
#         state = 7 #FRONTobject
#         reward = -10
#         return state, reward
#     if best == right  and target_close_r.mean() == 0:
#         state = 8 #RIGHTobject
#         reward = -10
#         return state, reward
#     else:
#         state = 4
#         reward = 5
#         return state, reward
#
# def maskered(images):
#     hsv0 = cv2.cvtColor(images[0], cv2.COLOR_BGR2HSV)
#     hsv1 = cv2.cvtColor(images[1], cv2.COLOR_BGR2HSV)
#     hsv2 = cv2.cvtColor(images[2], cv2.COLOR_BGR2HSV)
#     hsv3 = cv2.cvtColor(images[3], cv2.COLOR_BGR2HSV)
#     hsv4 = cv2.cvtColor(images[4], cv2.COLOR_BGR2HSV)
#     hsv5 = cv2.cvtColor(images[5], cv2.COLOR_BGR2HSV)
#     lower_red = np.array([0, 50, 50])
#     upper_red = np.array([10, 255, 255])
#     mask00 = cv2.inRange(hsv0, lower_red, upper_red)
#     mask01 = cv2.inRange(hsv1, lower_red, upper_red)
#     mask02 = cv2.inRange(hsv2, lower_red, upper_red)
#     mask03 = cv2.inRange(hsv3, lower_red, upper_red)
#     mask04 = cv2.inRange(hsv4, lower_red, upper_red)
#     mask05 = cv2.inRange(hsv5, lower_red, upper_red)
#     lower_red = np.array([170, 50, 50])
#     upper_red = np.array([180, 255, 255])
#     mask10 = cv2.inRange(hsv0, lower_red, upper_red)
#     mask11 = cv2.inRange(hsv1, lower_red, upper_red)
#     mask12 = cv2.inRange(hsv2, lower_red, upper_red)
#     mask13 = cv2.inRange(hsv3, lower_red, upper_red)
#     mask14 = cv2.inRange(hsv4, lower_red, upper_red)
#     mask15 = cv2.inRange(hsv5, lower_red, upper_red)
#     mask0 = cv2.bitwise_or(mask00, mask10)
#     mask1 = cv2.bitwise_or(mask01, mask11)
#     mask2 = cv2.bitwise_or(mask02, mask12)
#     mask3 = cv2.bitwise_or(mask03, mask13)
#     mask4 = cv2.bitwise_or(mask04, mask14)
#     mask5 = cv2.bitwise_or(mask04, mask15)
#     target_close_l = cv2.bitwise_and(images[3], images[3], mask=mask3)
#     target_close_c = cv2.bitwise_and(images[4], images[4], mask=mask4)
#     target_close_r = cv2.bitwise_and(images[5], images[5], mask=mask5)
#     target_far_l = cv2.bitwise_and(images[0], images[0], mask=mask0)
#     target_far_c = cv2.bitwise_and(images[1], images[1], mask=mask1)
#     target_far_r = cv2.bitwise_and(images[2], images[2], mask=mask2)
#
#     return target_close_c, target_close_l, target_close_r, target_far_l, target_far_c, target_far_r
#
# def find_target(image):
#     resized = cv2.resize(image, None, fx=0.5, fy=0.5)
#     hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
#     lower_red = np.array([0, 50, 50])
#     upper_red = np.array([10, 255, 255])
#     mask0 = cv2.inRange(hsv, lower_red, upper_red)
#     lower_red = np.array([170, 50, 50])
#     upper_red = np.array([180, 255, 255])
#     mask1 = cv2.inRange(hsv, lower_red, upper_red)
#     mask = cv2.bitwise_or(mask0, mask1)
#     img = resized
#     img2 = img
#
#     height, width, channels = img.shape
#     # Number of pieces Horizontally
#     CROP_W_SIZE  = 3
#     # Number of pieces Vertically to each Horizontal
#     CROP_H_SIZE = 2
#     images = []
#     for ih in range(CROP_H_SIZE ):
#         for iw in range(CROP_W_SIZE ):
#
#             x = int(width/CROP_W_SIZE * iw )
#             y = int(height/CROP_H_SIZE * ih)
#             h = int(height / CROP_H_SIZE)
#             w = int(width / CROP_W_SIZE )
#             img = img[y:y+h, x:x+w]
#             images.append(img)
#             # NAME = str(time.time())
#             # cv2.imwrite(str(time.time()) +  ".png",img)
#             img = img2
#     target_close_c, target_close_l, target_close_r, target_far_l, target_far_c, target_far_r = maskered(images)
#     # target_close_c, target_close_l, target_close_r, target_far = random.uniform(0,1), random.uniform(0,1), random.uniform(0,1), random.uniform(0,1)
#     return target_close_c, target_close_l, target_close_r, target_far_l, target_far_c, target_far_r
#
# def q_learning(q_table, state, alpha = 0.1, epsilon = 0, gamma = 0.9):
#     if random.uniform(0, 1) < epsilon:
#         action = np.random.choice([0, 1, 2, 3, 4, 5])
#         act = actions(action)
#         rob.sleep(0.1)
#         next_ir = np.array(rob.read_irs())
#         front = next_ir[5]
#         right = next_ir[3]
#         left = next_ir[-1]
#         image = rob.get_image_front()
#         c,l,r,far_l, far_c, far_r = find_target(image)
#         next_state, reward = observe_state(state,c,l,r, far_l, far_c, far_r, front, left, right)
#         print("epsilon has decided to change the track:", state, act)
#         # q_table[state,act] = (1-alpha) * q_table[state,act] + alpha * (reward + gamma * np.argmax(q_table[next_state,:]))
#     else:
#         action = np.argmax(q_table[state]) # Exploit learned values
#         act = actions(action)
#         rob.sleep(0.1)
#         next_ir = np.array(rob.read_irs())
#         front = next_ir[5]
#         right = next_ir[3]
#         left = next_ir[-1]
#         image = rob.get_image_front()
#         c,l,r,far_l, far_c, far_r = find_target(image)
#         next_state, reward = observe_state(state,c,l,r, far_l, far_c, far_r, front, left, right)
#         # q_table[state,act] = (1-alpha) * q_table[state,act] + alpha * (reward + gamma * np.argmax(q_table[next_state,:]))
#     return q_table, reward
#
# def actions(orientation):
#     #Lvl1 (Max:10, Turn:10)
#     #Lvl2 (Max:20 Turn:10)
#     #Lvl3 (Max:40, Turn:20)
#     if orientation == 0:
#         rob.move(20,20,300)
#         return orientation
#     # UpLeft
#     if orientation == 1:
#         rob.move(20,30,300)
#         return orientation
#     # upright
#     if orientation == 2:
#         rob.move(30,20,300)
#         return orientation
#     # down
#     if orientation == 3:
#         rob.move(-20,-20,300)
#         return orientation
#     # Set
#     if orientation == 4:
#         if np.random.random(1) < 0.75:
#             rob.move(20,-20,300)
#         else:
#             rob.move(20,20,300)
#         return orientation
#     if orientation == 5:
#         if np.random.random(1) < 0.75:
#             rob.move(-20,20,300)
#         else:
#             rob.move(20,20,300)
#         return orientation
#
# def main():
#     global rob
#     signal.signal(signal.SIGINT, terminate_program)
#     # rob = robobo.HardwareRobobo(camera=True).connect(address="192.168.2.16")
#     n = 1000
#     epochs = 1
#     rob = robobo.SimulationRobobo().connect(address='145.108.94.115', port=19997)
#     # q_table = np.zeros([12,6])
#     q_table = np.array([[4.04382538, 9.08914904, 4.08809245, 5.6205059, 5.0917869, 5.08168581],
#                         [10.7471202, 3.7193, 3.7923254, 4.873178, 3.999668, 6.04541],
#                         [6.29800066, 2.0, 9.32419968, -0.67704, 4.07044, 5.569336],
#                         [4.95748641, 0.6044, 4.19668879, 4.14025809, 2.95745, 2.49253152],
#                         [5.53953297, 6.74134012, 8.12132661, 4.009388, 5.52886574, 3.68104978],
#                         [1.87727663, 0.790112982, 4.19940093, 1.28180346, 1.75057404, 1.99137532],
#                         [-4.33805275, -4.30660619, -3.99381793, -4.48572763, -4.03338755, -2.62042215],
#                         [-2.86645338, -2.33295548, -2.56911925, -2.80654342, -2.69478782, -2.56222587],
#                         [-4.49047836, -3.94078157, -5.04447164, -5.07775868, -4.68199334, -3.78946255],
#                         [-0.5981, -0.59, -0.59, -0.5765, -0.6224, -0.5455],
#                         [-2.72223787, -3.03925178, -2.43331858, -3.38668665, -1.39782044, -2.58788025],
#                         [-0.77, -0.90058631, 0.241603915, -0.00864775, -0.49595, 0.241997478]])
#     cum_reward = []
#     time_total = []
#     prey_list = []
#     close = []
#     streaks = []
#     eps = 0
#     for e in range(epochs):
#         rob.play_simulation()
#         print("-----------Epoch:", e+1, "Epsilon:", eps)
#         # prey_robot = robobo.HardWareRobobo(camera=True).connect(address='145.108.94.115', port=19989)
#         # prey_controller = prey.Prey(robot=prey_robot, level=2)
#         # prey_controller.start()
#         state = 10
#         cum = 0
#         rob.set_phone_tilt(90, 100)
#         prey_found = 0
#         prey_close = 0
#         streak = 0
#         max_streak = 0
#         time_caught = []
#         for i in range(n):
#             print("--------------Step: ", i+1)
#             ir = np.array(rob.read_irs())
#             front = ir[5]
#             right = ir[3]
#             left = ir[-1]
#             image = rob.get_image_front()
#             target_close_c, target_close_l, target_close_r, target_far_l, target_far_c, target_far_r = find_target(image)
#             if front > 0:
#                 if front > 90:
#                     color =  max(target_close_c.mean(), target_close_l.mean(), target_close_r.mean())
#                     if color > 0.25:
#                         rob.talk("t'acchiappo")
#                         time_caught.append(i)
#
#             state, _ = observe_state(state, target_close_c, target_close_l, target_close_r, target_far_l, target_far_c, target_far_r, front, left, right)
#             if state in [0,1,2,3,4,5]:
#                 print("Prey Found!")
#                 prey_found += 1
#                 streak += 1
#                 if streak > max_streak:
#                     max_streak = streak
#             else:
#                 streak = 0
#             if state in [0,1,2]:
#                 print("Prey Close!")
#                 prey_close += 1
#             q_table, reward = q_learning(q_table, state, epsilon = eps)
#             # time.sleep(0.1)
#             cum += reward
#             print(q_table)
#         time_total.append(time_caught)
#         close.append(prey_close)
#         prey_list.append(prey_found)
#         streaks.append(max_streak)
#         cum_reward.append(cum)
#         # prey_controller.stop()
#         # prey_controller.join()
#         # prey_robot.disconnect()
#         rob.stop_world()
#         rob.wait_for_stop()
#         # eps -= 0.1
#
#     print(cum_reward)
#     print(time_caught)
#     print(prey_list)
#     print(close)
#     print(streaks)
#
def _sensor_better_reading(sensors_values):
    """
    Normalising simulation sensor reading due to reuse old code
    :param sensors_values:
    :return:
    """
    old_min = 0
    old_max = 0.20
    new_min = 20000
    new_max = 0
    return [0 if value is False else (((value - old_min) * (new_max - new_min)) / (old_max - old_min)) + new_min for value in sensors_values]

def observe_state(prev_state, target_close_c, target_close_l, target_close_r, target_far_l, target_far_c, target_far_r, front, left, right):
    best = max(front,left,right)
    best_mean = max(target_close_c.mean(), target_close_l.mean(), target_close_r.mean(), target_far_l.mean(), target_far_c.mean(), target_far_r.mean())
    if best_mean == target_close_l.mean() and target_close_l.mean() > 0.05:
        state = 0
        reward = 10
        return state, reward
    if best_mean == target_close_c.mean() and target_close_c.mean() > 0.05:
        state = 1
        reward = 20
        return state, reward
    if best_mean == target_close_r.mean() and target_close_r.mean() > 0.05:
        state = 2
        reward = 10
        return state, reward
    if best_mean == target_far_l.mean() and target_far_l.mean() > 0.05:
        state = 3
        reward = 5
        return state, reward
    if best_mean == target_far_c.mean() and target_far_c.mean() > 0.05:
        state = 4
        reward = 5
        return state, reward
    if best_mean == target_far_r.mean() and target_far_r.mean() > 0.05:
        state = 5
        reward = 5
        return state, reward
    if front==False and left==False and right == False:
        if prev_state == 3:
            state = 9
            reward = -10
            return state, reward
        if prev_state == 5:
            state = 11
            reward = -10
            return state, reward
        else:
            state = 10 #FREE
            reward = -10
            return state, reward
    if best == left  and target_close_l.mean() == 0:
        state = 6
        reward = -10
        return state, reward
    if best == front and target_close_c.mean() == 0:
        state = 7 #FRONTobject
        reward = -10
        return state, reward
    if best == right  and target_close_r.mean() == 0:
        state = 8 #RIGHTobject
        reward = -10
        return state, reward
    else:
        state = 4
        reward = 5
        return state, reward

def maskered(images):
    hsv0 = cv2.cvtColor(images[0], cv2.COLOR_BGR2HSV)
    hsv1 = cv2.cvtColor(images[1], cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(images[2], cv2.COLOR_BGR2HSV)
    hsv3 = cv2.cvtColor(images[3], cv2.COLOR_BGR2HSV)
    hsv4 = cv2.cvtColor(images[4], cv2.COLOR_BGR2HSV)
    hsv5 = cv2.cvtColor(images[5], cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([30, 255, 255])
    mask00 = cv2.inRange(hsv0, lower_red, upper_red)
    mask01 = cv2.inRange(hsv1, lower_red, upper_red)
    mask02 = cv2.inRange(hsv2, lower_red, upper_red)
    mask03 = cv2.inRange(hsv3, lower_red, upper_red)
    mask04 = cv2.inRange(hsv4, lower_red, upper_red)
    mask05 = cv2.inRange(hsv5, lower_red, upper_red)
    lower_red = np.array([30, 50, 50])
    upper_red = np.array([60, 255, 255])
    mask10 = cv2.inRange(hsv0, lower_red, upper_red)
    mask11 = cv2.inRange(hsv1, lower_red, upper_red)
    mask12 = cv2.inRange(hsv2, lower_red, upper_red)
    mask13 = cv2.inRange(hsv3, lower_red, upper_red)
    mask14 = cv2.inRange(hsv4, lower_red, upper_red)
    mask15 = cv2.inRange(hsv5, lower_red, upper_red)
    mask0 = cv2.bitwise_or(mask00, mask10)
    mask1 = cv2.bitwise_or(mask01, mask11)
    mask2 = cv2.bitwise_or(mask02, mask12)
    mask3 = cv2.bitwise_or(mask03, mask13)
    mask4 = cv2.bitwise_or(mask04, mask14)
    mask5 = cv2.bitwise_or(mask04, mask15)
    target_close_l = cv2.bitwise_and(images[3], images[3], mask=mask3)
    target_close_c = cv2.bitwise_and(images[4], images[4], mask=mask4)
    target_close_r = cv2.bitwise_and(images[5], images[5], mask=mask5)
    target_far_l = cv2.bitwise_and(images[0], images[0], mask=mask0)
    target_far_c = cv2.bitwise_and(images[1], images[1], mask=mask1)
    target_far_r = cv2.bitwise_and(images[2], images[2], mask=mask2)

    return target_close_c, target_close_l, target_close_r, target_far_l, target_far_c, target_far_r

def find_target(image):
    resized = cv2.resize(image, None, fx=0.5, fy=0.5)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])
    mask0 = cv2.inRange(hsv, lower_red, upper_red)
    lower_red = np.array([170, 50, 50])
    upper_red = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)
    mask = cv2.bitwise_or(mask0, mask1)
    img = resized
    img2 = img

    height, width, channels = img.shape
    # Number of pieces Horizontally
    CROP_W_SIZE  = 3
    # Number of pieces Vertically to each Horizontal
    CROP_H_SIZE = 2
    images = []
    for ih in range(CROP_H_SIZE ):
        for iw in range(CROP_W_SIZE ):

            x = int(width/CROP_W_SIZE * iw )
            y = int(height/CROP_H_SIZE * ih)
            h = int(height / CROP_H_SIZE)
            w = int(width / CROP_W_SIZE )
            img = img[y:y+h, x:x+w]
            images.append(img)
            # NAME = str(time.time())
            # cv2.imwrite(str(time.time()) +  ".png",img)
            img = img2
    target_close_c, target_close_l, target_close_r, target_far_l, target_far_c, target_far_r = maskered(images)
    # target_close_c, target_close_l, target_close_r, target_far = random.uniform(0,1), random.uniform(0,1), random.uniform(0,1), random.uniform(0,1)
    return target_close_c, target_close_l, target_close_r, target_far_l, target_far_c, target_far_r

def q_learning(q_table, state, alpha = 0.1, epsilon = 0, gamma = 0.9):
    if random.uniform(0, 1) < epsilon:
        action = np.random.choice([0, 1, 2, 3, 4, 5])
        act = actions(action)
        rob.sleep(0.1)
        next_ir = np.array(rob.read_irs())
        front = next_ir[5]
        right = next_ir[3]
        left = next_ir[-1]
        image = rob.get_image_front()
        c,l,r,far_l, far_c, far_r = find_target(image)
        next_state, reward = observe_state(state,c,l,r, far_l, far_c, far_r, front, left, right)
        print("epsilon has decided to change the track:", state, act)
        # q_table[state,act] = (1-alpha) * q_table[state,act] + alpha * (reward + gamma * np.argmax(q_table[next_state,:]))
    else:
        action = np.argmax(q_table[state]) # Exploit learned values
        act = actions(action)
        rob.sleep(0.1)
        next_ir = np.array(rob.read_irs())
        front = next_ir[5]
        right = next_ir[3]
        left = next_ir[-1]
        image = rob.get_image_front()
        c,l,r,far_l, far_c, far_r = find_target(image)
        next_state, reward = observe_state(state,c,l,r, far_l, far_c, far_r, front, left, right)
        # q_table[state,act] = (1-alpha) * q_table[state,act] + alpha * (reward + gamma * np.argmax(q_table[next_state,:]))
    return q_table, reward

def actions(orientation):
    #Lvl1 (Max:10, Turn:10)
    #Lvl2 (Max:20 Turn:10)
    #Lvl3 (Max:40, Turn:20)
    if orientation == 0:
        rob.move(20,20,300)
        return orientation
    # UpLeft
    if orientation == 1:
        rob.move(20,30,300)
        return orientation
    # upright
    if orientation == 2:
        rob.move(30,20,300)
        return orientation
    # down
    if orientation == 3:
        rob.move(-20,-20,300)
        return orientation
    # Set
    if orientation == 4:
        if np.random.random(1) < 0.75:
            rob.move(20,-20,300)
        else:
            rob.move(20,20,300)
        return orientation
    if orientation == 5:
        if np.random.random(1) < 0.75:
            rob.move(-20,20,300)
        else:
            rob.move(20,20,300)
        return orientation

def main():
    global rob
    signal.signal(signal.SIGINT, terminate_program)
    # rob = robobo.HardwareRobobo(camera=True).connect(address="192.168.2.16")
    n = 1000
    epochs = 1
    rob = robobo.SimulationRobobo().connect(address='145.108.94.115', port=19997)
    # q_table = np.zeros([12,6])
    q_table = np.array([[4.04382538, 9.08914904, 4.08809245, 5.6205059, 5.0917869, 5.08168581],
                        [10.7471202, 3.7193, 3.7923254, 4.873178, 3.999668, 6.04541],
                        [6.29800066, 2.0, 9.32419968, -0.67704, 4.07044, 5.569336],
                        [4.95748641, 0.6044, 4.19668879, 4.14025809, 2.95745, 2.49253152],
                        [5.53953297, 6.74134012, 8.12132661, 4.009388, 5.52886574, 3.68104978],
                        [1.87727663, 0.790112982, 4.19940093, 1.28180346, 1.75057404, 1.99137532],
                        [-4.33805275, -4.30660619, -3.99381793, -4.48572763, -4.03338755, -2.62042215],
                        [-2.86645338, -2.33295548, -2.56911925, -2.80654342, -2.69478782, -2.56222587],
                        [-4.49047836, -3.94078157, -5.04447164, -5.07775868, -4.68199334, -3.78946255],
                        [-0.5981, -0.59, -0.59, -0.5765, -0.6224, -0.5455],
                        [-2.72223787, -3.03925178, -2.43331858, -3.38668665, -1.39782044, -2.58788025],
                        [-0.77, -0.90058631, 0.241603915, -0.00864775, -0.49595, 0.241997478]])
    cum_reward = []
    time_total = []
    prey_list = []
    close = []
    streaks = []
    eps = 0
    for e in range(epochs):
        rob.play_simulation()
        print("-----------Epoch:", e+1, "Epsilon:", eps)
        # prey_robot = robobo.HardWareRobobo(camera=True).connect(address='145.108.94.115', port=19989)
        # prey_controller = prey.Prey(robot=prey_robot, level=2)
        # prey_controller.start()
        state = 10
        cum = 0
        rob.set_phone_tilt(90, 100)
        prey_found = 0
        prey_close = 0
        streak = 0
        max_streak = 0
        time_caught = []
        for i in range(n):
            print("--------------Step: ", i+1)
            ir = np.array(rob.read_irs())
            front = ir[5]
            right = ir[3]
            left = ir[-1]
            image = rob.get_image_front()
            target_close_c, target_close_l, target_close_r, target_far_l, target_far_c, target_far_r = find_target(image)
            if front > 0:
                if front > 90:
                    color =  max(target_close_c.mean(), target_close_l.mean(), target_close_r.mean())
                    if color > 0.25:
                        rob.talk("t'acchiappo")
                        time_caught.append(i)

            state, _ = observe_state(state, target_close_c, target_close_l, target_close_r, target_far_l, target_far_c, target_far_r, front, left, right)
            if state in [0,1,2,3,4,5]:
                print("Prey Found!")
                prey_found += 1
                streak += 1
                if streak > max_streak:
                    max_streak = streak
            else:
                streak = 0
            if state in [0,1,2]:
                print("Prey Close!")
                prey_close += 1
            q_table, reward = q_learning(q_table, state, epsilon = eps)
            # time.sleep(0.1)
            cum += reward
            print(q_table)
        time_total.append(time_caught)
        close.append(prey_close)
        prey_list.append(prey_found)
        streaks.append(max_streak)
        cum_reward.append(cum)
        # prey_controller.stop()
        # prey_controller.join()
        # prey_robot.disconnect()
        rob.stop_world()
        rob.wait_for_stop()
        # eps -= 0.1

    print(cum_reward)
    print(time_caught)
    print(prey_list)
    print(close)
    print(streaks)



if __name__ == "__main__":
    main()
