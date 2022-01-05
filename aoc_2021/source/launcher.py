import numpy as np
import math
# target area: x=124..174, y=-123..-86
target_x = [124, 174]
target_y = [-123, -86]

# target_x = [20, 30]
# target_y = [-10, -5]

def get_pos(step, v_init):
    return math.floor(step * v_init - (step * (step - 1)) / 2)

def get_step_of_furthest_pos(v_init):
    return max(v_init + 0.5,0)

def is_in_y(y):
    return ((y >= target_y[0]) and (y <= target_y[1]))

def is_in_x(x):
    return ((x >= target_x[0]) and (x <= target_x[1]))

def is_in(x, y):
    return (is_in_x(x) and is_in_y(y))

def part1():
    cur_best = 0
    for vy in range(-200, 200):
        for vx in range(1, 200):
            step_y_max = get_step_of_furthest_pos(vy)
            y_max = get_pos(step_y_max, vy)
            step_x_end = get_step_of_furthest_pos(vx)
            x_max = get_pos(step_x_end, vx)
            for step in range(math.floor(step_y_max), math.ceil(step_x_end)+500): #math.ceil(step_x_end)):
                x_ = max(get_pos(step,vx), x_max)
                y_ = get_pos(step,vy)
                if is_in(x_, y_):
                    if y_max > cur_best:
                        print("{},{} -> {}".format(vx, vy, y_max))
                        cur_best = y_max
                elif x_ > target_x[1]:
                    break
                elif y_ < target_y[0]:
                    break

def part2():
    counter_distinct = 0
    for vy in range(-600, 600):
        for vx in range(1, 550):
            flag=False
            has_reached_max = False
            step_x_end = get_step_of_furthest_pos(vx)
            x_max = get_pos(step_x_end, vx)
            if x_max >= target_x[0]:
                for step in range(0, math.ceil(step_x_end)+1000): #math.ceil(step_x_end)):
                    x_ = get_pos(step,vx)
                    if x_ >= x_max:
                        has_reached_max = True
                    if has_reached_max:
                        x_ = x_max
                    if x_ < 0:
                        x_ = x_max

                    y_ = get_pos(step,vy)
                    if is_in(x_, y_):
                        flag = True
                    elif x_ > target_x[1]:
                        break
                    elif y_ < target_y[0]:
                        break
            if flag:
                print("{},{}".format(vx, vy))
                counter_distinct += 1
    return counter_distinct

