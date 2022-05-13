import numpy as np
import time

class JumpDetector:

    def __init__(self, y_diff_grad_0=100, y_max_grad_1 = 40, timeframe=30):
        # self.y_diff_grad_0 = y_diff_grad_0
        # self.y_max_grad_1 = y_max_grad_1
        self.timeframe = timeframe
        self.margin = 1
        self.jump_dict = {}
        # self.half_jump_block = {}
        # self.half_jump_block_cntr = {}
        self.y_diff_left = 90
        self.y_diff_right = 70

    def reset_jump_dict(self):
        self.jump_dict = {}

    def check_if_jump_occured(self, jump_list, mode = 'half'):
        #Find maximum value and index in the jump_list (jump_list: y-values over time from the center of mass of the object)
        # ix_max = np.argmax(jump_list)
        # max_value = jump_list[ix_max]
        #
        # if ix_max >= self.margin and ix_max<self.timeframe - self.margin: #Check if the max-value is not on the edge of the list.
        #     argmin_value_left = np.argmin(jump_list[0:ix_max])
        #     min_value_left = jump_list[argmin_value_left]
        #     argmin_value_right = ix_max+np.argmin(jump_list[ix_max:])
        #     min_value_right = jump_list[argmin_value_right]
        #
        #     #Check if the difference between max and min values (from both the left and the right side of the max value) are bigger than a predefined threshold.
        #     cond1 = (max_value - min_value_left > self.y_diff_grad_0) if mode == 'half' else ((max_value - min_value_left > self.y_diff_grad_0) and (max_value - min_value_right > self.y_diff_grad_0/2))
        #
        #     if cond1 == True:
        #         return True
                # jump_list_grad = np.gradient(jump_list)
                # jump_list_grad = jump_list_grad[argmin_value_left:argmin_value_right+1]
                # # Check if the max and min values of the list's gradient values (np.gradient) are bigger than a predefined threshold.
                # # if (np.max(jump_list_grad) > self.y_max_grad_1 and np.min(jump_list_grad) < -(self.y_max_grad_1)):
                # cond2 = np.max(jump_list_grad) > self.y_max_grad_1 if mode == 'half' else (np.max(jump_list_grad) > self.y_max_grad_1 and np.min(jump_list_grad) < -(self.y_max_grad_1))
                #
                # if cond2:
                #     return True

        ix_min = np.argmin(jump_list)
        min_value = jump_list[ix_min]

        if ix_min >= self.margin and ix_min<len(jump_list) - self.margin: #Check if the max-value is not on the edge of the list.
            argmax_value_left = np.argmax(jump_list[0:ix_min])
            max_value_left = jump_list[argmax_value_left]
            argmax_value_right = ix_min+np.argmax(jump_list[ix_min:])
            max_value_right = jump_list[argmax_value_right]

            #Check if the difference between max and min values (from both the left and the right side of the max value) are bigger than a predefined threshold.
            cond1 = ((max_value_left - min_value >  self.y_diff_left) and (max_value_right - min_value > self.y_diff_right))

            if cond1 == True:
                return True

        return False

    def check_jump(self, centers_of_mass):

        # jumps_occured = []  #save all jumps into a list. List contains y-values of the center of mass for a specific object.
        jumps_init = []
        jumps_in_game = []
        keys_jump_dict = np.fromiter(self.jump_dict.keys(), dtype=int)
        keys_com = np.fromiter(centers_of_mass.keys(), dtype=int)
        new_ids = keys_com[np.isin(keys_com, keys_jump_dict, invert=True)] #new_ids: Objects which are not yet part of the sel.jump_dict dictionary.

        for ids in new_ids:
            self.jump_dict[ids] = []
            # self.half_jump_block[ids] = False
            # self.half_jump_block_cntr[ids] = 0

        for ids in centers_of_mass:
            # print(len(self.jump_dict[ids]))
            #At least self.timeframe values are required for performing a valid jump detection. In the beginning, the list is filled only without performing a test whether a jump has occured.
            if len(self.jump_dict[ids]) < (self.timeframe /2) :
                self.jump_dict[ids].append(centers_of_mass[ids][1]) #Append the y-value of the center of mass to the end of the list.

            else:
                if len(self.jump_dict[ids]) >= self.timeframe:
                    self.jump_dict[ids].pop(0) #Remove the oldest value from the list
                self.jump_dict[ids].append(centers_of_mass[ids][1]) #Append the y-value of the center of mass to the end of the list.

                # if self.half_jump_block[ids] == False:
                #     if self.check_if_jump_occured(self.jump_dict[ids], mode='half'):
                #         jumps_in_game.append(int(ids))
                #         zeit = str(time.time())
                #         np.save('/usr/local/lib/python3.5/dist-packages/testfolder/halfjump{}'.format(zeit[-4:]), self.jump_dict[ids])
                #         self.half_jump_block[ids] = True
                #         self.half_jump_block_cntr[ids] += 1
                #
                # elif self.half_jump_block_cntr[ids] > self.timeframe - self.margin:
                #     del self.jump_dict[ids], self.half_jump_block[ids], self.half_jump_block_cntr[ids]
                #
                # else:
                #     if self.check_if_jump_occured(self.jump_dict[ids], mode='full'):
                #         jumps_init.append(int(ids))
                #         zeit = str(time.time())
                #         np.save('/usr/local/lib/python3.5/dist-packages/testfolder/fulljump{}'.format(zeit[-4:]), self.jump_dict[ids])
                #         del self.jump_dict[ids], self.half_jump_block[ids], self.half_jump_block_cntr[ids]
                #     else:
                #         self.half_jump_block_cntr[ids] += 1
                if self.check_if_jump_occured(self.jump_dict[ids], mode='half'):
                    jumps_in_game.append(int(ids))
                    jumps_init.append(int(ids))
                    # zeit = str(time.time())
                    # np.save('/usr/local/lib/python3.5/dist-packages/testfolder/halfjumpron{}'.format(zeit[-4:]), self.jump_dict[ids])
                    del self.jump_dict[ids]

                # if self.check_if_jump_occured(self.jump_dict[ids], mode='full'):
                #     jumps_in_game.append(int(ids))
                #     zeit = str(time.time())
                #     np.save('/usr/local/lib/python3.5/dist-packages/testfolder/fulljump{}'.format(zeit[-4:]), self.jump_dict[ids])
                #     del self.jump_dict[ids]



        return jumps_init, jumps_in_game

