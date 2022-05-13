import cv2
import numpy as np

# from tracking.kalman_filter import KalmanFilter
from pyeyeengine.object_detection.key_points_extractor import CentroidExtractor

MAX_DISTANCES_CENTROIDS_BTWN_FRAMES = 25  # pixels
MAX_TRACK_AGE = 5
MAX_NUM_TRACKS = 20


class Tracker:
    def __init__(self):
        self._tracked_objects_tracked_pts = []
        self._tracked_objects_id = []
        self._tracked_objects_age = []
        self._tracked_objects = []
        self.centroid_extractor = CentroidExtractor()

    def _add_new_tracks(self, tracked_pts, valid_matches, key_pts_voxels):
        if len(key_pts_voxels) == 0:
            return
        open_track_ids = np.setdiff1d(np.arange(MAX_NUM_TRACKS), self._tracked_objects_id)
        new_objects_inds = valid_matches.sum(axis=1) == 0
        new_key_pts = tracked_pts[new_objects_inds, :][:open_track_ids.shape[0], :]  # limit to MAX_NUM_TRACKS

        self._tracked_objects_tracked_pts = np.concatenate([self._tracked_objects_tracked_pts, new_key_pts], axis=0)
        self._tracked_objects_id = np.concatenate([self._tracked_objects_id, open_track_ids[:new_key_pts.shape[0]]],
                                                  axis=0)
        self._tracked_objects_age = np.concatenate([self._tracked_objects_age, np.zeros(new_key_pts.shape[0])], axis=0)
        new_objs_voxels = [voxels for idx, voxels in enumerate(key_pts_voxels) if new_objects_inds[idx]]
        self._tracked_objects += [TrackedObject(open_track_ids[idx], key_pts) for idx, key_pts in
                                  enumerate(new_objs_voxels) if idx < new_key_pts.shape[0]]

    def _remove_old_tracks(self):
        if len(self._tracked_objects) == 0:
            return

        [obj.save_track() for idx, obj in enumerate(self._tracked_objects) if
         self._tracked_objects_age[idx] > MAX_TRACK_AGE]

        self._tracked_objects_tracked_pts = self._tracked_objects_tracked_pts[
                                            self._tracked_objects_age <= MAX_TRACK_AGE, :]
        self._tracked_objects_id = self._tracked_objects_id[self._tracked_objects_age <= MAX_TRACK_AGE]
        self._tracked_objects = [obj for idx, obj in enumerate(self._tracked_objects) if
                                 self._tracked_objects_age[idx] <= MAX_TRACK_AGE]
        self._tracked_objects_age = self._tracked_objects_age[self._tracked_objects_age <= MAX_TRACK_AGE]

    def _add_objects_to_exisiting_track(self, tracked_pts, valid_matches, key_pts_voxels):
        if len(key_pts_voxels) == 0:
            self._tracked_objects_age += 1
            return

        self._tracked_objects_age[valid_matches.sum(axis=0) == 0] += 1
        self._tracked_objects_age[valid_matches.sum(axis=0) > 0] = 0
        for old_pt_idx in np.array(np.argwhere(valid_matches.sum(axis=0) > 0)).reshape(-1).tolist():
            new_pts_idx = np.argwhere(valid_matches[:, old_pt_idx]).reshape(-1)[0]
            self._tracked_objects_tracked_pts[old_pt_idx, :] = tracked_pts[new_pts_idx, :]
            self._tracked_objects[old_pt_idx].add_pts(key_pts_voxels[new_pts_idx])

    def _match_new_with_tracked_objects(self, tracked_pts):
        # scipy.optimize.linear_sum_assignment(cost_matrix) may be better
        if len(tracked_pts) == 0:
            return np.array([])
        distances = np.sqrt(
            np.power(tracked_pts[:, 0].reshape((-1, 1)) - self._tracked_objects_tracked_pts[:, 0].reshape((1, -1)), 2) + \
            np.power(tracked_pts[:, 1].reshape((-1, 1)) - self._tracked_objects_tracked_pts[:, 1].reshape((1, -1)), 2))
        is_min_distance = (np.expand_dims(distances.min(axis=1), axis=1) - np.expand_dims(distances.min(axis=0),
                                                                                          axis=0)) == 0
        valid_distance = distances < MAX_DISTANCES_CENTROIDS_BTWN_FRAMES
        return is_min_distance * valid_distance  # new key pts x tracked bojects key pts

    def track(self, object_centroid_voxels, key_pts_voxels):  # detected objects must have a method called set_id
        if len(object_centroid_voxels) > 0:
            tracked_pts = np.concatenate(object_centroid_voxels, axis=0).reshape(-1, 2)
        else:
            tracked_pts = np.array([])

        if len(self._tracked_objects_tracked_pts) == 0:
            if len(object_centroid_voxels) > 0:
                self._tracked_objects_tracked_pts = tracked_pts
                self._tracked_objects_id = np.arange(tracked_pts.shape[0])
                self._tracked_objects_age = np.zeros(tracked_pts.shape[0])
                self._tracked_objects = [TrackedObject(idx, key_pts) for idx, key_pts in enumerate(key_pts_voxels) if
                                         idx <= MAX_NUM_TRACKS]
        else:

            valid_matches = self._match_new_with_tracked_objects(tracked_pts)
            self._add_objects_to_exisiting_track(tracked_pts, valid_matches, key_pts_voxels)
            self._remove_old_tracks()
            self._add_new_tracks(tracked_pts, valid_matches, key_pts_voxels)

    def get_tracked_objects(self):
        return self._tracked_objects

    def get_key_points(self):
        return self._tracked_objects_tracked_pts

    def plot_tracker(self, rgb):
        rgb_to_show = rgb.copy()
        for obj in self._tracked_objects:
            for pt_idx in range(len(obj.key_pts_history) - 1):
                thickness = 1  # int((len(obj.key_pts_history)- pt_idx)/3+1)

                if (np.linalg.norm(obj.prediction_history[pt_idx] - obj.key_pts_history[pt_idx + 1], axis=1) < 5) and \
                        (np.linalg.norm(obj.prediction_history[pt_idx] - obj.key_pts_history[pt_idx + 1], axis=1) > 1):
                    # rgb_to_show = cv2.line(rgb_to_show,
                    #                        (int(obj.prediction_history[pt_idx][0, 0]),
                    #                         int(obj.prediction_history[pt_idx][0, 1])),
                    #                        (obj.key_pts_history[pt_idx + 1][0, 0], obj.key_pts_history[pt_idx + 1][0, 1]),
                    #                        color=(255, 0, 0), thickness=thickness)
                    rgb_to_show = cv2.circle(rgb_to_show,
                                             (obj.key_pts_history[pt_idx + 1][0, 0],
                                              obj.key_pts_history[pt_idx + 1][0, 1]),
                                             radius=3, color=(0, 255, 0), thickness=-1)

                rgb_to_show = cv2.line(rgb_to_show,
                                       (obj.key_pts_history[pt_idx][0, 0], obj.key_pts_history[pt_idx][0, 1]),
                                       (obj.key_pts_history[pt_idx + 1][0, 0], obj.key_pts_history[pt_idx + 1][0, 1]),
                                       color=obj.track_color, thickness=thickness)
                # if obj.is_click_history[pt_idx]:
                #     rgb_to_show = cv2.circle(rgb_to_show,
                #                              (obj.key_pts_history[pt_idx + 1][0, 0],
                #                               obj.key_pts_history[pt_idx + 1][0, 1]),
                #                              radius=3, color=(0, 255, 0), thickness=-1)

        cv2.imshow("tracks", cv2.resize(rgb_to_show, (0, 0), fx=2, fy=2))
        cv2.waitKey(3)


class TrackedObject:
    def __init__(self, id, key_pts):
        # note that the tracked obj does not know how long its been since it was last updated
        self.id = id
        self.key_pts = key_pts
        self.key_pts_history = [key_pts]
        self.track_color = (np.random.randint(0, 2) * 122, np.random.randint(0, 2) * 122, np.random.randint(0, 2) * 122)
        self.is_click = False
        self.is_click_history = []
        self.prediction_history = []
        # self.kalman_filter = KalmanFilter()

    def get_key_points(self):
        return self.key_pts
        # if len(self.prediction_history) > 1 and \
        #     np.sqrt(np.sum(np.array([self.prediction_history[-2] - self.key_pts_history[-1]])**2)) < 1.5:
        #     return self.prediction_history[-1]
        # else:
        #     return self.key_pts

    def add_pts(self, key_pts):
        self.key_pts = key_pts
        self.key_pts_history.append(key_pts)
        # self.key_pts_history = self.key_pts_history[-20:]
        self.is_clicked()
        self.is_click_history.append(self.is_click)
        # self.is_click_history = self.is_click_history[-20:]
        self.prediction_history.append(self.predict_key_pts(1))
        # self.prediction_history = self.prediction_history[-20:]

        # self.kalman_filter.kalman_xy(key_pts.reshape(-1, 2).mean(axis=0))

    def predict_key_pts(self, n_steps):
        if len(self.key_pts_history) > 4:
            velocity = np.diff(np.concatenate(self.key_pts_history[-6:], axis=0), axis=0)
            acceleration = np.diff(velocity, axis=0)
            jerk = np.diff(acceleration, axis=0)
            position_expected = self.key_pts_history[-1] + velocity.mean(axis=0) * n_steps + \
                                acceleration.mean(axis=0) * (n_steps ** 2) + jerk.mean(axis=0) * (n_steps ** 3)
            return position_expected.reshape(1, 2)
            # return np.diff(np.concatenate(
            #     self.key_pts_history[-6:], axis=0), axis=0).mean(axis=0) * n_steps + self.key_pts_history[-1]
        else:
            return self.key_pts_history[-1]
        # return self.key_pts + self.kalman_filter.x[2:].reshape(-1, 1) * n_steps

    def is_clicked(self):
        if len(self.key_pts_history) > 2:
            expected_point_loc = np.diff(np.concatenate(self.key_pts_history[-7:-1], axis=0), axis=0).mean(axis=0) + \
                                 self.key_pts_history[-2]
            trajectory_vel = np.diff(np.concatenate(self.key_pts_history[-7:-1], axis=0), axis=0).mean(axis=0)
            current_vel = np.diff(np.concatenate(self.key_pts_history[-2:], axis=0), axis=0).mean(axis=0)
            self.is_click = np.abs((trajectory_vel - current_vel) /
                                   np.maximum(np.maximum(trajectory_vel, current_vel), 1)).max() > 5
            if len(self.key_pts_history) > 10:
                hi = 5
            # velocity = np.diff(np.concatenate(self.key_pts_history, axis=0), axis=0)
            # self.is_click = (velocity[-2, :] - velocity[-1, :]).max() > 3

    def save_track(self):
        np.save("track", np.concatenate(self.key_pts_history, axis=0))
