class KeyPointsLimiter:
    def __init__(self, screen_width, screen_height):
        self.screen_width = screen_width
        self.screen_height = screen_height

    def apply_limitations(self, pts):
        limits = ((pts >= 0).sum(axis=1) + (pts[:, 0] <= self.screen_width) + (pts[:, 1] <= self.screen_height)) == 4
        return pts[limits].reshape(-1,2)