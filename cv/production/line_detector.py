import math
from collections import defaultdict


class AutomaticLineDetector:
    """
    Automatically determines a counting line from ByteTrack
    trajectories.

    Workflow:

        Track trajectories
              ↓
        Calculate movement
              ↓
        Find dominant direction
              ↓
        Generate perpendicular counting line
              ↓
        Lock the line
    """

    def __init__(
        self,
        min_track_length=10,
        min_movement=20,
        percentile=50
    ):
        """
        Parameters
        ----------
        min_track_length:
            Minimum number of positions required for a track.

        min_movement:
            Minimum movement in pixels required for a track
            to be considered useful.

        percentile:
            Position along the conveyor where the counting
            line should be placed.

            50 = approximately middle of movement region.
        """

        self.min_track_length = min_track_length
        self.min_movement = min_movement
        self.percentile = percentile

        # Track history
        #
        # {
        #     track_id: [
        #         (x, y),
        #         (x, y),
        #         ...
        #     ]
        # }
        self.track_history = defaultdict(list)

        # Final detected line
        self.line_start = None
        self.line_end = None

        # Once locked, line does not change
        self.locked = False

    # ========================================================
    # ADD TRACK POSITION
    # ========================================================

    def update_track(
        self,
        track_id,
        center_x,
        center_y
    ):
        """
        Add the current center position of a tracked object.
        """

        if self.locked:
            return

        self.track_history[
            track_id
        ].append(
            (
                float(center_x),
                float(center_y)
            )
        )

    # ========================================================
    # ADD MULTIPLE TRACKS
    # ========================================================

    def update(self, tracks):
        """
        Update detector with tracks from ByteTrack.

        Expected format:

        [
            {
                "track_id": 1,
                "bbox": [x1, y1, x2, y2],
                "class_name": "gear"
            },
            ...
        ]
        """

        if self.locked:
            return

        for track in tracks:

            track_id = track["track_id"]

            x1, y1, x2, y2 = track["bbox"]

            center_x = (
                x1 + x2
            ) / 2

            center_y = (
                y1 + y2
            ) / 2

            self.update_track(
                track_id,
                center_x,
                center_y
            )

    # ========================================================
    # VALID TRACKS
    # ========================================================

    def _get_valid_tracks(self):
        """
        Return tracks that contain enough movement
        information.
        """

        valid_tracks = []

        for track_id, points in (
            self.track_history.items()
        ):

            if len(points) < self.min_track_length:
                continue

            first_x, first_y = points[0]

            last_x, last_y = points[-1]

            dx = last_x - first_x
            dy = last_y - first_y

            movement = math.sqrt(
                dx * dx +
                dy * dy
            )

            if movement < self.min_movement:
                continue

            valid_tracks.append(
                {
                    "track_id": track_id,
                    "points": points,
                    "dx": dx,
                    "dy": dy,
                    "movement": movement
                }
            )

        return valid_tracks

    # ========================================================
    # DOMINANT MOVEMENT
    # ========================================================

    def _calculate_dominant_direction(
        self,
        valid_tracks
    ):
        """
        Calculate the average normalized movement vector.
        """

        if not valid_tracks:
            return None

        total_dx = 0.0
        total_dy = 0.0

        for track in valid_tracks:

            dx = track["dx"]
            dy = track["dy"]

            movement = track["movement"]

            # Normalize so long tracks don't dominate
            total_dx += dx / movement
            total_dy += dy / movement

        count = len(valid_tracks)

        avg_dx = total_dx / count
        avg_dy = total_dy / count

        magnitude = math.sqrt(
            avg_dx * avg_dx +
            avg_dy * avg_dy
        )

        if magnitude == 0:
            return None

        # Normalize
        avg_dx /= magnitude
        avg_dy /= magnitude

        return (
            avg_dx,
            avg_dy
        )

    # ========================================================
    # CALCULATE LINE
    # ========================================================

    def detect_line(
        self,
        frame_width,
        frame_height
    ):
        """
        Automatically generate a counting line.

        Returns:

            {
                "start": (x1, y1),
                "end": (x2, y2),
                "direction": (dx, dy),
                "movement_direction": "..."
            }

        Returns None if there isn't enough trajectory data.
        """

        if self.locked:

            return self.get_line()

        valid_tracks = (
            self._get_valid_tracks()
        )

        if len(valid_tracks) == 0:

            return None

        movement = (
            self._calculate_dominant_direction(
                valid_tracks
            )
        )

        if movement is None:

            return None

        dx, dy = movement

        # ----------------------------------------------------
        # Find all trajectory points
        # ----------------------------------------------------

        all_points = []

        for track in valid_tracks:

            all_points.extend(
                track["points"]
            )

        if not all_points:

            return None

        # ----------------------------------------------------
        # Calculate center of movement region
        # ----------------------------------------------------

        xs = [
            point[0]
            for point in all_points
        ]

        ys = [
            point[1]
            for point in all_points
        ]

        # ----------------------------------------------------
        # Project points onto movement direction
        # ----------------------------------------------------

        projections = []

        for x, y in all_points:

            projection = (
                x * dx +
                y * dy
            )

            projections.append(
                projection
            )

        projections.sort()

        # ----------------------------------------------------
        # Choose position along movement direction
        # ----------------------------------------------------

        index = int(
            len(projections)
            * self.percentile
            / 100
        )

        index = max(
            0,
            min(
                index,
                len(projections) - 1
            )
        )

        target_projection = (
            projections[index]
        )

        # ----------------------------------------------------
        # Find point closest to target projection
        # ----------------------------------------------------

        best_point = all_points[0]

        best_difference = float(
            "inf"
        )

        for x, y in all_points:

            projection = (
                x * dx +
                y * dy
            )

            difference = abs(
                projection -
                target_projection
            )

            if difference < best_difference:

                best_difference = difference

                best_point = (
                    x,
                    y
                )

        center_x, center_y = best_point

        # ----------------------------------------------------
        # Perpendicular vector
        #
        # Movement:
        #
        #       dx
        #       →
        #
        # Perpendicular:
        #
        #       -dy
        #        ↑
        #
        #        dx
        # ----------------------------------------------------

        line_dx = -dy
        line_dy = dx

        # ----------------------------------------------------
        # Calculate line length
        # ----------------------------------------------------

        line_length = math.sqrt(
            frame_width ** 2 +
            frame_height ** 2
        ) / 2

        # ----------------------------------------------------
        # Create line endpoints
        # ----------------------------------------------------

        x1 = (
            center_x
            - line_dx * line_length
        )

        y1 = (
            center_y
            - line_dy * line_length
        )

        x2 = (
            center_x
            + line_dx * line_length
        )

        y2 = (
            center_y
            + line_dy * line_length
        )

        # ----------------------------------------------------
        # Clip line to image boundaries
        # ----------------------------------------------------

        x1 = max(
            0,
            min(
                int(x1),
                frame_width - 1
            )
        )

        y1 = max(
            0,
            min(
                int(y1),
                frame_height - 1
            )
        )

        x2 = max(
            0,
            min(
                int(x2),
                frame_width - 1
            )
        )

        y2 = max(
            0,
            min(
                int(y2),
                frame_height - 1
            )
        )

        self.line_start = (
            x1,
            y1
        )

        self.line_end = (
            x2,
            y2
        )

        return self.get_line()

    # ========================================================
    # LOCK LINE
    # ========================================================

    def lock(self):
        """
        Lock the automatically detected line.

        After locking, new trajectories will not change it.
        """

        if (
            self.line_start is None
            or self.line_end is None
        ):
            raise RuntimeError(
                "Cannot lock line before "
                "detect_line() succeeds."
            )

        self.locked = True

    # ========================================================
    # GET LINE
    # ========================================================

    def get_line(self):
        """
        Return the current line configuration.
        """

        if (
            self.line_start is None
            or self.line_end is None
        ):
            return None

        return {
            "start": self.line_start,
            "end": self.line_end
        }

    # ========================================================
    # RESET
    # ========================================================

    def reset(self):
        """
        Reset detector completely.
        """

        self.track_history.clear()

        self.line_start = None
        self.line_end = None

        self.locked = False

    # ========================================================
    # STATUS
    # ========================================================

    def get_status(self):
        """
        Return useful debugging information.
        """

        valid_tracks = (
            self._get_valid_tracks()
        )

        return {
            "tracks_collected": len(
                self.track_history
            ),
            "valid_tracks": len(
                valid_tracks
            ),
            "line_detected": (
                self.line_start is not None
            ),
            "line_locked": self.locked
        }