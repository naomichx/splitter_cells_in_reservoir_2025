import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, Rectangle, Wedge, FancyArrowPatch
from maze import line_intersect
np.random.seed(5)


def degree_range(n):
    """
    Divides 360 degrees into `n` equal parts and calculates midpoints.
    """
    start = np.linspace(0, 360, n, endpoint=False)
    end = np.linspace(360 / n, 360, n, endpoint=False)
    mid = start + (360 / (2 * n))
    return np.column_stack([start, end]), mid


def rot_text(angle):
    """
    Rotates text for better alignment.
    """
    if angle >= 90 and angle <= 270:
        return angle + 180
    return angle


def gauge(ax, arrow=0, title='', scale=0.3, position=(0.8, 0.8)):
    """
    Gauge function with 360-degree scale divided into 8 equal parts,
    labeled at the intersection of each slice in radians as multiples of π.

    Parameters:
    - ax: Matplotlib axis object.
    - colors: Colormap or list of colors for the segments.
    - arrow: Position of the arrow (1-based index).
    - title: Title text displayed at the center of the gauge.
    - scale: Scaling factor for the gauge size (default 0.5).
    - position: (x, y) position of the gauge center in axes coordinates (default upper-right).
    """
    N = 8  # Divide gauge into 8 equal parts
    colors = ['grey', 'black', 'grey', 'black', 'grey', 'black', 'grey', 'black']

    labels = ['π/2', '', 'π', '', '-π/2', '', '0', '', ''] #when rotating the figure

    def degree_range(n):
        step = 360 / n
        return [(i * step, (i + 1) * step) for i in range(n)], [i * step for i in range(n + 1)]

    def rot_text(angle):
        if 90 < angle < 270:
            return angle + 180
        return angle

    ang_range, label_positions = degree_range(N)
    # Create a new inset axis for scaling and positioning
    inset_ax = ax.inset_axes([position[0], position[1], scale, scale])
    patches = []
    for ang, c in zip(ang_range, colors):
        patches.append(Wedge((0., 0.), .25, *ang, facecolor='w', lw=1))
        patches.append(Wedge((0., 0.), .25, *ang, width=0.02, facecolor='grey', lw=1, alpha=0.5))

    [inset_ax.add_patch(p) for p in patches]

    for pos, lab in zip(label_positions, labels):
        inset_ax.text(0.30 * np.cos(np.radians(pos)), 0.3 * np.sin(np.radians(pos)), lab,
                      horizontalalignment='center', verticalalignment='center', fontsize=8,
                      fontweight='bold', rotation=rot_text(270))

    inset_ax.text(0, -0.05, title, horizontalalignment='center',
                  verticalalignment='center', fontsize=12, fontweight='bold')

    pos = label_positions[arrow]
    orientation = FancyArrowPatch((0, 0), (0.23 * np.cos(np.radians(pos)), 0.23 * np.sin(np.radians(pos))),
                             arrowstyle='->', mutation_scale=20, color='C3', alpha=0.5, linewidth=2)

    inset_ax.add_patch(orientation)
    inset_ax.add_patch(Circle((0, 0), radius=0.02, facecolor='k'))
    inset_ax.add_patch(Circle((0, 0), radius=0.01, facecolor='w', zorder=11))
    inset_ax.set_frame_on(False)
    inset_ax.set_xticks([])
    inset_ax.set_yticks([])
    inset_ax.axis('equal')

    return orientation, inset_ax, label_positions


class Bot:

    def __init__(self, save_bot_sates, sensor_size, decoder):
        self.size = 10
        self.position = 150, 150
        self.orientation = 0
        self.n_sensors = 8


        # Direction flag
        self.left_cue = True  # start by going left
        self.right_cue = False
        self.left_cue_prev = True
        self.right_cue_prev = False

        self.decoder = decoder
        self.predicted_pos = self.position
        self.decision_pred = None
        self.orientation_pred = None
        self.orientation_gauge =  None

        A = np.linspace(-np.pi / 2, +np.pi / 2, self.n_sensors + 2, endpoint=True)[1:-1]
        self.sensors = {
            "angle": A,
            "range": sensor_size*np.ones((self.n_sensors, 1)), # initial sensor size: 60
            "value": np.ones((self.n_sensors, 1))}
        self.sensors["range"][3:5] *= 1.25

        self.save_bot_sates = save_bot_sates
        self.all_orientations = []
        self.all_d_orientations = []
        self.all_positions = []
        self.all_sensors_vals = []
        self.all_cues = []

        self.all_predicted_or = []
        self.all_predicted_dec = []
        self.all_predicted_pos = []

        # For RR-LL
        self.enter_corridor = False
        self.iter_right = 2
        self.iter_left = 0

    def draw(self, ax):
        """Render the bot in the maze."""
        # Sensors
        # Two points per segment
        n = 2 * len(self.sensors["angle"])
        sensors = LineCollection(np.zeros((n, 2, 2)),
                                 colors=["0.75", "0.00"] * n,
                                 linewidths=[0.75, 1.00] * n,
                                 linestyles=["--", "-"] * n)
        # Body
        body = Circle(self.position, self.size, zorder=20, edgecolor="black",
                      facecolor=(1, 1, 1, .75))

        # Head
        P = np.zeros((1, 2, 2))
        P[0, 0] = self.position
        P[0, 1] = P[0, 1] + self.size * np.array([np.cos(self.orientation),
                                                  np.sin(self.orientation)])
        head = LineCollection(P, colors="black", zorder=30)

        if self.decoder:
            predicted_pos_artist = Circle(self.position, 5, zorder=20, edgecolor="red", facecolor='red')
            predicted_decision_artist = FancyArrowPatch((250, 100), (250, 20), arrowstyle='->',
                                                 mutation_scale=30, color='C1', alpha=0.5, linewidth=3)

            #self.orientation_gauge, self.ax_gauge, self.label_positions = gauge(arrow=6, title='', ax=ax)
            self.artists = [sensors, body, head, predicted_pos_artist, predicted_decision_artist]
            #self.artists = [sensors, body, head, predicted_pos_artist]

        else:
            self.artists = [sensors, body, head]

        ax.add_collection(sensors)
        ax.add_artist(body)
        ax.add_artist(head)
        if self.decoder:
            ax.add_artist(predicted_pos_artist)
            ax.add_artist(predicted_decision_artist)


    def compute_orientation(self):
        """ Calculates the orientation of the bot accoridng to the sensor values."""
        dv = (self.sensors["value"].ravel() * [-4, -3, -2, -1, 1, 2, 3, 4]).sum()
        #if abs(dv) > 0.01:  # if 75 sensor size
        self.orientation += 0.015 * dv
        if self.save_bot_sates:
            self.all_d_orientations.append(0.015 * dv.copy())

    def update_position(self, maze):
        """Updates the position of the bot according to the calculated orientation."""
        self.position += 2 * np.array([np.cos(self.orientation), np.sin(self.orientation)])

    def update(self, maze,  cues):
        """ Update the bot's position and orientation in the maze """
        #sensors, body, head, zone, decision, orientation = self.artists

        if self.decoder:
            #sensors, body, head, predicted_pos_artist = self.artists
            sensors, body, head, predicted_pos_artist, predicted_decision_artist = self.artists
        else:
            sensors, body, head = self.artists

        # Sensors
        verts = sensors.get_segments()
        linewidths = sensors.get_linewidth()

        # all angles of the sensors
        A = self.sensors["angle"] + self.orientation

        # cos and sin of the sensors
        T = np.stack([np.cos(A), np.sin(A)], axis=1)

        P1 = self.position + self.size * T
        P2 = P1 + self.sensors["range"] * T
        P3, P4 = maze.walls[:, 0], maze.walls[:, 1]

        for i, (p1, p2) in enumerate(zip(P1, P2)):
            verts[2 * i] = verts[2 * i + 1] = p1, p2
            linewidths[2 * i + 1] = 1
            C = line_intersect(p1, p2, P3, P4)
            index = np.argmin(np.sum((C - p1) ** 2, axis=1))
            p = C[index]
            if p[0] < np.inf:
                verts[2 * i + 1] = p1, p
                self.sensors["value"][i] = np.sqrt(np.sum((p1 - p) ** 2))
                self.sensors["value"][i] /= self.sensors["range"][i]
            else:
                self.sensors["value"][i] = 1

            # add noise
            self.sensors["value"][i] += np.random.normal(0, 0.03)
            if self.sensors["value"][i] > 1:
                self.sensors["value"][i] = 1

        sensors.set_verts(verts)
        sensors.set_linewidths(linewidths)

        # Update body
        body.set_center(self.position)

        if self.decoder:
            # Update place cell zone
            predicted_pos_artist.set_center(self.predicted_pos)
            # Update decision
            if self.decision_pred == 0:
                predicted_decision_artist.set_positions((270, 100), (270, 20))
            else:
                predicted_decision_artist.set_positions((270, 200), (270, 280))


            if 125 < self.position[1]< 175:
                predicted_decision_artist.set_color('C1')
            else:
                predicted_decision_artist.set_color('w')


            #pos = self.label_positions[self.orientation_pred]
            #self.orientation_gauge.set_positions((0, 0),
            #                                     (0.225 * np.cos(np.radians(pos)),
            #                                      0.225 * np.sin(np.radians(pos))))

        # Update head
        head_verts = np.array([self.position, self.position +
                               self.size * np.array([np.cos(self.orientation), np.sin(self.orientation)]).reshape(2, )])

        head.set_verts(np.expand_dims(head_verts, axis=0))

        if self.save_bot_sates:
            self.all_orientations.append(self.orientation.copy())
            self.all_sensors_vals.append(self.sensors['value'].ravel().copy())
            self.all_positions.append(self.position.copy())
            if cues:
                self.all_cues.append([int(self.right_cue), int(self.left_cue)])


    def update_cues(self, task):
        if task == 'R-L':
            if 125 <= self.position[1] <= 175:
                if self.left_cue_prev:
                    self.right_cue = True
                    self.left_cue = False
                elif self.right_cue_prev:
                    self.left_cue = True
                    self.right_cue = False
                cues = [self.right_cue, self.left_cue]
            elif self.position[1] < 125 or self.position[1] > 175:
                if self.left_cue:
                    self.left_cue_prev = True
                    self.right_cue_prev = False
                    self.left_cue = False
                elif self.right_cue:
                    self.right_cue_prev = True
                    self.left_cue_prev = False
                    self.right_cue = False
                cues = [self.right_cue, self.left_cue]
            else:
                cues = [0, 0]
        elif task == 'RR-LL':
            if 125 < self.position[1] < 175:
                if not self.enter_corridor:
                    if self.iter_right == 1 or self.iter_left == 2:
                        self.iter_right += 1
                        self.iter_left = 0
                        self.right_cue = True
                        self.left_cue = False
                    elif self.iter_left == 1 or self.iter_right == 2:
                        self.iter_left += 1
                        self.iter_right = 0
                        self.right_cue = False
                        self.left_cue = True
                self.enter_corridor = True
                if 0 < self.iter_right <= 2 and self.iter_left == 0:
                    self.right_cue = True
                    self.left_cue = False
                elif 0 < self.iter_left <= 2 and self.iter_right == 0:
                    self.right_cue = False
                    self.left_cue = True
                cues = [self.right_cue, self.left_cue]
            else:
                cues = [0, 0]
                self.enter_corridor = False
        return cues





