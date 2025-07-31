import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.ticker import MultipleLocator
import matplotlib.patches as patches

def darken_color(color, factor=0.7):
    import matplotlib.colors as mcolors
    rgb = mcolors.to_rgb(color)  # Convert color to RGB
    return tuple([factor * x for x in rgb])

def line_intersect(p1, p2, P3, P4):
    """ Calculates intersection point of two segments. segment 1: [p1,p2], segment 2: [P3,P4].
    Inputs:
    p1, p2: arrays containing the coordinates of the points of first segments.
    P3, P4: arrays containing the coordinates of the points of second segments
    Outputs: list of X,Y coordinates of the intersection points. Set to np.inf if no intersection.
    """
    p1 = np.atleast_2d(p1)
    p2 = np.atleast_2d(p2)
    P3 = np.atleast_2d(P3)
    P4 = np.atleast_2d(P4)

    x1, y1 = p1[:, 0], p1[:, 1]
    x2, y2 = p2[:, 0], p2[:, 1]
    X3, Y3 = P3[:, 0], P3[:, 1]
    X4, Y4 = P4[:, 0], P4[:, 1]

    D = (Y4 - Y3) * (x2 - x1) - (X4 - X3) * (y2 - y1)

    # Colinearity test
    C = (D != 0)

    # Calculate the distance to the intersection point
    UA = ((X4 - X3) * (y1 - Y3) - (Y4 - Y3) * (x1 - X3))
    UA = np.divide(UA, D, where=C)
    UB = ((x2 - x1) * (y1 - Y3) - (y2 - y1) * (x1 - X3))
    UB = np.divide(UB, D, where=C)

    # Test if intersections are inside each segment
    C = C * (UA > 0) * (UA < 1) * (UB > 0) * (UB < 1)

    # intersection of the point of the two lines
    X = np.where(C, x1 + UA * (x2 - x1), np.inf)
    Y = np.where(C, y1 + UA * (y2 - y1), np.inf)
    return np.stack([X, Y], axis=1)


class Maze:
    """
    A simple 8-maze made of straight walls (line segments)
    """

    def __init__(self, simulation_mode="esn"):

        self.walls = np.array([
            # Surrounding walls
            [(0, 100), (200, 0)],
            [(200, 0), (300, 0)],
            [(300, 0), (300, 300)],
            [(300, 300), (200, 300)],
            [(200, 300), (0, 200)],
            [(0, 200), (0, 100)],

            # Bottom hole
            [(50, 125), (250, 125)],
            [(250, 125), (250, 50)],
            [(250, 50), (50, 125)],

            # Top hole
            [(50, 175), (250, 175)],
            [(250, 175), (250, 250)],
            [(250, 250), (50, 175)],

            # Moving walls (invisibles) to constraining bot path
            [(0, 150), (60, 175)],
             [(250, 125), (300, 150)]
        ])

        self.walls_asym = np.array([
            # Surrounding walls
            [(0, 150), (200, 0)],
            [(200, 0), (300, 0)],
            [(300, 0), (300, 225)],
            [(300, 225), (250, 225)],
            [(250, 225), (250, 400)],
            [(250, 400), (200, 400)],
            [(200, 400), (0, 250)],
            [(0, 250), (0, 150)],

            # Bottom hole
            [(50, 175), (250, 175)],
            [(250, 175), (250, 50)],
            [(250, 50), (50, 175)],

            # Top hole
            [(50, 225), (250, 225)],
            [(250, 225), (250, 350)],
            [(250, 350), (50, 225)],

            # Moving walls (invisibles) to constraining bot path
            [(0, 200), (60, 225)],
            [(250, 225), (300, 200)]
        ])

        if simulation_mode == "walls":
            self.invisible_walls = True
        else:
            self.invisible_walls = False
            self.walls[12:] = [[(0, 0), (0, 0)],
                               [(0, 0), (0, 0)]]

        self.alternate = None
        self.iter = 0
        self.in_corridor = False

    def draw(self, ax, grid=True, margin=5):
        """
        Render the maze
        """
        # Buidling a filled patch from walls
        V, C, S = [], [], self.walls
        V.extend(S[0 + i, 0] for i in [0, 1, 2, 3, 4, 5, 0])
        V.extend(S[6 + i, 0] for i in [0, 1, 2, 0])
        V.extend(S[9 + i, 0] for i in [0, 2, 1, 0])

        C = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY,
            Path.MOVETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY,
            Path.MOVETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]

        path = Path(V, C)
        patch = PathPatch(path, clip_on=False, linewidth=1.5,
                          edgecolor="black", facecolor="white", alpha=1)

        #C_2 = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
        #V_2 = []
        #V_2.extend(S[9 + i, 0] for i in [0, 1, 2, 0])
        #path_2 = Path(V_2, C_2)
        #patch_2 = PathPatch(path_2, clip_on=False, linewidth=1.5,
                           # edgecolor="black", facecolor="white", alpha=1)
        #ax.add_artist(patch_2)





        # Set figure limits, grid and ticks
        ax.set_axisbelow(True)
        ax.add_artist(patch)
        #ax.set_xlim(0 - margin, 300 + margin)
        #ax.set_ylim(0 - margin, 400 + margin)

        ax.set_xlim(0 - margin, 300 + margin)
        ax.set_ylim(0 - margin, 300 + margin)
        if grid:
            ax.xaxis.set_major_locator(MultipleLocator(100))
            ax.xaxis.set_minor_locator(MultipleLocator(10))
            ax.yaxis.set_major_locator(MultipleLocator(100))
            ax.yaxis.set_minor_locator(MultipleLocator(10))

            ax.xaxis.set_major_locator(MultipleLocator(50))
            ax.yaxis.set_major_locator(MultipleLocator(50))
            ax.grid(True, "major", color="0.75", linewidth=1.00, clip_on=False)
            ax.grid(True, "minor", color="0.75", linewidth=0.50, clip_on=False)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(axis='both', which='major', size=0)
        ax.tick_params(axis='both', which='minor', size=0)


    def draw_static(self, ax, margin=5):
        """
        Render the maze
        """

        # Building a filled patch from walls
        V, C, S = [], [], self.walls
        V.extend(S[0 + i, 0] for i in [0, 1, 2, 3, 0])
        V.extend(S[4 + i, 0] for i in [0, 1, 2, 3, 0])
        V.extend(S[8 + i, 0] for i in [0, 1, 2, 3, 0])
        C = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY] * 3
        path = Path(V, C)
        patch = PathPatch(path, clip_on=False, linewidth=1.5,
                          edgecolor="black", facecolor="white")

        # Set figure limits, grid and ticks
        ax.set_axisbelow(True)
        ax.add_artist(patch)
        ax.set_xlim(0 - margin, 300 + margin)
        ax.set_ylim(0 - margin, 500 + margin)

        # Add thick red dotted line at y=250 from x=100 to x=200
        ax.plot([50, 200], [245, 245], color=darken_color('C3', 0.7), linestyle='dotted', linewidth=4)
        ax.plot([50, 200], [255, 255], color=darken_color('C0', 0.7), linestyle='dotted', linewidth=4)

        ax.plot([50, 50], [50, 250], color=darken_color('C3', 0.7), linestyle='dotted', linewidth=4)

        ax.plot([50, 50], [250, 440], color=darken_color('C0', 0.7), linestyle='dotted', linewidth=4)

        # Add markers at (100, 250), (150, 250), (200, 250)
        #marker_coords = [(100, 250), (150, 250), (200, 250)]
        #ax.scatter(*zip(*marker_coords), color='black', s=80, edgecolor='black', zorder=5, marker='x')

        # Add labels C1, C2, C3
        #labels = ['C3', 'C2', 'C1']
        #for i, (x, y) in enumerate(marker_coords):
         #   ax.text(x, y + 10, labels[i], ha='center', fontsize=20, color='black')

        # Uncomment the following block to add a grey zone (if needed)
        grey_zone = patches.Rectangle((100, 200), 100, 100, linewidth=0, edgecolor='none', facecolor='grey',
                                       alpha=0.5, label='Recorded zone')
        ax.add_patch(grey_zone)

        ax.legend(bbox_to_anchor=(1, -0.05))

        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(axis='both', which='major', size=0)
        ax.tick_params(axis='both', which='minor', size=0)

    def update_walls(self, bot_position):
        """ Add the invisible walls to force the bot alternating right and left direction."""
        if bot_position[1] < 100:
            self.walls[12:] = [[(0, 150), (60, 175)],
                               [(250, 125), (300, 150)]]
        elif bot_position[1] > 200:
            self.walls[12:] = [[(0, 150), (60, 125)],
                               [(250, 175), (300, 150)]]
        else:
            pass


    def update_walls_RR_LL(self, bot_position):
        """ Add the invisible walls to force the bot alternating right and left direction overy other time."""
        if 125 < bot_position[1] < 175:
            if not self.in_corridor:
                if self.iter == 1:
                    self.iter = 0
                else:
                    self.iter += 1
            self.in_corridor = True
        else:
            self.in_corridor = False

        if bot_position[1] < 100 and self.iter < 1:
            self.walls[12:] = [[(0, 150), (60, 175)],
                               [(250, 175), (300, 150)]]

        elif bot_position[1] < 100 and self.iter == 1:
            self.walls[12:] = [[(0, 150), (60, 175)],
                               [(250, 125), (300, 150)]]

        elif bot_position[1] > 200 and self.iter < 1:
            self.walls[12:] = [[(0, 150), (60, 125)],
                               [(250, 125), (300, 150)]]

        elif bot_position[1] > 200 and self.iter == 1:
            self.walls[12:] = [[(0, 150), (60, 125)],
                               [(250, 175), (300, 150)]]
        else:
            pass

