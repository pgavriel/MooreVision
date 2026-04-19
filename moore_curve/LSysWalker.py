#!/usr/bin/env python3
import math

# Used to aquire the list of coordinates that corresponds the the Moore curve with desired iterations
class Walker:
    def __init__(self, mode=0, ax=None, rules=None, i=3, step_size=20, angle=90):
        self.modes = ["moore","zigzag","zigzag2","rxr"]
        self.mode = mode
        assert mode in range(len(self.modes)), f"LSYS ERROR: Mode should be integer index for defined modes {self.modes} (Got {mode})."

        self.iterations = i
        self.step_size = step_size
        if mode == 0: # MOORE CURVE L-SYS
            self.axiom = 'LFL+F+LFL' # Moore Curve Axiom default
            self.rules = {'L': '-RF+LFL+FR-', 'R': '+LF-RFR-FL+'} # Moore Curve Rules default
            self.width = (2 ** self.iterations)*self.step_size
            self.origin = [(2 ** self.iterations)-1,1]
        elif mode == 1: # ZIGZAG SQUARE L-SYS
            self.axiom = "RX"
            self.rules = {
                "R": "R",
                "L": "L",
                "U": "U",
                "X": "ULY",   # append an L row, then next time append an R row
                "Y": "URX"    # append an R row, then next time append an L row
            }
            self.width = (self.iterations)*self.step_size
            self.origin = [1,1]
        elif mode == 2: # ZIGZAG2  SQUARE L-SYS
            self.axiom = "RULX"
            self.rules = {
                "R": "R",
                "L": "L",
                "U": "U",
                "X": "URULX",   # append a new set of rows
            }
            self.width = (2*self.iterations)*self.step_size
            self.origin = [1,1]
        elif mode == 3: # ROW BY ROW SQUARE L-SYS
            self.axiom = "RX"
            self.rules = {
                "R": "R",
                "U": "U",
                "X": "URX",   # append a new R row
            }
            self.width = (self.iterations)*self.step_size
            self.origin = [1,1]

        # if ax == None:
        #     self.axiom = 'LFL+F+LFL' # Moore Curve Axiom default
        # else:
        #     self.axiom = ax

        # if rules == None:
        #     self.rules = {'L': '-RF+LFL+FR-', 'R': '+LF-RFR-FL+'} # Moore Curve Rules default
        # else:
        #     self.rules = rules

        self.pos = [0, 0]
        self.angle = angle * (math.pi / 180)
        self.coords = []
        self.xrange = [0, 0]
        self.yrange = [0, 0]

        # Width specified assumes Moore Curve
        # self.width = (2 ** self.iterations)*self.step_size
        # self.width = (self.iterations)*self.step_size

        # Iterate to construct final LSys String
        self.result_string = self.axiom
        for _ in range(1, self.iterations):
            self.result_string = self.iterate(self.result_string)

        # Construct L System
        self.render_coords()
        
        # print(self.coords)
        # Center coordinates around 0,0
        self.rectify()
        # print(self.xrange, self.yrange)
        # print(self.coords)
        print("[L-Sys Walker Initiated]")
        # print(self)

    def __str__(self):
        s = "LSysTurtle: Iterations: {}  StepSize: {}\n\t   Axiom: {}  Rules: {}".format(self.iterations,self.step_size,self.axiom,self.rules)
        # s += "\n\nResult: {}".format(self.result_string)
        s += "\nPoints: {}".format(len(self.coords))
        s += "\nXRange: {}  YRange: {}".format(self.xrange,self.yrange)
        return s

    def iterate(self, string):
        production = ''

        for character in string:
            if character in self.rules:
                production += self.rules[character]
            else:
                production += character  # just copy other characters
        # print("PRODUCE: ",production)
        return production

    def render_coords(self):
        # Reset Coords, Add Starting Point
        self.coords = []
        self.coords.append(self.pos)

        # print(f"[LSYS][i={self.iterations}] STR: {self.result_string}")
        # Parse Result String
        for character in self.result_string:
            if self.mode == 0: # MOORE CURVE RULES ===========================================================
                if character == 'F': # Step Forward
                    # Update Position
                    dx = int(math.cos(self.angle)*self.step_size)
                    dy = int(math.sin(self.angle)*self.step_size)
                    self.pos = [self.pos[0]+dx, self.pos[1]+dy]
                    # Append Position to Coord list
                    self.coords.append(self.pos)
                    # Update Coordinate bounds
                    # if self.pos[0] < self.xrange[0]: self.xrange[0] = self.pos[0]
                    # if self.pos[0] > self.xrange[1]: self.xrange[1] = self.pos[0]
                    # if self.pos[1] < self.yrange[0]: self.yrange[0] = self.pos[1]
                    # if self.pos[1] > self.yrange[1]: self.yrange[1] = self.pos[1]
                elif character == '+': # Turn Right
                    self.angle = self.angle - (90 * (math.pi / 180))
                elif character == '-': # Turn Left
                    self.angle = self.angle + (90 * (math.pi / 180))
                else:
                    pass  # Ignore other characters
            elif self.mode == 1: # ZIGZAG RULES ===========================================================
                if character == 'R':
                    for _ in range(self.iterations-1):
                        self.pos = [self.pos[0]+self.step_size, self.pos[1]]
                        self.coords.append(self.pos)
                        # print(f"Append {self.pos}")
                elif character == 'L':
                    for _ in range(self.iterations-1):
                        self.pos = [self.pos[0]-self.step_size, self.pos[1]]
                        self.coords.append(self.pos)
                        # print(f"Append {self.pos}")
                elif character == 'U':
                    self.pos = [self.pos[0], self.pos[1]+self.step_size]
                    self.coords.append(self.pos)
                    # print(f"Append {self.pos}")
                else:
                    pass  # Ignore other characters
            elif self.mode == 2: # ZIGZAG2 RULES ===========================================================
                if character == 'R':
                    for _ in range((self.iterations*2)-1):
                        self.pos = [self.pos[0]+self.step_size, self.pos[1]]
                        self.coords.append(self.pos)
                        # print(f"Append {self.pos}")
                elif character == 'L':
                    for _ in range((self.iterations*2)-1):
                        self.pos = [self.pos[0]-self.step_size, self.pos[1]]
                        self.coords.append(self.pos)
                        # print(f"Append {self.pos}")
                elif character == 'U':
                    self.pos = [self.pos[0], self.pos[1]+self.step_size]
                    self.coords.append(self.pos)
                    # print(f"Append {self.pos}")
                else:
                    pass  # Ignore other characters
            elif self.mode == 3: # ROW BY ROW RULES ===========================================================
                if character == 'R':
                    for _ in range(self.iterations-1):
                        self.pos = [self.pos[0]+self.step_size, self.pos[1]]
                        self.coords.append(self.pos)
                elif character == 'U':
                    self.pos = [0, self.pos[1]+self.step_size]
                    self.coords.append(self.pos)
                else:
                    pass  # Ignore other characters
            else:
                raise RuntimeError("Unknown mode used for rules")
            
            # Update Coordinate bounds after each character
            if self.pos[0] < self.xrange[0]: self.xrange[0] = self.pos[0]
            if self.pos[0] > self.xrange[1]: self.xrange[1] = self.pos[0]
            if self.pos[1] < self.yrange[0]: self.yrange[0] = self.pos[1]
            if self.pos[1] > self.yrange[1]: self.yrange[1] = self.pos[1]

    # Center Coordinates around 0,0
    def rectify(self):
        # Find proper offsets to center
        xdiff = (self.xrange[1]-self.xrange[0])//2
        xdiff = self.xrange[0] + xdiff
        ydiff = (self.yrange[1]-self.yrange[0])//2
        ydiff = self.yrange[0] + ydiff
        # print("xdiff: ",xdiff,"  ydiff: ",ydiff)
        # Offset all coordinates
        for coord in self.coords:
            coord[0] = coord[0] - xdiff
            coord[1] = coord[1] - ydiff
        #self.width = (2 ** self.iterations)*self.step_size
