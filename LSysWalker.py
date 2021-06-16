#!/usr/bin/env python3
import math

# Used to aquire the list of coordinates that corresponds the the Moore curve with desired iterations
class Walker:
    def __init__(self, ax=None, rules=None, i=3, step_size=20, angle=90):
        if ax == None:
            self.axiom = 'LFL+F+LFL' # Moore Curve Axiom default
        else:
            self.axiom = ax

        if rules == None:
            self.rules = {'L': '-RF+LFL+FR-', 'R': '+LF-RFR-FL+'} # Moore Curve Rules default
        else:
            self.rules = rules

        self.pos = [0, 0]
        self.angle = angle * (math.pi / 180)
        self.coords = []
        self.xrange = [0, 0]
        self.yrange = [0, 0]

        self.iterations = i
        self.step_size = step_size

        # Width specified assumes Moore Curve
        self.width = (2 ** self.iterations)*self.step_size

        # Iterate to construct final LSys String
        self.result_string = self.axiom
        for _ in range(1, self.iterations):
            self.result_string = self.iterate(self.result_string)

        # Construct L System 
        self.render_coords()
        # Center coordinates around 0,0
        self.rectify()

        print(self)
    
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
        
        # Parse Result String
        for character in self.result_string:
            if character == 'F': # Step Forward
                # Update Position
                dx = int(math.cos(self.angle)*self.step_size)
                dy = int(math.sin(self.angle)*self.step_size)
                self.pos = [self.pos[0]+dx, self.pos[1]+dy]
                # Append Position to Coord list
                self.coords.append(self.pos)
                # Update Coordinate bounds
                if self.pos[0] < self.xrange[0]: self.xrange[0] = self.pos[0]
                if self.pos[0] > self.xrange[1]: self.xrange[1] = self.pos[0]
                if self.pos[1] < self.yrange[0]: self.yrange[0] = self.pos[1]
                if self.pos[1] > self.yrange[1]: self.yrange[1] = self.pos[1]
            elif character == '+': # Turn Right
                self.angle = self.angle - (90 * (math.pi / 180))
            elif character == '-': # Turn Left
                self.angle = self.angle + (90 * (math.pi / 180))
            else:
                pass  # Ignore other characters

    # Center Coordinates around 0,0
    def rectify(self):
        # Find proper offsets to center
        xdiff = (self.xrange[1]-self.xrange[0])//2
        xdiff = self.xrange[0] + xdiff
        ydiff = (self.yrange[1]-self.yrange[0])//2
        ydiff = self.yrange[0] + ydiff
        print("xdiff: ",xdiff,"  ydiff: ",ydiff)
        # Offset all coordinates
        for coord in self.coords:
            coord[0] = coord[0] - xdiff
            coord[1] = coord[1] - ydiff
        #self.width = (2 ** self.iterations)*self.step_size
        
