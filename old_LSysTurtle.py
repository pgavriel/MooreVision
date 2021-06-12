from turtle import Screen, Turtle

# DEPRECATED, figured out it was super easy to implement this
# Used to aquire the list of coordinates that corresponds the the Moore curve with desired iterations
class LSysTurtle:
    def __init__(self, ax=None, rules=None, i=3, step_size=10, show=True):
        if ax == None:
            self.axiom = 'LFL+F+LFL' # Moore Curve Axiom default
        else:
            self.axiom = ax

        if rules == None:
            self.rules = {'L': '-RF+LFL+FR-', 'R': '+LF-RFR-FL+'} # Moore Curve Rules default
        else:
            self.rules = rules

        self.iterations = i
        self.step_size = step_size
        self.coords = []
        self.xrange = [0, 0]
        self.yrange = [0, 0]

        self.result_string = self.axiom
        for _ in range(1, self.iterations):
            self.result_string = self.produce(self.result_string)

        self.show = show
        # Construct L System 
        self.draw()
        # Center coordinates around 0,0
        self.rectify()

        print(self)
    
    def __str__(self):
        s = "LSysTurtle: Iterations: {}  StepSize: {}\n\t   Axiom: {}  Rules: {}".format(self.iterations,self.step_size,self.axiom,self.rules)
        # s += "\n\nResult: {}".format(self.result_string)
        s += "\nPoints: {}".format(len(self.coords))
        s += "\nXRange: {}  YRange: {}".format(self.xrange,self.yrange)
        return s 

    def produce(self, string):
        production = ''

        for character in string:
            if character in self.rules:
                production += self.rules[character]
            else:
                production += character  # just copy other characters
        # print("PRODUCE: ",production)
        return production

    def draw(self):
        #if self.show:
        screen = Screen()
        screen.tracer(False)

        turtle = Turtle()
        turtle.hideturtle()
        turtle.setheading(90)

        self.coords = []
        pos = turtle.pos()
        self.xrange = [int(pos[0]), int(pos[0])]
        self.yrange = [int(pos[1]), int(pos[1])]
        coord = [ int(pos[0]), int(pos[1])]
        self.coords.append(coord)
        for character in self.result_string:
            if character == 'F':
                turtle.forward(self.step_size)
                pos = turtle.pos()
                x, y = int(pos[0]), int(pos[1])
                if x < self.xrange[0]: self.xrange[0] = x
                if x > self.xrange[1]: self.xrange[1] = x
                if y < self.yrange[0]: self.yrange[0] = y
                if y > self.yrange[1]: self.yrange[1] = y
                coord = [x, y]
                self.coords.append(coord)
            elif character == '+':
                turtle.right(90)
            elif character == '-':
                turtle.left(90)
            else:
                pass  # ignore other characters
            
        if self.show:
            screen.tracer(True)
            screen.exitonclick()

    def rectify(self):
        xdiff = (self.xrange[1]-self.xrange[0])//2
        xdiff = self.xrange[0] + xdiff
        ydiff = (self.yrange[1]-self.yrange[0])//2
        ydiff = self.yrange[0] + ydiff
        print("xdiff: ",xdiff,"  ydiff: ",ydiff)
        for coord in self.coords:
            coord[0] = coord[0] - xdiff
            coord[1] = coord[1] - ydiff
        self.width = (2 ** self.iterations)*self.step_size
        
