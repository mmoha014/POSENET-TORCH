class Hands:
    def __init__(self):
        self.index=0
        self.points = []
        self.wrist=[]
        self.angle = []
        self.max_c =0
        self.min_c=1000
        self.index_max = 0
        self.index_min = 0
        self.peaks = []
        self.romUpDwn = []
        self.romDwnUp = []
        self.top = 0
        self.down = 0
        self.reps = 0
        self.half_rep_count = 0
        self.VUpDwn = [] #Velocity UpDwn
        self.VDwnUp = [] #Velocity DwnUp
        self.DDwnUp = [] #Duration DwnUp
        self.DUpDwn = [] # Duration UpDwn
        self.ts = 0
        self.time_recorder = []

RHand = Hands()
LHand = Hands()