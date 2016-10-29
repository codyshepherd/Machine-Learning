import time
import datetime
import pandas as pd
import ipaddress as ipad
import calendar

#enum substitute
class Doc_t:
    BRO = range(1)

#basic object for holding conn.log fields
class Conn_t:
    def __init__(self):
        self.time = None
        self.uid  = None
        self.src  = None
        self.dest = None
        self.port = None
        self.dur = None
        self.obytes = None
        self.rbytes = None

        self.series = None

    def __str__(self):
        return self.time + ' ' + self.uid + ' ' + self.src + ' ' +\
                self.dest + ' ' + str(self.port) + ' ' + str(self.dur) + ' ' +\
                str(self.obytes) + ' ' + str(self.rbytes)

#manager class for data layer
class Data:
    def __init__(self, fname, typenum):
        self.type = typenum
        self.fname = fname
        #self.lines = self.get_lines()

    def __enter__(self):
        self.fhandle = open(self.fname, 'r')            #open file read-only
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.fhandle.close()

    def set_type(self, t=Doc_t.BRO):
        self.type = t

    def get_lines(self, num):
        for i in xrange(num):
            line = self.fhandle.readline()

            if not line:
                return

            if line.startswith('#'):
                continue

            else:
                spline = line.split()

                try:
                    conn = Conn_t()
                    conn.time = str(datetime.datetime.fromtimestamp(float(spline[0])))
                    conn.uid = spline[1]
                    conn.src = spline[2]
                    conn.dest = spline[4]
                    conn.port = self.integerize(spline[5])
                    conn.dur = self.floatize(spline[8])
                    conn.obytes = self.integerize(spline[9])
                    conn.rbytes = self.integerize(spline[10])
                except ValueError:
                    print "ValueError thrown... Contents of spline in Data.get_lines():"
                    print spline
                    return

                self.series = pd.Series({'time': calendar.timegm(self.time.timetuple()),\
                                         'src': int(ipad.ip_address(self.src)),\
                                         'dest': int(ipad.ip_address(self.dest)),\
                                         'port': int(self.port),\
                                         'dur': self.dur,\
                                         'obytes': self.obytes,\
                                         'rbytes': self.rbytes})

                yield conn

    #Conversion methods
    def integerize(self, line):
        try:
            num = int(line)
        except ValueError, TypeError:
            num = 0
        return num

    def floatize(self,line):
        try:
            num = float(line)
        except ValueError, TypeError:
            num = 0
        return num


"""
#Testing iterator functionality
with Data('conn.log',Doc_t.BRO) as d:
    for conn in d.get_lines(20):
        print conn
        print
"""
