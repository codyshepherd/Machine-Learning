import time
import datetime
import pandas as pd
import ipaddress as ipad
import calendar

#enum substitute
#TODO: Expand to accept pcaps, netflow
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
        return str(self.time) + ' ' + self.uid + ' ' + self.src + ' ' +\
                self.dest + ' ' + str(self.port) + ' ' + str(self.dur) + ' ' +\
                str(self.obytes) + ' ' + str(self.rbytes)

#manager class for data layer
class Data:
    FILE_MODE = 'r'         #read-only
    BRO_COMMENT = '#'

    BRO_TIME = 0
    BRO_UID = 1
    BRO_SRC = 2
    BRO_DEST = 4
    BRO_PORT = 5
    BRO_DUR = 8
    BRO_OBYTES = 9
    BRO_RBYTES = 10

    def __init__(self, fname, typenum):
        self.type = typenum
        self.fname = fname
        #self.lines = self.get_lines()

    def __enter__(self):
        self.fhandle = open(self.fname, self.FILE_MODE)
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

            if line.startswith(self.BRO_COMMENT):
                continue

            else:
                spline = line.split()

                try:
                    conn = Conn_t()
                    conn.time = datetime.datetime.fromtimestamp(float(spline[self.BRO_TIME]))
                    conn.uid = spline[self.BRO_UID]
                    conn.src = unicode(spline[self.BRO_SRC])
                    conn.dest = unicode(spline[self.BRO_DEST])
                    conn.port = self.integerize(spline[self.BRO_PORT])
                    conn.dur = self.floatize(spline[self.BRO_DUR])
                    conn.obytes = self.integerize(spline[self.BRO_OBYTES])
                    conn.rbytes = self.integerize(spline[self.BRO_RBYTES])
                except ValueError:
                    print "ValueError thrown... Contents of spline in Data.get_lines():"
                    print spline
                    return

                self.series = pd.Series({'time': calendar.timegm(conn.time.timetuple()),\
                                         'src': int(ipad.ip_address(conn.src)),\
                                         'dest': int(ipad.ip_address(conn.dest)),\
                                         'port': int(conn.port),\
                                         'dur': conn.dur,\
                                         'obytes': conn.obytes,\
                                         'rbytes': conn.rbytes})

                yield conn

    #Exception-safe Conversion methods
    def integerize(self, line):
        try:
            num = int(line)
        except ValueError, TypeError:
            print "Exception converting line to int in Data.integerize"
            num = None
        return num

    def floatize(self,line):
        try:
            num = float(line)
        except ValueError, TypeError:
            print "Exception converting line to float in Data.floatize"
            num = None
        return num


"""
#Testing iterator functionality
with Data('conn.log',Doc_t.BRO) as d:
    for conn in d.get_lines(20):
        print conn
        print
"""
