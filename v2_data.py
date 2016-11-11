import time
import datetime
import pandas as pd
import ipaddress as ipad
import calendar

#TODO: May not need Conn object, depending on how much 
# info we decide we want to store for a given sample


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

    def __init__(self, fname, typenum, start):
        self.type = typenum
        self.fname = fname
        self.series = []
        self.start = start

    def __enter__(self):
        self.fhandle = open(self.fname, self.FILE_MODE)
        self.fhandle.seek(self.start)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.fhandle.close()

    def set_type(self, t=Doc_t.BRO):
        self.type = t

    def tell(self):
        return self.fhandle.tell()

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
                    self.series = pd.Series({'atime': calendar.timegm(datetime.datetime.fromtimestamp(float(spline[self.BRO_TIME])).timetuple()),\
                                             'bsrc': self.integerize(self.ip(unicode(spline[self.BRO_SRC]))),\
                                             'cdest': self.integerize(self.ip(unicode(spline[self.BRO_DEST]))),\
                                             'dport': self.integerize(spline[self.BRO_PORT]),\
                                             'edur': self.floatize(spline[self.BRO_DUR]),\
                                             'fobytes': self.integerize(spline[self.BRO_OBYTES]),\
                                             'grbytes': self.integerize(spline[self.BRO_RBYTES])})
                    """
                    conn = Conn_t()
                    conn.time = datetime.datetime.fromtimestamp(float(spline[self.BRO_TIME]))
                    conn.uid = spline[self.BRO_UID]
                    conn.src = unicode(spline[self.BRO_SRC])
                    conn.dest = unicode(spline[self.BRO_DEST])
                    conn.port = self.integerize(spline[self.BRO_PORT])
                    conn.dur = self.floatize(spline[self.BRO_DUR])
                    conn.obytes = self.integerize(spline[self.BRO_OBYTES])
                    conn.rbytes = self.integerize(spline[self.BRO_RBYTES])
                    """
                except ValueError:
                    print "ValueError thrown... Contents of spline in Data.get_lines():"
                    print spline
                    return


                yield self.series

    #Exception-safe Conversion methods
    def integerize(self, line):
        try:
            num = int(line)
        except ValueError, TypeError:
            #print "Exception converting line to int in Data.integerize"
            #print line
            num = 0
        return num

    def floatize(self,line):
        try:
            num = float(line)
        except ValueError, TypeError:
            #print "Exception converting line to float in Data.floatize"
            #print line
            num = 0.0
        return num

    def ip(self, line):
        try:
            num = ipad.ip_address(line)
        except ValueError:
            #print "Exception converting line to ipaddress in Data.ip"
            #print line
            num = 0
        return num

"""
#Testing iterator functionality
with Data('conn.log',Doc_t.BRO) as d:
    for series in d.get_lines(20):
        print series.keys()
        for item in series.values:
            print item
        print
"""
