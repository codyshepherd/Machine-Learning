import pandas as pd
import datetime
import calendar
import ipaddress as ipad

class Capsule:

    def __init__(self, content):
        self.content = ' '.join([str(content.time), content.uid, content.src, content.dest, content.uri, content.ref, str(content.cat)])
        self.series = pd.Series({'time': calendar.timegm(content.time.timetuple()), 'src': int(ipad.ip_address(content.src)), 'dest': int(ipad.ip_address(content.dest))})
        self.cat = int(content.cat)
        self.uid = content.uid

    def getQarray(self):
        return self.qr.get_matrix()

    def getImage(self):
        return self.qr.make_image()

class Capsule_c:

    def __init__(self, content):
        self.content = ' '.join([str(content.time), content.uid, content.src, content.dest, str(content.port), str(content.dur), str(content.obytes), str(content.rbytes)])
        self.series = pd.Series({'time': calendar.timegm(content.time.timetuple()), 'src': int(ipad.ip_address(content.src)), 'dest': int(ipad.ip_address(content.dest)), 'port': content.port, 'dur':content.dur , 'obytes':content.obytes, 'rbytes':content.rbytes})
        self.uid = content.uid