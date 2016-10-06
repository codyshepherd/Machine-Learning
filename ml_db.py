import peewee
##########  DB INITIALIZATION  ##############
db = peewee.MySQLDatabase('db_name', user='uname', passwd='my_pw')

class BaseModel(peewee.Model):
    class Meta:
        database = db

class T_Http(BaseModel):
    time = peewee.DateTimeField()
    uid = peewee.TextField()
    src = peewee.CharField()
    dest = peewee.CharField()
    uri = peewee.TextField()
    ref = peewee.TextField()
    cat = peewee.IntegerField()

class T_Conn(BaseModel):
    time = peewee.DateTimeField()
    uid = peewee.TextField()
    src = peewee.CharField()
    dest = peewee.CharField()
    port = peewee.IntegerField()
    #svc = peewee.CharField()
    dur = peewee.FloatField()
    obytes = peewee.IntegerField()
    rbytes = peewee.IntegerField()   