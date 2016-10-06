################################################
#                  PACKAGES                    #
################################################
#GUI
import Tkinter as tk

#Datatype / formatting
import time
import datetime

#Math and Machine Learning
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

#Utility
from collections import Counter
import pickle
import os.path

#Custom
from ml_db import *
import net

#import seaborn as sns
#sns.set(style='whitegrid', color_codes=True)
#from sklearn.preprocessing import RobustScaler

#################################################
#              CONTROL FLAGS                    #
#################################################

running = True

#################################################
#         GUI WINDOW AND FRAMES                 #
#################################################

root = tk.Tk()
root.wm_title("Network ML Framework")
root.rowconfigure(3,weight=1)
root.columnconfigure(0,weight=1)
root.columnconfigure(1,minsize=300)

topframe = tk.Frame(root)
topframe.grid(row=0, column=0, sticky='nsew')

midframe = tk.Frame(root)
midframe.grid(row=1,column=0, sticky='nsew')

midframe_nn = tk.Frame(root)
midframe_nn.grid(row=2, column=0, sticky='nsew')

midframe_dbscan = tk.Frame(root)
midframe_dbscan.grid(row=2, column=0, sticky='nsew')

midframe_kmeans = tk.Frame(root)
midframe_kmeans.grid(row=2, column=0, sticky='nsew')

bl_frame = tk.Frame(root)
bl_frame.grid(row=3, column=0, sticky='nsew')
bl_frame.columnconfigure(0, weight=1)
bl_frame.rowconfigure(0, weight=1)
bl_frame.rowconfigure(1, weight=1)
bl_frame.columnconfigure(0, weight=1)

br_frame = tk.Frame(root, width=450)
br_frame.grid(row=3, column=1, sticky='nsew')


##############################################################
#                      WINDOW ELEMENTS                       #
##############################################################

#############  Database Settings  #########
#title
sql_titletxt = tk.StringVar()
sql_titletxt.set("Database Options")
sql_titlelab = tk.Label(topframe, textvariable=sql_titletxt, font='Helvetica 16 bold',  pady=5, padx=20)

#bro script type radio and label
sql_labtxt0 = tk.StringVar()
sql_labtxt0.set('Bro log:')
sql_lab0 = tk.Label(topframe, textvariable=sql_labtxt0, pady=5)

u = tk.IntVar()
u.set(0)
sql_rad0 = tk.Radiobutton(topframe, text='http.log', variable=u, value=0)
sql_rad1 = tk.Radiobutton(topframe, text='conn.log', variable=u, value=1)
sql_rad1.select()

#remove old DB radio buttons and label
sql_labtxt1 = tk.StringVar()
sql_labtxt1.set('Clear old database?')
sql_lab1 = tk.Label(topframe, textvariable=sql_labtxt1, pady=5)

d = tk.IntVar()
d.set(0)
sql_rad2 = tk.Radiobutton(topframe, text="Yes", variable=d, value=0)
sql_rad3 = tk.Radiobutton(topframe, text="No", variable=d, value=1)
sql_rad2.select()

#DB fill buttons
sql_labtxt2 = tk.StringVar()
sql_labtxt2.set('Fill Database')
sql_lab2 = tk.Label(topframe, textvariable=sql_labtxt2, pady=5)
sql_labtxt3 = tk.StringVar()
sql_labtxt3.set('Number of items processed: ')
sql_lab3 = tk.Label(topframe, textvariable=sql_labtxt3, pady=5)
sql_labtxt4 = tk.IntVar()
sql_labtxt4.set(0)
sql_lab4 = tk.Label(topframe, textvariable=sql_labtxt4, pady=5)
sql_b1 = tk.Button(topframe, text="Start", command = lambda: sql_start())

#################  Machine Learning Options
#title
mlopt_labtxt0 = tk.StringVar()
mlopt_labtxt0.set('Machine Learning Options')
mlopt_lab0 = tk.Label(midframe, textvariable=mlopt_labtxt0, font='Helvetica 16 bold', pady=5)

#ML Model radio buttons
m = tk.IntVar()
mlopt_rad0 = tk.Radiobutton(midframe, text="Neural Net", variable=m, value=0, command=lambda: (m.set(0),midframe_nn.tkraise(),nn_b0.config(state=tk.NORMAL), nn_b1.config(state=tk.NORMAL), db_b0.config(state=tk.DISABLED), kmeans_b0.config(state=tk.DISABLED)))
mlopt_rad1 = tk.Radiobutton(midframe, text="DBSCAN Cluster Analysis", variable=m, value=1, command=lambda: (m.set(1),midframe_dbscan.tkraise(), nn_b0.config(state=tk.DISABLED), nn_b1.config(state=tk.DISABLED), db_b0.config(state=tk.NORMAL), kmeans_b0.config(state=tk.DISABLED)))
mlopt_rad2 = tk.Radiobutton(midframe, text="K-Means Cluster Analysis", variable=m, value=2, command=lambda: (m.set(2),midframe_kmeans.tkraise(), nn_b0.config(state=tk.DISABLED), nn_b1.config(state=tk.DISABLED), db_b0.config(state=tk.DISABLED), kmeans_b0.config(state=tk.NORMAL)))
m.set(2)
mlopt_rad2.select()

############ Beaconing Neural Net (midframe_nn)

#indicator for presence of trained model
nn_labtxt0 = tk.StringVar()
nn_lab0 = tk.Label(midframe_nn, textvariable=nn_labtxt0, pady=5)
if os.path.isfile('trained'):
    nn_lab0.config(fg='red')
    nn_labtxt0.set('TRAINED NN MODEL PRESENT')
else:
    nn_labtxt0.set('No Trained Model')

#Radios for loading Trained NN model
n = tk.IntVar()
n.set(1)
nn_rad0 = tk.Radiobutton(midframe_nn, text="Train New", variable=n, value=0)
nn_rad1 = tk.Radiobutton(midframe_nn, text="Load Model", variable=n, value=1)

#Train and predict buttons
nn_b0 = tk.Button(midframe_nn, text="Train", command = lambda: nnTrain(u), state=tk.DISABLED)
nn_b1 = tk.Button(midframe_nn, text="Predict", command=lambda: nn_predict(), state=tk.DISABLED)

#Labels for displaying training/prediction process
nn_labtxt1 = tk.StringVar()
nn_labtxt1.set('Number of training samples: ')
nn_lab1 = tk.Label(midframe_nn, textvariable=nn_labtxt1, pady=5)
nn_labtxt2 = tk.IntVar()
nn_labtxt2.set(0)
nn_lab2 = tk.Label(midframe_nn, textvariable=nn_labtxt2, pady=5)

nn_labtxt3 = tk.StringVar()
nn_labtxt3.set('Number of prediction samples processed: ')
nn_lab3 = tk.Label(midframe_nn, textvariable=nn_labtxt3, pady=5)
nn_labtxt4 = tk.IntVar()
nn_labtxt4.set(0)
nn_lab4 = tk.Label(midframe_nn, textvariable=nn_labtxt4, pady=5)

############## DBSCAN Cluster Analysis (midframe_dbscan)

#Radios for loading Cluster Data
o = tk.IntVar()
o.set(1)
db_rad0 = tk.Radiobutton(midframe_dbscan, text="New Cluster", variable=o, value=0)
db_rad1 = tk.Radiobutton(midframe_dbscan, text="Load Cluster", variable=o, value=1)

#indicator for presence of pre-pickled DBSCAN data
db_labtxt0 = tk.StringVar()
db_lab0 = tk.Label(midframe_dbscan, textvariable=db_labtxt0, pady=5)
if os.path.isfile('cluster'):
    db_lab0.config(fg='red')
    db_labtxt0.set('CLUSTER DATA PRESENT')
    o.set(1)
else:
    db_labtxt0.set('No Cluster Data')
    o.set(0)

#Buttons for execution
db_b0 = tk.Button(midframe_dbscan, text="Compute", command = lambda: db_scan())
db_b1 = tk.Button(midframe_dbscan, text="Plot", command = lambda: plot_options(), state=tk.DISABLED)

#labels for indicating progress
db_labtxt1 = tk.StringVar()
db_labtxt1.set('Number of database entries processed: ')
db_lab1 = tk.Label(midframe_dbscan, textvariable=db_labtxt1, pady=5)
db_labtxt2 = tk.IntVar()
db_labtxt2.set(0)
db_lab2 = tk.Label(midframe_dbscan, textvariable=db_labtxt2, pady=5)

##################  K-Means Cluster Analysis (midframe_kmeans)

#radios for loading model data
p = tk.IntVar()
p.set(0)
kmeans_rad0 = tk.Radiobutton(midframe_kmeans, text="New Cluster", variable=p, value=0)
kmeans_rad1 = tk.Radiobutton(midframe_kmeans, text="Load Cluster", variable=p, value=1)

#user input for number of clusters
kmeans_sv0 = tk.IntVar()
kmeans_sv0.set(20)
kmeans_lab0 = tk.Label(midframe_kmeans, text='Number of clusters: ', pady=5)
kmeans_ent0 = tk.Entry(midframe_kmeans, textvariable=kmeans_sv0)

#indicator for pre-pickled model
kmeans_sv3 = tk.StringVar()
kmeans_lab1 = tk.Label(midframe_kmeans, textvariable=kmeans_sv3, pady=5)
if os.path.isfile('kmeans'):
    kmeans_lab1.config(fg='red')
    kmeans_sv3.set('CLUSTER DATA PRESENT')
    p.set(1)
else:
    kmeans_sv3.set('No Cluster Data')
    p.set(0)

#Buttons for execution
kmeans_b0 = tk.Button(midframe_kmeans, text="Compute", command=lambda:kmeans_start())
kmeans_b1 = tk.Button(midframe_kmeans, text="Plot", command=lambda:kmeans_plot(), state=tk.DISABLED)

#labels for indicating progress
kmeans_sv1 = tk.StringVar()
kmeans_sv1.set('Number of database entries processed: ')
kmeans_lab2 = tk.Label(midframe_kmeans, textvariable=kmeans_sv1, pady=5)
kmeans_sv2 = tk.IntVar()
kmeans_sv2.set(0)
kmeans_lab3 = tk.Label(midframe_kmeans, textvariable=kmeans_sv2, pady=5)

#################  br_frame
#session contents display (for Beaconing Neural Net)
br_lab0 = tk.Label(br_frame, text='Selected Session:', pady=5)
br_labtxt0 = tk.StringVar()
br_labtxt0.set('No session selected.')
br_message0 = tk.Message(br_frame, textvariable=br_labtxt0, width=300, pady=5, padx=5, bg='white')

#Confirm/Reject buttons (TODO: NO FUNCTIONALITY CURRENTLY)
br_b0 = tk.Button(br_frame, text="Confirm", command= lambda: 0)
br_b1 = tk.Button(br_frame, text="Reject", command= lambda: 0)

################  bl_frame  
#upper list box (console output, status updates)
lb = tk.Frame(bl_frame)
scroll1 = tk.Scrollbar(lb)
listbox = tk.Listbox(lb, yscrollcommand=scroll1.set)
scroll1.config(command=listbox.yview)
lb.grid(row=0, column=0, sticky='nsew')

#lower list box (Beaconing NN data points)
vb = tk.Frame(bl_frame)
scroll2 = tk.Scrollbar(vb)
varbox = tk.Listbox(vb, yscrollcommand=scroll2.set)
scroll2.config(command=varbox.yview)
vb.grid(row=1, column=0, sticky='nsew')

##########################################################
#                     CALLBACKS                          #
##########################################################

#database reset
def sql_reset():
    if u.get() == 0:
        print "attempting to create T_Http table"
        if T_Http.table_exists():
            print "deleting old http table"
            T_Http.drop_table()
        T_Http.create_table()

    elif u.get() == 1:
        print "Attempting to create T_Conn table"
        if T_Conn.table_exists():
            print "deleting old conn table"
            T_Conn.drop_table()
        T_Conn.create_table()

#handle click of DB_START button
def sql_start():
    if d.get() == 0:
        sql_reset()
    ug = u.get()
    global fhandle 
    fhandle = sql_setup(u)
    global running
    running = True
    sql_loop(ug)
   
#prepare socket for reading in
def sql_setup(u):
    #print "starting"
    listbox.insert(tk.END, 'Starting...')

    if u.get() == 0:
        #print "http log chosen"
        f_handle = open('http.log', 'r')
    elif u.get() == 1:
        f_handle = open('conn.log', 'r')
    
        return f_handle

#read line from socket, write to DB, write to UI
def sql_loop(u):
    global fhandle
    global running
    if (not running):
        listbox.insert(tk.END, 'DB fill done')
        #print 'done'
        return

    #actions for reading from http.log
    if u == 0:
        line = fhandle.readline()
    
        if not line: 
            listbox.insert(tk.END, 'End of File')
            running = False
            return

        if line.startswith('#'):
            pass
            #root.after(1, lambda: sql_loop(u))

        else:
            spline = line.split()

            if len(spline) > 11:
                try:
                    http = {'time':str(datetime.datetime.fromtimestamp(float(spline[0]))), 'uid': spline[1], 'src': spline[2], 'dest': spline[4], 'uri': spline[9], 'ref': spline[10], 'cat': spline[-1]}
                except ValueError:
                    print "ValueError thrown... Contents of spline in sql_loop(u):"
                    print spline
                    return

                val = sql_labtxt4.get()
                sql_labtxt4.set(val+1)
                iq = peewee.InsertQuery(T_Http, http)
                iq.execute()

    elif u == 1:
        line = fhandle.readline()

        if not line:
            listbox.insert(tk.END, 'End of File')
            running = False
            return

        if line.startswith('#'):
            pass

        else:
            spline = line.split()

            try:
                conn = {'time':str(datetime.datetime.fromtimestamp(float(spline[0]))), 'uid': spline[1], 'src': spline[2], 'dest': spline[4], 'port':integerize(spline[5]), 'dur':floatize(spline[8]), 'obytes':integerize(spline[9]), 'rbytes':integerize(spline[10])}
            except ValueError:
                print "ValueError thrown... Contents of spline in sql_loop(u):"
                print spline
                return

            val = sql_labtxt4.get()
            sql_labtxt4.set(val+1)
            iq = peewee.InsertQuery(T_Conn, conn)
            iq.execute()

    root.after(1, lambda: sql_loop(u))

#Conversion methods
def integerize(line):
    try:
        num = int(line)
    except ValueError, TypeError:
        num = 0
    return num

def floatize(line):
    try:
        num = float(line)
    except ValueError, TypeError:
        num = 0
    return num

#Training method for Neural Net
def nnTrain(u):

    global clf
    clf = MLPClassifier(activation='logistic', algorithm='l-bfgs', alpha=1e-5, hidden_layer_sizes=(5, 3), random_state=1)
    #clf = MLPClassifier(activation='relu', algorithm='sgd', alpha=1e-5, hidden_layer_sizes=(5, 3), random_state=1)

    global pdata
    pdata = []
    global data
    data = T_Http.select()

    if not data: 
        listbox.insert(tk.END, 'nnTrain done')
        return

    global lst
    lst = []
    global cats
    cats = []
    num = 0

    nn_trainLoop(0, 20000)

#Training / updating loop used by nnTrain()
def nn_trainLoop(index, length):
    global lst
    global cats
    global clf

    #load scaled data from pickled file and train model
    if n.get() == 1:
        with open('trained', 'r') as fhand:
            pkl = pickle.Unpickler(fhand)
            X, y = pkl.load()
        ln = len(X)
        listbox.insert(tk.END, 'Training ' + str(ln) + " samples...")
        clf.fit(X, y)
        listbox.insert(tk.END, 'Training Done')
        
        return

    #use built data, scale it, pickle it, and train model
    elif length == 0:
        listbox.insert(tk.END,'nn_trainLoop hit end')
        listbox.insert(tk.END, 'Scaling...')
        scaler = StandardScaler()
        lst = scaler.fit_transform(lst)
        listbox.insert(tk.END, 'scaling done')

        if m.get() == 0:
            X = np.array(lst)
            y = np.array(cats)
            ln = len(X)
            listbox.insert(tk.END, 'Training ' + str(ln) + " samples...")
            clf.fit(X, y)

            with open('trained', 'w+') as fhand:
                pkl = pickle.Pickler(fhand)
                pkl.dump((X, y))
                
            listbox.insert(tk.END, 'Training Done')

        return

    #build data in a loop
    cp = net.Capsule(data[index])
    qr = cp.series.get_values()
    lst.append(qr)
    cats.append(cp.cat)

    val = nn_labtxt2.get()
    nn_labtxt2.set(val+1)

    root.after(1, lambda: nn_trainLoop(index+1, length-1))

#Wrapper method for prediction loop
def nn_predict():
    global data
    if not data: 
        listbox.insert(tk.END, 'Predict done')
        return

    global prd
    prd = []
    global pcats
    pcats = []
    global bad
    bad = []

    nn_predictLoop(20000, len(data)-20000)

#loop method used by predict()
def nn_predictLoop(index, length):
    global prd
    global pcats
    global clf
    global bad

    #end of loop
    if length == 0:
        listbox.insert(tk.END,'nn_predictLoop hit end')
        prd = StandardScaler().fit_transform(prd)
        X = np.array(prd)
        y = np.array(pcats)
        ln = len(X)
        listbox.insert(tk.END, 'Predicting ' + str(ln) + " samples...")
        predictions = clf.predict(X)
        nn_predictDisplay(predictions, pcats, 0, len(prd))
        return

    cp = net.Capsule(data[index])
    pdata.append(cp)
    qr = cp.series.get_values()
    prd.append(qr)          
    pcats.append(cp.cat)

    val = nn_labtxt4.get()
    nn_labtxt4.set(val+1)

    root.after(1, lambda: nn_predictLoop(index+1, length-1))

#method for outputting results to tkinter box
def nn_predictDisplay(predictions, pcats, index, length):
    global bad
    if length == 0:
        varbox.bind('<<ListboxSelect>>', nn_onVarboxSelect)
        listbox.insert(tk.END, "Prediction done")
        print "number anomalies detected: ", len(bad)
        return
    #listbox.insert(tk.END, "Category: " + str(pcats[index]) + "  Prediction: " + str(predictions[index]))
    
    if not predictions[index] == 0:
        #varbox.insert(tk.END, net.Capsule(data[index+20000]).content)
        bad.append(pdata[index])
        varbox.insert(tk.END, pdata[index].content)

    root.after(1, lambda: nn_predictDisplay(predictions, pcats, index+1, length-1))

#displays details of predicted NN data from DB
def nn_onVarboxSelect(evt):
    global bad
    ind = int(varbox.curselection()[0])
    #print pdata[ind]
    spline = bad[ind].content.split()
    spline = '\n\n'.join(spline)
    br_labtxt0.set(spline)

#starter method for DBSCAN cluster analysis
def db_scan():
    global data
    data = T_Conn.select()
    numloops = len(data) - .10*len(data)

    global lst
    lst = []
    db_scanloop(0, numloops)

#looping method for DBSCAN cluster analysis
def db_scanloop(index, length):
    global data
    global lst
    global labels
    global core_samples_mask
    global n_clusters_

    if o.get() == 1:
        with open('cluster', 'r') as fhand:
            pkl = pickle.Unpickler(fhand)
            lst, db = pkl.load()

    elif o.get() == 0 and length <= 0:
        lst = StandardScaler().fit_transform(lst)

        db = DBSCAN(eps=0.3, min_samples=10).fit(lst)

        with open('cluster', 'w+') as fhand:
            pkl = pickle.Pickler(fhand)
            pkl.dump((lst, db))

    if o.get() == 1 or length <= 0:
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_

        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

        listbox.insert(tk.END, "Estimated number of clusters %d" % n_clusters_)
        listbox.insert(tk.END, "DBSCAN finished.")

        db_b1.config(state=tk.NORMAL)
        
        return

    #build data in a loop
    cp = net.Capsule_c(data[index]).series.get_values()
    lst.append(cp)

    val = db_labtxt2.get()
    db_labtxt2.set(val+1)

    root.after(1, lambda: db_scanloop(index+1,length-1))

#plotter for DBSCAN Cluster analysis
def db_plot(x, y):
    global labels
    global core_samples_mask
    global n_clusters_

    listbox.insert(tk.END, "x: %d y: %d" % (x, y))

    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0,1,len(unique_labels)))
    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = 'k'

        class_member_mask = (labels == k)

        xy = lst[class_member_mask & core_samples_mask]
        plt.plot(xy[:,x], xy[:, y], 'o', markerfacecolor=col, markeredgecolor='k', markersize=(3 if k == -1 else 14))

        xy = lst[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:,x], xy[:, y], 'o', markerfacecolor=col, markeredgecolor='k', markersize=(3 if k == -1 else 14))

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()

#UI Window definition for DBSCAN plotter
def plot_options():
    plotter = tk.Tk()
    plotter.wm_title("Plot Options")

    left = tk.Frame(plotter)
    left.pack(side=tk.LEFT)

    right = tk.Frame(plotter)
    right.pack(side=tk.RIGHT)

    a = tk.IntVar()
    tk.Label(left, text="X Axis").pack()
    tk.Radiobutton(left, text="time", variable=a, value=0, command=lambda: a.set(0)).pack()
    tk.Radiobutton(left, text="src", variable=a, value=1, command=lambda: a.set(1)).pack()
    tk.Radiobutton(left, text="dest", variable=a, value=2, command=lambda: a.set(2)).pack()
    tk.Radiobutton(left, text="port", variable=a, value=3, command=lambda: a.set(3)).pack()
    tk.Radiobutton(left, text="dur", variable=a, value=4, command=lambda: a.set(4)).pack()
    tk.Radiobutton(left, text="obytes", variable=a, value=5, command=lambda: a.set(5)).pack()
    tk.Radiobutton(left, text="rbytes", variable=a, value=6, command=lambda: a.set(6)).pack()
    tk.Button(left, text="Plot", command=lambda: db_plot(a.get(), b.get())).pack()

    b = tk.IntVar()
    tk.Label(right, text="Y Axis").pack()
    tk.Radiobutton(right, text="time", variable=b, value=0, command=lambda: b.set(0)).pack()
    tk.Radiobutton(right, text="src", variable=b, value=1, command=lambda: b.set(1)).pack()
    tk.Radiobutton(right, text="dest", variable=b, value=2, command=lambda: b.set(2)).pack()
    tk.Radiobutton(right, text="port", variable=b, value=3, command=lambda: b.set(3)).pack()
    tk.Radiobutton(right, text="dur", variable=b, value=4, command=lambda: b.set(4)).pack()
    tk.Radiobutton(right, text="obytes", variable=b, value=5, command=lambda: b.set(5)).pack()
    tk.Radiobutton(right, text="rbytes", variable=b, value=6, command=lambda: b.set(6)).pack()
    tk.Button(right, text="Cancel", command=lambda: plotter.destroy()).pack()

#kickstarter/callback method for the k-means analysis
def kmeans_start():
    global data
    data = T_Conn.select()
    #numloops = len(data) - .10*len(data)
    numloops = len(data)

    global num_clusters
    num_clusters = kmeans_sv0.get()

    global lst
    lst = []
    kmeans_loop(0, numloops)

#working loop for k-means analysis
def kmeans_loop(index, length):
    global data
    global lst
    global flat_lst
    global km

    if p.get() == 1:
        #retrieve saved data
        with open('kmeans', 'r') as fhand:
            pkl = pickle.Unpickler(fhand)
            lst, flat_lst, km = pkl.load()

    elif p.get() == 0 and length <= 0:

        #Normalize and flatten data prior to running through k-means

        #lst = RobustScaler().fit_transform(lst)    ====== Robust scaler doesn't produce normalized data like we want
        scaled_lst = StandardScaler().fit_transform(lst)

        flat_lst = PCA(n_components=2).fit_transform(scaled_lst)

        km = KMeans(init='k-means++', n_clusters=num_clusters, n_init=10)
        km.fit(flat_lst)

        #write out populated data, flattened data, and k-means model for quicker use next time
        with open('kmeans', 'w+') as fhand:
            pkl = pickle.Pickler(fhand)
            pkl.dump((lst, flat_lst, km))

    if p.get() == 1 or length <= 0:
        #find least populated cluster (outliers) and display items classified as such
        least = Counter(km.labels_).most_common()[:-3-1:-1]
        indices = []
        for i, val in enumerate(km.labels_):
            if(val==least[0][0] or val== least[1][0] or val==least[2][0]):
                indices.append(i)
        
        #clear varbox list and populate it with outliers
        global kdata
        kdata = []
        varbox.delete(0, tk.END)
        for ind in indices:
            kdata.append(ind)
            varbox.insert(tk.END, net.Capsule_c(data[ind]).content)

        #rebind varbox item select callback
        varbox.bind('<<ListboxSelect>>', kmeans_onVarboxSelect)

        listbox.insert(tk.END, ('Least common labels: ', least))
        listbox.insert(tk.END, "K-Means finished.")

        kmeans_b1.config(state=tk.NORMAL)
        
        return

    #build data in a loop
    cp = net.Capsule_c(data[index]).series.get_values()
    lst.append(cp)

    val = kmeans_sv2.get()
    kmeans_sv2.set(val+1)

    root.after(1, lambda: kmeans_loop(index+1,length-1))
   
#callback method for clicking on list item populated by k-means
def kmeans_onVarboxSelect(evt):
    global kdata
    ind = int(varbox.curselection()[0])
    spline = net.Capsule_c(data[kdata[ind]]).content.split()
    spline = '\n\n'.join(spline)
    br_labtxt0.set(spline)

def kmeans_plot():
    #step size of mesh
    h = .02

    #plot results
    x_min, x_max = flat_lst[:, 0].min() -1, flat_lst[:, 0].max()+1
    y_min, y_max = flat_lst[:, 1].min() -1, flat_lst[:, 1].max()+1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = km.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired,
               aspect='auto', origin='lower')

    plt.plot(flat_lst[:,0], flat_lst[:,1], 'k.', markersize=2)
    centroids = km.cluster_centers_
    plt.scatter(centroids[:,0], centroids[:,1],
                marker='x', s=169, linewidths=3,
                color='w', zorder=10)

    plt.title('K-Means clustering\n' 'Centroids are marked with white cross')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()

##########################################################
#                           MAIN                         #
#########################################################

##########  UI geometry  ############

############ topframe
sql_titlelab.grid(row=0, column=0)
sql_lab0.grid(row=2, column=0)
#http or conn
sql_rad0.grid(row=2, column=1)
sql_rad1.grid(row=2, column=2)
#clear DB label and radios
sql_lab1.grid(row=4, column=0)
sql_rad2.grid(row=4, column=1)
sql_rad3.grid(row=4, column=2)
#buttons
sql_lab2.grid(row=5, column=0)
sql_b1.grid(row=6, column=0)
sql_lab3.grid(row=6, column=1)
sql_lab4.grid(row=6, column=2)

############# midframe
mlopt_lab0.grid(row=0, column=0)
mlopt_rad0.grid(row=1, column=0)
mlopt_rad1.grid(row=1, column=1)
mlopt_rad2.grid(row=1, column=2)

############# midframe_nn
nn_lab0.grid(row=0, column=0)
nn_rad0.grid(row=1, column=0)
nn_rad1.grid(row=1, column=1)
nn_b0.grid(row=2, column=0)
nn_lab1.grid(row=2, column=1)
nn_lab2.grid(row=2, column=2)
nn_b1.grid(row=3, column=0)
nn_lab3.grid(row=3, column=1)
nn_lab4.grid(row=3, column=2)

############ midframe_dbscan
db_lab0.grid(row=0, column=0)
db_rad0.grid(row=1, column=0)
db_rad1.grid(row=1, column=1)
db_b0.grid(row=2, column=0)
db_lab1.grid(row=2, column=1)
db_lab2.grid(row=2, column=2)
db_b1.grid(row=3, column=0)

############ midframe_kmeans
kmeans_lab1.grid(row=1,column=0)
kmeans_rad0.grid(row=2, column=0)
kmeans_rad1.grid(row=2, column=1)
kmeans_lab0.grid(row=3, column=0)
kmeans_ent0.grid(row=3, column=1)
kmeans_b0.grid(row=4, column=0)
kmeans_lab2.grid(row=4, column=1)
kmeans_lab3.grid(row=4, column=2)
kmeans_b1.grid(row=5, column=0)

######### bl_frame
scroll1.pack(side=tk.RIGHT, fill=tk.Y)
listbox.pack(fill=tk.BOTH, expand=tk.YES)
scroll2.pack(side=tk.RIGHT, fill=tk.Y)
varbox.pack(fill=tk.BOTH, expand=tk.YES)
varbox.bind('<<ListboxSelect>>', nn_onVarboxSelect)

############# br_frame
br_lab0.pack()
br_message0.pack(expand=True, fill=tk.X)
br_b0.pack(side=tk.RIGHT)
br_b1.pack(side=tk.LEFT)


############  Execution  ############


root.mainloop()
