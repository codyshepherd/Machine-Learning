import Tkinter as tk
from v2_data import *
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

class GUI:
    MAIN_TITLE = 'Machine Learning Tool'
    L1_TITLE = 'Source Log Options'

    STD_STICKY = 'nsew'
    STD_TITLE_FONT = 'Helvetica 16 bold'
    STD_PADX = 20
    STD_PADY = 5

    def __init__(self):

        #Frames
        self.root = tk.Tk()
        self.root.wm_title(self.MAIN_TITLE)
        self.root.columnconfigure(0,weight=1)

        self.l1_frame = tk.Frame(self.root)
        self.l1_frame.grid(row=0, column=0, sticky=self.STD_STICKY)
        
        self.l1_subframe = tk.Frame(self.root)
        self.l1_subframe.grid(row=1, column=0, sticky=self.STD_STICKY)

        self.l1_subframe2 = tk.Frame(self.root)
        self.l1_subframe2.grid(row=2, column=0, sticky=self.STD_STICKY)

        self.l2_frame = tk.Frame(self.root)
        self.l2_frame.grid(row=3, column=0, sticky=self.STD_STICKY)

        self.l2_subframe = tk.Frame(self.root)
        self.l2_subframe.grid(row=4, column=0, sticky=self.STD_STICKY)

        self.l3_nn = tk.Frame(self.root)
        self.l3_nn.grid(row=5, column=0, sticky=self.STD_STICKY)

        self.l3_kmeans = tk.Frame(self.root)
        self.l3_kmeans.grid(row=5,column=0, sticky=self.STD_STICKY)

        self.l3_dbscan = tk.Frame(self.root)
        self.l3_dbscan.grid(row=5,column=0, sticky=self.STD_STICKY)
        self.l3_nn.tkraise()
        
        self.l4_frame = tk.Frame(self.root)
        self.l4_frame.grid(row=6,column=0, sticky=self.STD_STICKY)
        self.root.rowconfigure(6,weight=1)      #ensures the listbox will grow as windows is expanded

        #l1_frame elements -- log type and path location
        self.l1_frame_titlelab = tk.Label(self.l1_frame, text=self.L1_TITLE, font=self.STD_TITLE_FONT, pady=self.STD_PADY, padx=self.STD_PADX).grid(row=0, column=0, sticky="w")

        self.l1_frame_typelab = tk.Label(self.l1_frame, text='Source Log Type', pady=self.STD_PADY, padx=self.STD_PADX).grid(row=1, column=0, sticky="w")

        self.logtype = tk.IntVar()
        tk.Radiobutton(self.l1_subframe, text="Bro conn.log", variable=self.logtype, value=0, command=None, padx=self.STD_PADX).grid(row=0,column=0, sticky="w") #TODO: callback
        #callback: backend_set_doc(BRO)
        tk.Radiobutton(self.l1_subframe, text="Other", variable=self.logtype, value=1, command=None, padx=self.STD_PADX).grid(row=0, column=1, sticky="w")
        #callback: backend_set_doc(OTHER)

        self.l1_frame_pathlab = tk.Label(self.l1_subframe, text="Log Path", pady=self.STD_PADY, padx=self.STD_PADX).grid(row=1, column=0, sticky="w")

        self.path = tk.StringVar()
        self.l1_frame_pathentry = tk.Entry(self.l1_subframe2, textvariable=self.path).grid(row=2, column=0, sticky="w", padx=self.STD_PADX)

        #l2_frame elements -- ML algorithm selection

        self.l2_frame_titlelab = tk.Label(self.l2_frame, text="Machine Learning Options", font=self.STD_TITLE_FONT, pady=self.STD_PADY, padx=self.STD_PADX).grid(row=0,column=0, sticky="w")

        self.l2_frame_alglab = tk.Label(self.l2_frame, text="Select Algorithm", pady=self.STD_PADY, padx=self.STD_PADX).grid(row=1, column=0, sticky="w")

        #Radio Buttons for Algorithm Selection
        self.alg = tk.IntVar()
        self.alg.set(0)
        tk.Radiobutton(self.l2_subframe, text="Neural Network", variable=self.alg, value=0, \
            command=lambda: (self.alg.set(0), self.l3_nn.tkraise()), \
            padx=self.STD_PADX).grid(row=0, column=0, sticky="e")
        tk.Radiobutton(self.l2_subframe, text="K-Means Clustering", variable=self.alg, value=1, \
            command=lambda: (self.alg.set(1), self.l3_kmeans.tkraise()), \
            padx=self.STD_PADX).grid(row=0, column=1, sticky="w") 
        tk.Radiobutton(self.l2_subframe, text="DBSCAN Clustering", variable=self.alg, value=2, \
            command=lambda: (self.alg.set(2), self.l3_dbscan.tkraise()), \
            padx=self.STD_PADX).grid(row=0, column=2, sticky="w")

        #l3_frame elements -- algorithm config panes

        #Neural Net Pane
        self.nn_title_lab = tk.Label(self.l3_nn, text=" ", pady=self.STD_PADY, padx=self.STD_PADX).grid(row=0, column=0)

        self.nn_b0 = tk.Button(self.l3_nn, text="Train", command=None).grid(row=1, column=0, sticky='e', padx=self.STD_PADX)            #TODO: Callback
        #callback: backend_nn_train()
        #subcallback: gui_update(nn_trained_num)
        self.nn_b1 = tk.Button(self.l3_nn, text="Load", command=None, state=tk.DISABLED).grid(row=1, column=1, padx=self.STD_PADX)      #TODO: Callback
        #callback: backend_nn_unpickle()
        self.nn_b2 = tk.Button(self.l3_nn, text="Predict", command=None, state=tk.DISABLED).grid(row=1, column=2, padx=self.STD_PADX)   #TODO: Callback
        #callback: backend_nn_predict()
        #subcallback: gui_update(nn_predict_num)

        self.nn_trained_lab = tk.Label(self.l3_nn, text="Number of Samples Trained:", pady=self.STD_PADY, padx=self.STD_PADX).grid(row=2,column=0)
        self.nn_trained_num = tk.IntVar()
        self.nn_trained_num.set(0)
        self.nn_trained_num_lab = tk.Label(self.l3_nn, textvariable=self.nn_trained_num, pady=self.STD_PADY).grid(row=2,column=1)

        self.nn_predict_lab = tk.Label(self.l3_nn, text='Number of Samples Predicted:', pady=self.STD_PADY, padx=self.STD_PADX).grid(row=3,column=0)
        self.nn_predict_num = tk.IntVar()
        self.nn_predict_num.set(0)
        self.nn_predict_num_lab = tk.Label(self.l3_nn, textvariable=self.nn_predict_num, pady=self.STD_PADY).grid(row=3,column=1)

        #Kmeans Pane
        self.kmeans_title_lab = tk.Label(self.l3_kmeans, text=" ", pady=self.STD_PADY, padx=self.STD_PADX).grid(row=0, column=0)

        self.kmeans_cluster_lab = tk.Label(self.l3_kmeans, text='Number of Clusters:', pady=self.STD_PADY, padx=self.STD_PADX).grid(row=2,column=0)
        self.kmeans_cluster_num = tk.IntVar()
        self.kmeans_cluster_num.set(20)
        self.kmeans_cluster_num_lab = tk.Entry(self.l3_kmeans, textvariable=self.kmeans_cluster_num, width=3).grid(row=2,column=1)

        self.kmeans_b0 = tk.Button(self.l3_kmeans, text="Compute", command=None).grid(row=3,column=0,sticky='e', padx=self.STD_PADX)        #TODO: Callback
        #callback: backend_kmeans_compute()
        #subcallback: gui_update(kmeans_processed_num)
        self.kmeans_b1 = tk.Button(self.l3_kmeans, text="Load", command=None, state=tk.DISABLED).grid(row=3,column=1, padx=self.STD_PADX)   #TODO: Callback
        #callback: backend_kmeans_unpickle()
        self.kmeans_b2 = tk.Button(self.l3_kmeans, text="Plot", command=None, state=tk.DISABLED).grid(row=3,column=2, padx=self.STD_PADX)   #TODO: Callback
        #callback: newgui_plot(kmeans_results)

        self.kmeans_processed_lab = tk.Label(self.l3_kmeans, text='Number of Samples Processed:', pady=self.STD_PADY, padx=self.STD_PADX).grid(row=4,column=0)
        self.kmeans_processed_num = tk.IntVar()
        self.kmeans_processed_num.set(0)
        self.kmeans_processed_num_lab = tk.Label(self.l3_kmeans, textvariable=self.kmeans_processed_num).grid(row=4,column=1)

        #DBSCAN Pane
        self.dbscan_title_lab = tk.Label(self.l3_dbscan, text=" ", pady=self.STD_PADY, padx=self.STD_PADX).grid(row=0, column=0)
        
        self.db_b0 = tk.Button(self.l3_dbscan, text="Compute", command=None).grid(row=1, column=0, sticky='e', padx=self.STD_PADX)      #TODO:Callback
        #callback: backend_dbscan_compute()
        #callback: gui_update(db_processed_num)
        self.db_b1 = tk.Button(self.l3_dbscan, text="Load", command=None, state=tk.DISABLED).grid(row=1, column=1, padx=self.STD_PADX)  #TODO:Callback
        #callback: backend_dbscan_unpickle()
        self.db_b2 = tk.Button(self.l3_dbscan, text="Plot", command=None, state=tk.DISABLED).grid(row=1, column=2, padx=self.STD_PADX)  #TODO:Callback
        #callback: newgui_plot(dbscan_results)

        self.db_processed_lab = tk.Label(self.l3_dbscan, text='Number of Samples Processed:', pady=self.STD_PADY, padx=self.STD_PADX).grid(row=2,column=0)
        self.db_processed_num = tk.IntVar()
        self.db_processed_num.set(0)
        self.db_processed_num_lab = tk.Label(self.l3_dbscan, textvariable=self.db_processed_num).grid(row=2,column=1)

        #l4_frame elements

        self.listscroll=tk.Scrollbar(self.l4_frame)
        self.out_list = tk.Listbox(self.l4_frame, yscrollcommand=self.listscroll.set)
        self.listscroll.config(command=self.out_list.yview)
        self.out_list.pack(fill='both', expand=True)
        

        #TODO: callbacks in v2_callbacks.py
        #
        #consider threads or multiprocessing library for GUI updating concurrent with computations
        
        self.root.mainloop()














gui = GUI()


"""
    def nn_init(self, act, alg, alph, sz, st):
        self.clf = MLPClassifier(activation=act, solver=alg, alpha=alph, hidden_layer_sizes=sz, random_state=st)

    def nn_train(self, size):
        batch = []
        with Data('conn.log',Doc_t.BRO) as d:
            batch += [line.series.get_values() for line in d.get_lines(size)]
        scaler = StandardScaler()
        batch = scaler.fit_transform(batch)
        y = [1 for i in xrange(size)]
        self.clf.fit(batch, y)
"""



