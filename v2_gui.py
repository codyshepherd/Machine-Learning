import Tkinter as tk
from v2_data import *
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

class GUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.wm_title("Machine Learning Tool")
        self.root.columnconfigure(0,weight=1)

        self.l1_frame = tk.Frame(self.root)
        self.l1_frame.grid(row=0, column=0, sticky='nsew')
        
        self.l1_subframe = tk.Frame(self.root)
        self.l1_subframe.grid(row=1, column=0, sticky='nsew')

        self.l1_subframe2 = tk.Frame(self.root)
        self.l1_subframe2.grid(row=2, column=0, sticky='nsew')

        self.l2_frame = tk.Frame(self.root)
        self.l2_frame.grid(row=3, column=0, sticky='nsew')

        self.l2_subframe = tk.Frame(self.root)
        self.l2_subframe.grid(row=4, column=0, sticky='nsew')

        self.l3_frame = tk.Frame(self.root)
        self.l3_frame.grid(row=5, column=0, sticky='nsew')

        #l1_frame elements -- log type and path location
        self.l1_frame_titlelab = tk.Label(self.l1_frame, text="Source Log Options", font='Helvetica 16 bold', pady=5, padx=20).grid(row=0, column=0, sticky="w")

        self.l1_frame_typelab = tk.Label(self.l1_frame, text="Source Log Type", pady=5, padx=20).grid(row=1, column=0, sticky="w")

        self.logtype = tk.IntVar()
        tk.Radiobutton(self.l1_subframe, text="Bro conn.log", variable=self.logtype, value=0, command=None, padx=20).grid(row=0,column=0, sticky="w") #TODO: callback
        tk.Radiobutton(self.l1_subframe, text="Other", variable=self.logtype, value=1, command=None, padx=20).grid(row=0, column=1, sticky="w")

        self.l1_frame_pathlab = tk.Label(self.l1_subframe, text="Log Path", pady=5, padx=20).grid(row=1, column=0, sticky="w")

        self.path = tk.StringVar()
        self.l1_frame_pathentry = tk.Entry(self.l1_subframe2, textvariable=self.path).grid(row=2, column=0, sticky="w", padx=20)

        #l2_frame elements -- ML algorithm selection

        self.l2_frame_titlelab = tk.Label(self.l2_frame, text="Machine Learning Options", font='Helvetica 16 bold', pady=5, padx=20).grid(row=0,column=0, sticky="w")

        self.l2_frame_alglab = tk.Label(self.l2_frame, text="Select Algorithm", pady=5, padx=20).grid(row=1, column=0, sticky="w")

        self.alg = tk.IntVar()
        self.alg.set(0)
        tk.Radiobutton(self.l2_subframe, text="Neural Network", variable=self.alg, value=0, command=None, padx=20).grid(row=0, column=0, sticky="e") #TODO: callback
        tk.Radiobutton(self.l2_subframe, text="K-Means Clustering", variable=self.alg, value=1, command=None, padx=20).grid(row=0, column=1, sticky="w") #TODO: callback
        tk.Radiobutton(self.l2_subframe, text="DBSCAN Clustering", variable=self.alg, value=2, command=None, padx=20).grid(row=0, column=2, sticky="w") #TODO: callback

        #l3_frame elements -- algorithm config panes

        #TODO: the rest
        
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



