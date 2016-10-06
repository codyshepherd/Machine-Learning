#Network Traffic & Machine Learning

This program is a GUI interface for running a few different machine learning algorithms on network traffic data.

It uses the `conn.log` and `http.log` generated by the program [Bro](www.bro.org) as its input. 

An active local MySQL or MariaDB database is required for this application to run. This tool will create and manage its own tables – no manual setup of the database is required other than, in the future, inputting database credentials. Currently the credentials for the local database are hard-coded in the ml_db.py file.

Three algorithms are used by this program:

###FF Neural Net

The Neural Net looks for beaconing behavior. It reads data from the http.log produced by Bro, which has been edited by hand to include beaconing classifications for each traffic session. After training on 20,000 http sessions and their provided classifications, it then can attempt to classify the 2,000 or so sessions remaining from the log. Any sessions deemed to display beaconing will be output to the lower list box on the GUI frame.

This algorithm has demonstrated modest success in detecting repetitious behavior.

Currently the neural net initialization parameters are hard-coded in the nnTrain() method in nfgen.py. However, in the future I intend to include UI elements for tweaking these initialization parameters. The current parameters have been chosen because they display the best performance in regards to beaconing. The documentation for the parameters can be found at the sklearn webpage.

The samples are scaled to zero mean and variance before training and prediction.

###DBSCAN Clustering

This is a clustering algorithm which defines clusters according to density parameters. In this tool it is used as a possible method of characterizing network traffic, and also as a way to visualize some of the relationships between the collected pieces of metadata. 

The Plot feature of this algorithm allows the user to choose two features from each sample and display one each on the chosen axis in the 2D plane. 

DBSCAN could potentially be used in the way that K-Means is used, as a way to define outliers. However, it tends to discard a large amount of collected samples (traffic sessions) as noise, and so may not ultimately be useful for that purpose.

Note that this algorithm requires that samples be scaled to zero mean and variance.

The initialization parameters for this algorithm can be tweaked.

###K-Means Clustering

The k-means algorithm separates samples into clusters based on variance, with the goal of creating clusters with equal variance. The advantage of this algorithm is that it includes every sample – i.e. it does not disregard some samples as noise the way DBSCAN does.

For the purposes of this tool, k-means has been leveraged to identify the three least-populated clusters. The tool then displays the samples (traffic sessions) that belong to those clusters as possible outliers or anomalies.

The Plot feature of this algorithm displays the clusters the algorithm has defined.

This algorithm requires scaling samples to zero mean and variance, as well as flattening them to two dimensions via PCA.

As always, initialization parameters are customizable.
