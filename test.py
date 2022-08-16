import nexusformat.nexus as nx
f = nx.nxload('data/fmri/sub-02_perceptionNaturalImageTraining_original_VC.h5')
print(f.tree)