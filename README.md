# ptltsne

Parametric time-lagged tSNE. The code fits the input trajectory to a reference PDB
file. Next, it introduces a time lag in the spirit of time-lagged independent component
analysis (TICA] method. Finally, it optimizes parameters of an artificial neural
network to provide tSNE representation as the output. It is possible to specify
time lag and neural network hyperparameters.

It uses Keras, PyTorch, MDtraj, and numpy.

Usage:
```
  -h, --help            show this help message and exit
  -i INFILE             Input trajectory in pdb, xtc, trr, dcd, netcdf or
                        mdcrd, WARNING: the trajectory must be 1. centered in
                        the PBC box, 2. fitted to a reference structure and 3.
                        must contain only atoms to be analysed!
  -p INTOP              Input topology in pdb, WARNING: the structure must be
                        1. centered in the PBC box and 2. must contain only
                        atoms to be analysed!
  -lag LAGTIME          Time lag in number of frames (default 1)
  -maxpcs MAXPCS        Number of TICA coordinates to be passed to t-SNE
                        (default 50)
  -dim EMBED_DIM        Number of output dimensions (default 2)
  -perplex PERPLEX      Value of t-SNE perplexity (default 30.0)
  -boxx BOXX            Size of x coordinate of PBC box (from 0 to set value
                        in nm)
  -boxy BOXY            Size of y coordinate of PBC box (from 0 to set value
                        in nm)
  -boxz BOXZ            Size of z coordinate of PBC box (from 0 to set value
                        in nm)
  -nofit NOFIT          Disable fitting, the trajectory must be properly fited
                        (default False)
  -layers LAYERS        Number of hidden layers (allowed values 1-3, default =
                        1)
  -layer1 LAYER1        Number of neurons in the first encoding layer (default
                        = 256)
  -layer2 LAYER2        Number of neurons in the second encoding layer
                        (default = 256)
  -layer3 LAYER3        Number of neurons in the third encoding layer (default
                        = 256)
  -actfun1 ACTFUN1      Activation function of the first layer (default =
                        sigmoid, for options see keras documentation)
  -actfun2 ACTFUN2      Activation function of the second layer (default =
                        linear, for options see keras documentation)
  -actfun3 ACTFUN3      Activation function of the third layer (default =
                        linear, for options see keras documentation)
  -optim OPTIM          Optimizer (default = adam, for options see keras
                        documentation)
  -epochs EPOCHS        Number of epochs (default = 100, >1000 may be
                        necessary for real life applications)
  -shuffle_interval SHUFFLE_INTERVAL
                        Shuffle interval (default = number of epochs + 1)
  -batch BATCH_SIZE     Batch size (0 = no batches, default = 0)
  -o OFILE              Output file with values of t-SNE embeddings (txt,
                        default = no output)
  -plumed PLUMEDFILE    Output file for Plumed (default = plumed.dat)
  -plumed2 PLUMEDFILE2  Output file for Plumed >= 2.6 (default - no output)
```


