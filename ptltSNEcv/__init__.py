name = "ptltSNEcv"

# Loading necessary libraries
libnames = [('mdtraj', 'md'), ('numpy', 'np'), ('tensorflow', 'tf'), ('keras', 'krs'), ('argparse', 'arg'), ('datetime', 'dt')]

for (name, short) in libnames:
  try:
    lib = __import__(name)
  except ImportError:
    print("Library %s is not installed or not working properly, exiting" % name)
    exit(0)
  else:
    globals()[short] = lib


def ptltSNEcollectivevariable(infilename='', intopname='', embed_dim=2, perplexity=30,
                              boxx=0.0, boxy=0.0, boxz=0.0, nofit=0, lagtime=1, maxpcs=50,
                              layers=1, layer1=256, layer2=256, layer3=256,
                              actfun1='sigmoid', actfun2='linear', actfun3='linear',
                              optim='adam', epochs=100, shuffle_interval=0, batch_size=0,
                              ofilename='', plumedfile='', plumedfile2='', fullcommand=''):

  def Hbeta(D, beta):
    P = np.exp(-D*beta)
    sumP = np.sum(P)
    H = np.log(sumP)+beta*np.sum(D*P)/sumP
    P = P/sumP
    return H, P

  def x2p(X, perplex, tol=1e-5):
    print("Computing pairwise distances...")
    (n, d) = X.shape
    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    P = np.zeros((n, n))
    beta = np.ones((n, 1))
    logU = np.log(perplex)
    for i in range(n):
      if i % 500 == 0:
        print("Computing P-values for point %d of %d..." % (i, n))
      betamin = -np.inf
      betamax = np.inf
      Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
      (H, thisP) = Hbeta(Di, beta[i])
      Hdiff = H - logU
      tries = 0
      while np.abs(Hdiff) > tol and tries < 50:
        if Hdiff > 0:
          betamin = beta[i].copy()
          if betamax == np.inf or betamax == -np.inf:
            beta[i] = beta[i] * 2.
          else:
            beta[i] = (beta[i] + betamax) / 2.
        else:
          betamax = beta[i].copy()
          if betamin == np.inf or betamin == -np.inf:
            beta[i] = beta[i] / 2.
          else:
            beta[i] = (beta[i] + betamin) / 2.
        (H, thisP) = Hbeta(Di, beta[i])
        Hdiff = H - logU
        tries += 1
      P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP
    print("Mean value of sigma: %f" % np.mean(np.sqrt(1 / beta)))
    return P

  def calculate_P(X, perplex, tol):
    n = X.shape[0]
    P = np.zeros([n, batch_size])
    for i in range(0, n, batch_size):
      P_batch = x2p(X[i:i + batch_size], perplex, tol)
      P_batch[np.isnan(P_batch)] = 0
      P_batch = P_batch + P_batch.T
      P_batch = P_batch / P_batch.sum()
      P_batch = np.maximum(P_batch, 1e-12)
      P[i:i + batch_size] = P_batch
    return P

  def KLdivergence(P, Y):
    alpha = embed_dim - 1.
    sum_Y = krs.backend.sum(krs.backend.square(Y), axis=1)
    #eps = krs.backend.variable(10e-15)
    D = sum_Y + krs.backend.reshape(sum_Y, [-1, 1]) - 2 * krs.backend.dot(Y, krs.backend.transpose(Y))
    Q = krs.backend.pow(1 + D / alpha, -(alpha + 1) / 2)
    #Q *= krs.backend.variable(1 - np.eye(batch_size))
    Q *= (1.0 - np.eye(batch_size))
    Q /= krs.backend.sum(Q)
    Q = krs.backend.maximum(Q, eps)
    C = krs.backend.log((P + eps) / (Q + eps))
    C = krs.backend.sum(P * C)
    return C

  eps = krs.backend.variable(10e-15)
  try:
    print("Loading trajectory")
    refpdb = md.load_pdb(intopname)
    traj = md.load(infilename, top=intopname)
    print("Fitting trajectory")
    if nofit==0:
      traj.superpose(refpdb)
  except IOError:
    print("Cannot load %s or %s, exiting." % (infilename, intopname))
    exit(0)
  else:
    print("%s succesfully loaded and fitted" % traj)
  print("")

  # Conversion of the trajectory from Nframes x Natoms x 3 to Nframes x (Natoms x 3)
  trajsize = traj.xyz.shape
  traj2 = np.zeros((trajsize[0], trajsize[1]*3))
  for i in range(trajsize[1]):
    traj2[:,3*i]   = traj.xyz[:,i,0]
    traj2[:,3*i+1] = traj.xyz[:,i,1]
    traj2[:,3*i+2] = traj.xyz[:,i,2]

  # Checking whether all atoms fit the box
  if (np.amin(traj2)) < 0.0:
    print("ERROR: Some of atom has negative coordinate (i.e. it is outside the box)")
    exit(0)

  if boxx == 0.0 or boxy == 0.0 or boxz == 0.0:
    print("WARNING: box size not set, it will be determined automatically")
    if boxx == 0.0:
      boxx = 1.2*np.amax(traj.xyz[:,:,0])
    if boxy == 0.0:
      boxy = 1.2*np.amax(traj.xyz[:,:,1])
    if boxz == 0.0:
      boxz = 1.2*np.amax(traj.xyz[:,:,2])
  print("box size set to %6.3f x %6.3f x %6.3f nm" % (boxx, boxy, boxz))
  print("")

  if np.amax(traj.xyz[:,:,0]) > boxx or np.amax(traj.xyz[:,:,1]) > boxy or np.amax(traj.xyz[:,:,2]) > boxz:
    print("ERROR: Some of atom has coordinate higher than box size (i.e. it is outside the box)")
    exit(0)

  if boxx > 2.0*np.amax(traj.xyz[:,:,0]) or boxy > 2.0*np.amax(traj.xyz[:,:,1]) or boxz > 2.0*np.amax(traj.xyz[:,:,2]):
    print("WARNING: Box size is bigger than 2x of highest coordinate,")
    print("maybe the box is too big or the molecule is not centered")

  maxbox = max([boxx, boxy, boxz])
  traj2 = traj2/maxbox

  meanc = np.mean(traj2, axis=0)
  traj2c = traj2-meanc
  cov1 = np.cov(np.transpose(traj2c))
  eva, eve = np.linalg.eig(cov1)
  order=np.argsort(eva)[::-1]
  eve = eve[:,order]
  eva = eva[order]
  projs = traj2c.dot(eve)
  projs = projs/np.sqrt(eva)
  C1 = np.transpose(projs[:-lagtime,]).dot(projs[lagtime:,])/(trajsize[0]-lagtime-1)
  C1 = (C1+np.transpose(C1))/2.0
  eva2, eve2 = np.linalg.eig(C1)
  order= np.argsort(eva2)[::-1]
  eve2 = eve2[:,order]
  eva2 = eva2[order]
  projs = projs.dot(eve2[:,:maxpcs])
  traj3 = projs*np.sqrt(np.real(eva2[:maxpcs]))

  n = trajsize[0]
  if batch_size > 0:
    batch_num = int(n // batch_size)
    m = batch_num * batch_size
  else:
    batch_num = 1
    batch_size = n
    m = n
  if shuffle_interval==0:
    shuffle_interval = epochs + 1

  # Model building
  print("Building model")
  input_coord = krs.layers.Input(shape=(3*trajsize[1],))
  encoded = krs.layers.Dense(layer1, activation=actfun1, use_bias=False)(input_coord)
  if layers == 3:
    encoded = krs.layers.Dense(layer2, activation=actfun2, use_bias=False)(encoded)
    encoded = krs.layers.Dense(layer3, activation=actfun3, use_bias=False)(encoded)
  if layers == 2:
    encoded = krs.layers.Dense(layer2, activation=actfun2, use_bias=False)(encoded)
  encoded = krs.layers.Dense(embed_dim, activation='linear', use_bias=True)(encoded)
  codecvs = krs.models.Model(input_coord, encoded)
  codecvs.compile(optimizer=optim, loss=KLdivergence)

  # Learning  
  print("Training model")
  for epoch in range(epochs):
    if epoch % shuffle_interval == 0:
      nr = np.random.permutation(n)
      X = traj2[nr[:m]]
      Xl = traj3[nr[:m]]
      P = calculate_P(Xl, perplexity, tol=1e-5)
    loss = 0.0
    for i in range(0, m, batch_size):
      loss += codecvs.train_on_batch(X[i:i+batch_size], P[i:i+batch_size])
    print("Epoch: {}/{}, loss: {}".format(epoch+1, epochs, loss / batch_num))

  # Encoding and decoding the trajectory
  coded_cvs = codecvs.predict(traj2) #/maxbox)
  # Generating low-dimensional output
  if len(ofilename) > 0:
    print("Writing tSNE collective variables for the training set into %s" % ofilename)
    print("")
    ofile = open(ofilename, "w")
    for i in range(trajsize[0]):
      for j in range(embed_dim):
        ofile.write(" %f" % coded_cvs[i,j])
      ofile.write("\n")
    ofile.close()

  if plumedfile != '':
    print("Writing Plumed < 2.6 input into %s" % plumedfile)
    print("")
    traj = md.load(infilename, top=intopname)
    table, bonds = traj.topology.to_dataframe()
    atoms = table['serial'][:]
    ofile = open(plumedfile, "w")
    if fullcommand != '':
      ofile.write("# command:\n")
      ofile.write("# %s\n" % fullcommand)
    ofile.write("# final KL devergence: %f\n" % (loss/batch_num))
    ofile.write("WHOLEMOLECULES ENTITY0=1-%i\n" % np.max(atoms))
    ofile.write("FIT_TO_TEMPLATE STRIDE=1 REFERENCE=%s TYPE=OPTIMAL\n" % intopname)
    for i in range(trajsize[1]):
      ofile.write("p%i: POSITION ATOM=%i NOPBC\n" % (i+1,atoms[i]))
    for i in range(trajsize[1]):
      ofile.write("p%ix: COMBINE ARG=p%i.x COEFFICIENTS=%f PERIODIC=NO\n" % (i+1,i+1,1.0/maxbox))
      ofile.write("p%iy: COMBINE ARG=p%i.y COEFFICIENTS=%f PERIODIC=NO\n" % (i+1,i+1,1.0/maxbox))
      ofile.write("p%iz: COMBINE ARG=p%i.z COEFFICIENTS=%f PERIODIC=NO\n" % (i+1,i+1,1.0/maxbox))
    if layers==1:
      for i in range(layer1):
        toprint = "l1_%i: COMBINE ARG=" % (i+1)
        for j in range(trajsize[1]):
          toprint = toprint + "p%ix,p%iy,p%iz," % (j+1,j+1,j+1)
        toprint = toprint[:-1] + " COEFFICIENTS="
        for j in range(3*trajsize[1]):
          toprint = toprint + "%0.6f," % (codecvs.layers[1].get_weights()[0][j,i])
        toprint = toprint[:-1] + " PERIODIC=NO\n"
        ofile.write(toprint)
      for i in range(layer1):
        if actfun1 == 'elu': printfun = "(exp(x)-1.0)*step(-x)+x*step(x)"
        elif actfun1 == 'selu': printfun = "1.0507*(1.67326*exp(x)-1.67326)*step(-x)+1.0507*x*step(x)"
        elif actfun1 == 'softplus': printfun = "log(1.0+exp(x))"
        elif actfun1 == 'softsign': printfun = "x/(1.0+step(x)*x+step(-x)*(-x))"
        elif actfun1 == 'relu': printfun = "step(x)*x"
        elif actfun1 == 'tanh': printfun = "(exp(x)-exp(-x))/(exp(x)+exp(-x))"
        elif actfun1 == 'sigmoid': printfun = "1.0/(1.0+exp(-x))"
        elif actfun1 == 'hard_sigmoid': printfun = "step(x+2.5)*((0.2*(x)+0.5)-step(x-2.5)*(0.2*(x)-0.5))"
        elif actfun1 == 'linear': printfun = "x"
        ofile.write("l1r_%i: MATHEVAL ARG=l1_%i FUNC=%s PERIODIC=NO\n" % (i+1,i+1,printfun))
      for i in range(embed_dim):
        toprint = "l2_%i: COMBINE ARG=" % (i+1)
        for j in range(layer1):
          toprint = toprint + "l1r_%i," % (j+1)
        toprint = toprint[:-1] + " COEFFICIENTS="
        for j in range(layer1):
          toprint = toprint + "%0.6f," % (codecvs.layers[2].get_weights()[0][j,i])
        toprint = toprint[:-1] + " PERIODIC=NO\n"
        ofile.write(toprint)
      for i in range(embed_dim):
        if codecvs.layers[2].get_weights()[1][i]>0.0:
          ofile.write("l2r_%i: MATHEVAL ARG=l2_%i FUNC=(x+%0.6f) PERIODIC=NO\n" % (i+1,i+1,codecvs.layers[2].get_weights()[1][i]))
        else:
          ofile.write("l2r_%i: MATHEVAL ARG=l2_%i FUNC=(x-%0.6f) PERIODIC=NO\n" % (i+1,i+1,-codecvs.layers[2].get_weights()[1][i]))
      toprint = "PRINT ARG="
      for i in range(embed_dim):
        toprint = toprint + "l2r_" + str(i+1) + ","
      toprint = toprint[:-1] + " STRIDE=100 FILE=COLVAR\n"
      ofile.write(toprint)
    if layers==2:
      for i in range(layer1):
        toprint = "l1_%i: COMBINE ARG=" % (i+1)
        for j in range(trajsize[1]):
          toprint = toprint + "p%ix,p%iy,p%iz," % (j+1,j+1,j+1)
        toprint = toprint[:-1] + " COEFFICIENTS="
        for j in range(3*trajsize[1]):
          toprint = toprint + "%0.6f," % (codecvs.layers[1].get_weights()[0][j,i])
        toprint = toprint[:-1] + " PERIODIC=NO\n"
        ofile.write(toprint)
      for i in range(layer1):
        if actfun1 == 'elu': printfun = "(exp(x)-1.0)*step(-x)+x*step(x)"
        elif actfun1 == 'selu': printfun = "1.0507*(1.67326*exp(x)-1.67326)*step(-x)+1.0507*x*step(x)"
        elif actfun1 == 'softplus': printfun = "log(1.0+exp(x))"
        elif actfun1 == 'softsign': printfun = "x/(1.0+step(x)*x+step(-x)*(-x))"
        elif actfun1 == 'relu': printfun = "step(x)*x"
        elif actfun1 == 'tanh': printfun = "(exp(x)-exp(-x))/(exp(x)+exp(-x))"
        elif actfun1 == 'sigmoid': printfun = "1.0/(1.0+exp(-x))"
        elif actfun1 == 'hard_sigmoid': printfun = "step(x+2.5)*((0.2*(x)+0.5)-step(x-2.5)*(0.2*(x)-0.5))"
        elif actfun1 == 'linear': printfun = "x"
        ofile.write("l1r_%i: MATHEVAL ARG=l1_%i FUNC=%s PERIODIC=NO\n" % (i+1,i+1,printfun))
      for i in range(layer2):
        toprint = "l2_%i: COMBINE ARG=" % (i+1)
        for j in range(layer1):
          toprint = toprint + "l1r_%i," % (j+1)
        toprint = toprint[:-1] + " COEFFICIENTS="
        for j in range(layer1):
          toprint = toprint + "%0.6f," % (codecvs.layers[2].get_weights()[0][j,i])
        toprint = toprint[:-1] + " PERIODIC=NO\n"
        ofile.write(toprint)
      for i in range(layer2):
        if actfun2 == 'elu': printfun = "(exp(x)-1.0)*step(-x)+x*step(x)"
        elif actfun2 == 'selu': printfun = "1.0507*(1.67326*exp(x)-1.67326)*step(-x)+1.0507*x*step(x)"
        elif actfun2 == 'softplus': printfun = "log(1.0+exp(x))"
        elif actfun2 == 'softsign': printfun = "x/(1.0+step(x)*x+step(-x)*(-x))"
        elif actfun2 == 'relu': printfun = "step(x)*x"
        elif actfun2 == 'tanh': printfun = "(exp(x)-exp(-x))/(exp(x)+exp(-x))"
        elif actfun2 == 'sigmoid': printfun = "1.0/(1.0+exp(-x))"
        elif actfun2 == 'hard_sigmoid': printfun = "step(x+2.5)*((0.2*(x)+0.5)-step(x-2.5)*(0.2*(x)-0.5))"
        elif actfun2 == 'linear': printfun = "x"
        ofile.write("l2r_%i: MATHEVAL ARG=l2_%i FUNC=%s PERIODIC=NO\n" % (i+1,i+1,printfun))
      for i in range(embed_dim):
        toprint = "l3_%i: COMBINE ARG=" % (i+1)
        for j in range(layer2):
          toprint = toprint + "l2r_%i," % (j+1)
        toprint = toprint[:-1] + " COEFFICIENTS="
        for j in range(layer2):
          toprint = toprint + "%0.6f," % (codecvs.layers[3].get_weights()[0][j,i])
        toprint = toprint[:-1] + " PERIODIC=NO\n"
        ofile.write(toprint)
      for i in range(embed_dim):
        if codecvs.layers[3].get_weights()[1][i]>0.0:
          ofile.write("l3r_%i: MATHEVAL ARG=l3_%i FUNC=(x+%0.6f) PERIODIC=NO\n" % (i+1,i+1,codecvs.layers[3].get_weights()[1][i]))
        else:
          ofile.write("l3r_%i: MATHEVAL ARG=l3_%i FUNC=(x-%0.6f) PERIODIC=NO\n" % (i+1,i+1,-codecvs.layers[3].get_weights()[1][i]))
      toprint = "PRINT ARG="
      for i in range(embed_dim):
        toprint = toprint + "l3r_" + str(i+1) + ","
      toprint = toprint[:-1] + " STRIDE=100 FILE=COLVAR\n"
      ofile.write(toprint)
    if layers==3:
      for i in range(layer1):
        toprint = "l1_%i: COMBINE ARG=" % (i+1)
        for j in range(trajsize[1]):
          toprint = toprint + "p%ix,p%iy,p%iz," % (j+1,j+1,j+1)
        toprint = toprint[:-1] + " COEFFICIENTS="
        for j in range(3*trajsize[1]):
          toprint = toprint + "%0.6f," % (codecvs.layers[1].get_weights()[0][j,i])
        toprint = toprint[:-1] + " PERIODIC=NO\n"
        ofile.write(toprint)
      for i in range(layer1):
        if actfun1 == 'elu': printfun = "(exp(x)-1.0)*step(-x)+x*step(x)"
        elif actfun1 == 'selu': printfun = "1.0507*(1.67326*exp(x)-1.67326)*step(-x)+1.0507*x*step(x)"
        elif actfun1 == 'softplus': printfun = "log(1.0+exp(x))"
        elif actfun1 == 'softsign': printfun = "x/(1.0+step(x)*x+step(-x)*(-x))"
        elif actfun1 == 'relu': printfun = "step(x)*x"
        elif actfun1 == 'tanh': printfun = "(exp(x)-exp(-x))/(exp(x)+exp(-x))"
        elif actfun1 == 'sigmoid': printfun = "1.0/(1.0+exp(-x))"
        elif actfun1 == 'hard_sigmoid': printfun = "step(x+2.5)*((0.2*(x)+0.5)-step(x-2.5)*(0.2*(x)-0.5))"
        elif actfun1 == 'linear': printfun = "x"
        ofile.write("l1r_%i: MATHEVAL ARG=l1_%i FUNC=%s PERIODIC=NO\n" % (i+1,i+1,printfun))
      for i in range(layer2):
        toprint = "l2_%i: COMBINE ARG=" % (i+1)
        for j in range(layer1):
          toprint = toprint + "l1r_%i," % (j+1)
        toprint = toprint[:-1] + " COEFFICIENTS="
        for j in range(layer1):
          toprint = toprint + "%0.6f," % (codecvs.layers[2].get_weights()[0][j,i])
        toprint = toprint[:-1] + " PERIODIC=NO\n"
        ofile.write(toprint)
      for i in range(layer2):
        if actfun2 == 'elu': printfun = "(exp(x)-1.0)*step(-x)+x*step(x)"
        elif actfun2 == 'selu': printfun = "1.0507*(1.67326*exp(x)-1.67326)*step(-x)+1.0507*x*step(x)"
        elif actfun2 == 'softplus': printfun = "log(1.0+exp(x))"
        elif actfun2 == 'softsign': printfun = "x/(1.0+step(x)*x+step(-x)*(-x))"
        elif actfun2 == 'relu': printfun = "step(x)*x"
        elif actfun2 == 'tanh': printfun = "(exp(x)-exp(-x))/(exp(x)+exp(-x))"
        elif actfun2 == 'sigmoid': printfun = "1.0/(1.0+exp(-x))"
        elif actfun2 == 'hard_sigmoid': printfun = "step(x+2.5)*((0.2*(x)+0.5)-step(x-2.5)*(0.2*(x)-0.5))"
        elif actfun2 == 'linear': printfun = "x"
        ofile.write("l2r_%i: MATHEVAL ARG=l2_%i FUNC=%s PERIODIC=NO\n" % (i+1,i+1,printfun))
      for i in range(layer3):
        toprint = "l3_%i: COMBINE ARG=" % (i+1)
        for j in range(layer2):
          toprint = toprint + "l2r_%i," % (j+1)
        toprint = toprint[:-1] + " COEFFICIENTS="
        for j in range(layer2):
          toprint = toprint + "%0.6f," % (codecvs.layers[3].get_weights()[0][j,i])
        toprint = toprint[:-1] + " PERIODIC=NO\n"
        ofile.write(toprint)
      for i in range(layer3):
        if actfun3 == 'elu': printfun = "(exp(x)-1.0)*step(-x)+x*step(x)"
        elif actfun3 == 'selu': printfun = "1.0507*(1.67326*exp(x)-1.67326)*step(-x)+1.0507*x*step(x)"
        elif actfun3 == 'softplus': printfun = "log(1.0+exp(x))"
        elif actfun3 == 'softsign': printfun = "x/(1.0+step(x)*x+step(-x)*(-x))"
        elif actfun3 == 'relu': printfun = "step(x)*x"
        elif actfun3 == 'tanh': printfun = "(exp(x)-exp(-x))/(exp(x)+exp(-x))"
        elif actfun3 == 'sigmoid': printfun = "1.0/(1.0+exp(-x))"
        elif actfun3 == 'hard_sigmoid': printfun = "step(x+2.5)*((0.2*(x)+0.5)-step(x-2.5)*(0.2*(x)-0.5))"
        elif actfun3 == 'linear': printfun = "x"
        ofile.write("l3r_%i: MATHEVAL ARG=l3_%i FUNC=%s PERIODIC=NO\n" % (i+1,i+1,printfun))
      for i in range(embed_dim):
        toprint = "l4_%i: COMBINE ARG=" % (i+1)
        for j in range(layer3):
          toprint = toprint + "l3r_%i," % (j+1)
        toprint = toprint[:-1] + " COEFFICIENTS="
        for j in range(layer3):
          toprint = toprint + "%0.6f," % (codecvs.layers[4].get_weights()[0][j,i])
        toprint = toprint[:-1] + " PERIODIC=NO\n"
        ofile.write(toprint)
      for i in range(embed_dim):
        if codecvs.layers[4].get_weights()[1][i]>0.0:
          ofile.write("l4r_%i: MATHEVAL ARG=l4_%i FUNC=(x+%0.6f) PERIODIC=NO\n" % (i+1,i+1,codecvs.layers[4].get_weights()[1][i]))
        else:
          ofile.write("l4r_%i: MATHEVAL ARG=l4_%i FUNC=(x-%0.6f) PERIODIC=NO\n" % (i+1,i+1,-codecvs.layers[4].get_weights()[1][i]))
      toprint = "PRINT ARG="
      for i in range(embed_dim):
        toprint = toprint + "l4r_" + str(i+1) + ","
      toprint = toprint[:-1] + " STRIDE=100 FILE=COLVAR\n"
      ofile.write(toprint)
    ofile.close()

  if plumedfile2 != '':
    print("Writing Plumed >= 2.6 input into %s" % plumedfile2)
    print("")
    traj = md.load(infilename, top=intopname)
    table, bonds = traj.topology.to_dataframe()
    atoms = table['serial'][:]
    ofile = open(plumedfile2, "w")
    if fullcommand != '':
      ofile.write("# command:\n")
      ofile.write("# %s\n" % fullcommand)
    ofile.write("# final KL devergence: %f\n" % (loss/batch_num))
    ofile.write("WHOLEMOLECULES ENTITY0=1-%i\n" % np.max(atoms))
    ofile.write("FIT_TO_TEMPLATE STRIDE=1 REFERENCE=%s TYPE=OPTIMAL\n" % intopname)
    for i in range(trajsize[1]):
      ofile.write("p%i: POSITION ATOM=%i NOPBC\n" % (i+1,atoms[i]))
    for i in range(trajsize[1]):
      ofile.write("p%ix: COMBINE ARG=p%i.x COEFFICIENTS=%f PERIODIC=NO\n" % (i+1,i+1,1.0/maxbox))
      ofile.write("p%iy: COMBINE ARG=p%i.y COEFFICIENTS=%f PERIODIC=NO\n" % (i+1,i+1,1.0/maxbox))
      ofile.write("p%iz: COMBINE ARG=p%i.z COEFFICIENTS=%f PERIODIC=NO\n" % (i+1,i+1,1.0/maxbox))
    if layers==1:
      ofile.write("ANN ...\n")
      ofile.write("LABEL=ann\n")
      toprint = "ARG="
      for j in range(trajsize[1]):
        toprint = toprint + "p%ix,p%iy,p%iz," % (j+1,j+1,j+1)
      toprint = toprint[:-1] + "\n"
      ofile.write(toprint)
      ofile.write("NUM_LAYERS=3\n")
      ofile.write("NUM_NODES=%i,%i,%i\n" % (3*trajsize[1],layer1,embed_dim))
      if actfun1 == 'tanh': 
        ofile.write("ACTIVATIONS=Tanh,Linear\n")
      elif actfun1 == 'linear': 
        ofile.write("ACTIVATIONS=Linear,Linear\n")
      else:
        print("ERROR: Only tanh or linear activation function supported in ANN module")
        exit(0)
      toprint = "WEIGHTS0="
      for i in range(layer1):
        for j in range(3*trajsize[1]):
          toprint = toprint + "%0.6f," % (codecvs.layers[1].get_weights()[0][j,i])
      toprint = toprint[:-1] + "\n"
      ofile.write(toprint)
      toprint = "WEIGHTS1="
      for i in range(embed_dim):
        for j in range(layer1):
          toprint = toprint + "%0.6f," % (codecvs.layers[2].get_weights()[0][j,i])
      toprint = toprint[:-1] + "\n"
      ofile.write(toprint)
      toprint = "BIASES0="
      for i in range(layer1):
        toprint = toprint + "0.0,"
      toprint = toprint[:-1] + "\n"
      ofile.write(toprint)
      toprint = "BIASES1="
      for i in range(embed_dim):
        toprint = toprint + "%0.6f," % (codecvs.layers[2].get_weights()[1][i])
      toprint = toprint[:-1] + "\n"
      ofile.write(toprint)
      ofile.write("... ANN\n")
      toprint = "PRINT ARG="
      for i in range(embed_dim):
        toprint = toprint + "ann.node-" + str(i) + ","
      toprint = toprint[:-1] + " STRIDE=100 FILE=COLVAR\n"
      ofile.write(toprint)
    if layers==2:
      ofile.write("ANN ...\n")
      ofile.write("LABEL=ann\n")
      toprint = "ARG="
      for j in range(trajsize[1]):
        toprint = toprint + "p%ix,p%iy,p%iz," % (j+1,j+1,j+1)
      toprint = toprint[:-1] + "\n"
      ofile.write(toprint)
      ofile.write("NUM_LAYERS=4\n")
      ofile.write("NUM_NODES=%i,%i,%i,%i\n" % (3*trajsize[1],layer1,layer2,embed_dim))
      if actfun1 == 'linear' and actfun2 == 'linear':
        ofile.write("ACTIVATIONS=Linear,Linear,Linear\n")
      elif actfun1 == 'linear' and actfun2 == 'tanh':
        ofile.write("ACTIVATIONS=Linear,Tanh,Linear\n")
      elif actfun1 == 'tanh' and actfun2 == 'linear':
        ofile.write("ACTIVATIONS=Tanh,Linear,Linear\n")
      elif actfun1 == 'tanh' and actfun2 == 'tanh':
        ofile.write("ACTIVATIONS=Tanh,Tanh,Linear\n")
      else:
        print("ERROR: Only tanh or linear activation function supported in ANN module")
        exit(0)
      toprint = "WEIGHTS0="
      for i in range(layer1):
        for j in range(3*trajsize[1]):
          toprint = toprint + "%0.6f," % (codecvs.layers[1].get_weights()[0][j,i])
      toprint = toprint[:-1] + "\n"
      ofile.write(toprint)
      toprint = "WEIGHTS1="
      for i in range(layer2):
        for j in range(layer1):
          toprint = toprint + "%0.6f," % (codecvs.layers[2].get_weights()[0][j,i])
      toprint = toprint[:-1] + "\n"
      ofile.write(toprint)
      toprint = "WEIGHTS2="
      for i in range(embed_dim):
        for j in range(layer2):
          toprint = toprint + "%0.6f," % (codecvs.layers[3].get_weights()[0][j,i])
      toprint = toprint[:-1] + "\n"
      ofile.write(toprint)
      toprint = "BIASES0="
      for i in range(layer1):
        toprint = toprint + "0.0,"
      toprint = toprint[:-1] + "\n"
      ofile.write(toprint)
      toprint = "BIASES1="
      for i in range(layer2):
        toprint = toprint + "0.0,"
      toprint = toprint[:-1] + "\n"
      ofile.write(toprint)
      toprint = "BIASES2="
      for i in range(embed_dim):
        toprint = toprint + "%0.6f," % (codecvs.layers[3].get_weights()[1][i])
      toprint = toprint[:-1] + "\n"
      ofile.write(toprint)
      ofile.write("... ANN\n")
      toprint = "PRINT ARG="
      for i in range(embed_dim):
        toprint = toprint + "ann.node-" + str(i) + ","
      toprint = toprint[:-1] + " STRIDE=100 FILE=COLVAR\n"
      ofile.write(toprint)
    if layers==3:
      ofile.write("ANN ...\n")
      ofile.write("LABEL=ann\n")
      toprint = "ARG="
      for j in range(trajsize[1]):
        toprint = toprint + "p%ix,p%iy,p%iz," % (j+1,j+1,j+1)
      toprint = toprint[:-1] + "\n"
      ofile.write(toprint)
      ofile.write("NUM_LAYERS=5\n")
      ofile.write("NUM_NODES=%i,%i,%i,%i,%i\n" % (3*trajsize[1],layer1,layer2,layer3,embed_dim))
      if actfun1 == 'linear' and actfun2 == 'linear' and actfun3 == 'linear':
        ofile.write("ACTIVATIONS=Linear,Linear,Linear,Linear\n")
      elif actfun1 == 'linear' and actfun2 == 'linear' and actfun3 == 'tanh':
        ofile.write("ACTIVATIONS=Linear,Linear,Tanh,Linear\n")
      elif actfun1 == 'linear' and actfun2 == 'tanh' and actfun3 == 'linear':
        ofile.write("ACTIVATIONS=Linear,Tanh,Linear,Linear\n")
      elif actfun1 == 'linear' and actfun2 == 'tanh' and actfun3 == 'tanh':
        ofile.write("ACTIVATIONS=Linear,Tanh,Tanh,Linear\n")
      elif actfun1 == 'tanh' and actfun2 == 'linear' and actfun3 == 'linear':
        ofile.write("ACTIVATIONS=Tanh,Linear,Linear,Linear\n")
      elif actfun1 == 'tanh' and actfun2 == 'linear' and actfun3 == 'tanh':
        ofile.write("ACTIVATIONS=Tanh,Linear,Tanh,Linear\n")
      elif actfun1 == 'tanh' and actfun2 == 'tanh' and actfun3 == 'linear':
        ofile.write("ACTIVATIONS=Tanh,Tanh,Linear,Linear\n")
      elif actfun1 == 'tanh' and actfun2 == 'tanh' and actfun3 == 'tanh':
        ofile.write("ACTIVATIONS=Tanh,Tanh,Tanh,Linear\n")
      else:
        print("ERROR: Only tanh activation function supported in ANN module")
        exit(0)
      toprint = "WEIGHTS0="
      for i in range(layer1):
        for j in range(3*trajsize[1]):
          toprint = toprint + "%0.6f," % (codecvs.layers[1].get_weights()[0][j,i])
      toprint = toprint[:-1] + "\n"
      ofile.write(toprint)
      toprint = "WEIGHTS1="
      for i in range(layer2):
        for j in range(layer1):
          toprint = toprint + "%0.6f," % (codecvs.layers[2].get_weights()[0][j,i])
      toprint = toprint[:-1] + "\n"
      ofile.write(toprint)
      toprint = "WEIGHTS2="
      for i in range(layer3):
        for j in range(layer2):
          toprint = toprint + "%0.6f," % (codecvs.layers[3].get_weights()[0][j,i])
      toprint = toprint[:-1] + "\n"
      ofile.write(toprint)
      toprint = "WEIGHTS3="
      for i in range(embed_dim):
        for j in range(layer3):
          toprint = toprint + "%0.6f," % (codecvs.layers[4].get_weights()[0][j,i])
      toprint = toprint[:-1] + "\n"
      ofile.write(toprint)
      toprint = "BIASES0="
      for i in range(layer1):
        toprint = toprint + "0.0,"
      toprint = toprint[:-1] + "\n"
      ofile.write(toprint)
      toprint = "BIASES1="
      for i in range(layer2):
        toprint = toprint + "0.0,"
      toprint = toprint[:-1] + "\n"
      ofile.write(toprint)
      toprint = "BIASES2="
      for i in range(layer3):
        toprint = toprint + "0.0,"
      toprint = toprint[:-1] + "\n"
      ofile.write(toprint)
      toprint = "BIASES3="
      for i in range(embed_dim):
        toprint = toprint + "%0.6f," % (codecvs.layers[4].get_weights()[1][i])
      toprint = toprint[:-1] + "\n"
      ofile.write(toprint)
      ofile.write("... ANN\n")
      toprint = "PRINT ARG="
      for i in range(embed_dim):
        toprint = toprint + "ann.node-" + str(i) + ","
      toprint = toprint[:-1] + " STRIDE=100 FILE=COLVAR\n"
      ofile.write(toprint)
    ofile.close()
  return coded_cvs

