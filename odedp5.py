"""
This software is copyrighted material (C) Shai Revzen 2009-2020

The library is provided under the GNU Public License version 3.0,
    as found at http://www.gnu.org/licenses/gpl-3.0-standalone.html

  OVERVIEW:

  This file contains a multipurpose ODE integrator designed to support
  integration of smooth ODEs and of hybrid systems. Its performance
  at the time of development was comparable to that of ode45 when
  running on the same hardware and operating system.

  If you have matplotlib installed, running this file should generate
  plots of key unit tests.

  MAIN CLASSES:

    odeDP5 -- typically the only class you will use, integrates an ODE
    integro -- "guts" of integrator, ported from C
    TrajPatch -- a "dense" output trajectory patch. This wraps a polynomial that solves the ODE locally.

  TECHNICAL DETAILS:

  This integrator is a python port of the dopri5 integrator code described
  by: E. Hairer & G. Wanner
      Universite de Geneve, dept. de Mathematiques
      CH-1211 GENEVE 4, SWITZERLAND
      E-mail : HAIRER@DIVSUN.UNIGE.CH, WANNER@DIVSUN.UNIGE.CH

  The code is described in : E. Hairer, S.P. Norsett and G. Wanner, Solving
  ordinary differential equations I, nonstiff problems, 2nd edition,
  Springer Series in Computational Mathematics, Springer-Verlag (1993).
"""

__all__=['odeDP5','Integro','TrajPatch']

#
# I derived the code from a MEX interface to dopri5 written by
# Madhusudhan Venkadesan during his work with John Gukenheimer
#
# By Shai Revzen, Berkeley 2008
from sys import stdout
from numpy import \
    asarray,array,empty,sign,dot,sqrt,newaxis,ones_like,\
    zeros,concatenate,tensordot,vstack,diff,c_

# Python 3 compatibility hacks
try:
    assert callable(xrange)
except NameError:
    xrange = range

class TrajPatch(object):
  """Trajectory patch from integrator

     This is a callable that returns the trajectory evaluated at the
     specified times.

     Use the .range field to find the range in which the patch is valid
  """
  def __init__(self, xold, hout, rcont):
    self.rcont = rcont
    self.xold = xold
    self.hout = hout
    self.range = (xold,xold+hout)

  def __call__(self, x, idx=slice(None)):
    """evaluate trajectory at specified time"""
    theta = (x - self.xold) / self.hout
    theta1 = 1.0 - theta
    coef = vstack([ones_like(theta),theta,theta1,theta,theta1]).cumprod(axis=0)
    return tensordot(coef,self.rcont,(0,0))[newaxis,idx]

  def refine( self, evtFun, tol=1e-9, *args ):
    """Find the time at which the trajectory is a root of the function evtFun.

       evtFun is a function taking (t,y,*args) and returning a real number.
       refine() uses trajectory patch to find the "time"
       at which evtFun(t,y(t),*args)==0.

       This code requires that the signs of evtFun in the beginning and end of
       the patch are opposite. It finds the root by bisection -- a binary
       search.

       returns t,y -- the end time and state; t is correct to within +/- tol
    """
    t0,t1 = self.range
    for k in xrange(4):
      y = self(t0).squeeze()
      f0 = evtFun(t0,y, *args )
      y = self(t1).squeeze()
      f1 = evtFun(t1,y, *args )
      if f0*f1 <= 0:
        break
      t0,t1 = t0-(t1-t0)/2,t1+(t1-t0)/2
      print("WARNING: function did not change sign -- extrapolating order ",k)
    if f1<0:
      t0,f0,t1,f1 = t1,f1,t0,f0
    while abs(t1-t0)>tol:
      t = (t1+t0)/2
      y = self(t).squeeze()
      f = evtFun(t,y, *args )
      if f==0:
        break
      if f>0:
        t1,f1 = t,f
      else:
        t0,f0 = t,f
    return (t1+t0)/2,y

class Integro(object):
  """Abstract superclass implementing a dopri5 integrator

  """
  def __init__(self):
    for fn in ( "nfcn, nstep, naccpt, nrejct, hout, xold, xout, nrds, "
                "indir, yy1, kk, ysti, rcont, trj").split(", "):
      setattr(self,fn,None)
    self.ODE_ATOL = 1e-9
    self.ODE_RTOL = 1e-6
    self.ODE_EVTTOL = 1e-12
    self.ODE_MAX = 1e20
    self.FORCE_STEP = None

  def safeOdeFunc (self, n, x, y, pars, dy ):
    """PURE METHOD: override this to implement the vector field function being
       integrated.

       n -- dimension of the system
       x -- independent variable (scalar)
       y -- state as an array of shape (N,)
       pars -- additional pass through parameters
       dy -- output array, with shape (N,)

       If the method returns a python "true", i.e. something that an if
       statement would treat as true, integration is stopped.

       WARNING: Never assign to dy! Values should be outputed by setting the
       contents of dy, e.g. dy[:] = -y would give a decaying exponential.
    """
    raise TypeError("Pure method called")

  def solout (self,ni,xold, x, y, ny, *args):
    """PURE METHOD: Solution output

       ni -- index of output point
       xold -- start of most recent time segment
       x -- end of most recent time segment
       y -- state at time x, an array
       ny -- dimension of y
       *args -- additional parameters
    """
    raise TypeError("Pure method called")

  def hinit(self, n, x, y, pars, posneg, f0, f1, yy1, iord, hmax, atoler, rtoler, itoler):
    """port of hinit()"""
    dnf = 0.0
    dny = 0.0
    for k in xrange(n):
      sk = atoler[k] + rtoler[k] * abs (y[k])
      sqr = f0[k] / sk
      dnf += sqr * sqr
      sqr = y[k] / sk
      dny += sqr * sqr
    if ((dnf <= 1.0E-10) or (dny <= 1.0E-10)):
      h = 1.0E-6
    else:
      h = sqrt(dny / dnf) * 0.01
    h = min(h, hmax)
    if self.FORCE_STEP:
      h = self.FORCE_STEP
    h = h * sign(posneg)         #  perform an explicit Euler step
    for k in xrange(n):
      self.yy1[k] = y[k] + h * f0[k]
    self.safeOdeFunc (n, x + h, self.yy1, pars, f1)  #  estimate the second derivative of the solution
    der2 = 0.0
    for k in xrange(n):
      sk = atoler[k] + rtoler[k] * abs (y[k])
      sqr = (f1[k] - f0[k]) / sk
      der2 += sqr * sqr

    der2 = sqrt (der2) / h       # step size is computed such that h**iord * max(norm(f0),norm(der2)) = 0.01
    der12 = max (abs (der2), sqrt (dnf))
    if (der12 <= 1.0E-15):
      h1 = max (1.0E-6, abs (h) * 1.0E-3)
    else:
      h1 = pow (0.01 / der12, 1.0 /  iord)
    h = min (100.0 * h, min (h1, hmax))
    if self.FORCE_STEP:
      h = self.FORCE_STEP
    return h * sign(posneg)

  def dopcor (self,n, x, y, pars, xend, hmax, h, rtoler, atoler, itoler, fileout,
    iout, nmax, magbound, uround, meth, nstiff, safe, beta, fac1, fac2, icont):
    """port of dopcore(); this is the core function of the integrator"""
    c2 = 0.2; c3 = 0.3; c4 = 0.8; c5 = 8.0 / 9.0
    a21 = 0.2
    a31 = 3.0 / 40.0; a32 = 9.0 / 40.0
    a4 = array([44.0 / 45.0,-56.0 / 15.0,32.0 / 9.0])
    a5 = array([19372.0 / 6561.0, -25360.0 / 2187.0,
            64448.0 / 6561.0,-212.0 / 729.0])
    a6 = array([9017.0 / 3168.0,-355.0 / 33.0,46732.0 / 5247.0,
            49.0 / 176.0, -5103.0 / 18656.0])
    a7 = array([35.0 / 384.0,0,500.0 / 1113.0,
            125.0 / 192.0,-2187.0 / 6784.0,11.0 / 84.0])
    ee = array([ 71.0 / 57600.0, -1.0 / 40.0, -71.0 / 16695.0,
            71.0 / 1920.0, -17253.0 / 339200.0, 22.0 / 525.0 ])
    dd = array([ -12715105075.0 / 11282082432.0, 69997945.0 / 29380423.0,
            87487479700.0 / 32700410799.0, -10690763975.0 / 1880347072.0,
            701980252875.0 / 199316789632.0, -1453857185.0 / 822651844.0 ])

    facold = 1.0E-4
    expo1 = 0.2 - beta * 0.75
    facc1 = 1.0 / fac1
    facc2 = 1.0 / fac2
    posneg = sign (xend - x)
    last = 0
    hlamb = 0.0
    iasti = 0
    self.safeOdeFunc (n, x, y, pars, self.kk[0,:])
    hmax = abs (hmax)
    iord = 5

    if (h == 0.0):
      h = self.hinit (n, x, y, pars, posneg,
            self.kk[0,:], self.kk[1,:], self.kk[2,:],
            iord, hmax, atoler, rtoler, itoler)
    self.nfcn += 2
    reject = 0
    self.xold = x

    self.hout = h
    self.xout = x
    if self.solout (self.naccpt + 1, self.xold, x, y, n, *pars):
      return -1
    while True:
      if (self.nstep > nmax):
        self.xout = x
        self.hout = h
        return -2
      if (not self.FORCE_STEP) and (0.1 * abs (h) <= abs (x) * uround):
        if (fileout):
          fileout.write( "Step size too small h = %.16e\n" % h)
        self.xout = x
        self.hout = h
        return -3

      for k in xrange(n):
        if (abs (y[k]) > magbound):
          fileout.write(
                   "Solution exceeds bound on component's magnitude  ('magbound'=%f)\n"
                   % magbound)
          self.xout = x
          self.hout = h
          return -5
      if ((x + 1.01 * h - xend) * posneg > 0.0):
        h = xend - x
        last = 1
      self.nstep += 1
      #  the first 6 stages

      self.yy1 = y + h*a21 * self.kk[0,:]
      self.safeOdeFunc (n, x + c2 * h, self.yy1, pars, self.kk[1,:])
      self.yy1 = y + h*( a31 * self.kk[0,:] + a32 * self.kk[1,:] )
      self.safeOdeFunc (n, x + c3 * h, self.yy1, pars, self.kk[2,:])
      self.yy1 = y + h*dot( a4, self.kk[:3,:] )
      self.safeOdeFunc (n, x + c4 * h, self.yy1, pars, self.kk[3,:])
      self.yy1 = y + h*dot( a5, self.kk[:4,:] )
      self.safeOdeFunc (n, x + c5 * h, self.yy1, pars, self.kk[4,:])
      self.ysti = y + h*dot(a6,self.kk[:5,:])
      xph = x + h
      self.safeOdeFunc (n, xph, self.ysti, pars, self.kk[5,:])
      self.yy1 = y + h*dot(a7,self.kk[:6,:])
      self.safeOdeFunc (n, xph, self.yy1, pars, self.kk[1,:])
      #self.rcont5 = h*dot(dd,self.kk[:6,:])
      self.rcont[4,:] = h*dot(dd,self.kk[:6,:])
      self.kk[3,:] = h*dot(ee,self.kk[:6,:])
      self.nfcn += 6                #   error estimation
      err = 0.0
      for k in xrange(n):
        sk = atoler[k] + rtoler[k] * max (abs (y[k]), abs (self.yy1[k]))
        sqr = self.kk[3,k] / sk
        err += sqr * sqr
      err = sqrt (err / float(n)) #  computation of hnew
      fac11 = pow (err, expo1)  #  Lund-stabilization
      fac = fac11 / pow (facold, beta) #  we require fac1 <= hnew/h <= fac2
      fac = max (facc2, min (facc1, fac / safe))
      hnew = h / fac
      if (err <= 1.0) or self.FORCE_STEP: #            step accepted
        facold = max (err, 1.0E-4)
        self.naccpt+=1      #            stiffness test
        if ( not (self.naccpt % nstiff) or (iasti > 0)):
          stnum = 0.0
          stden = 0.0
          for k in xrange(n):
            sqr = self.kk[1,k] - self.kk[5,k]
            stnum += sqr * sqr
            sqr = self.yy1[k] - self.ysti[k]
            stden += sqr * sqr

          if (stden > 0.0):
            hlamb = h * sqrt (stnum / stden)
          if (hlamb > 3.25):
            nonsti = 0
            iasti+=1
            if (iasti == 15):
              if (fileout):
                fileout.write( "Stiffness detected.\n")
              self.xout = x
              self.hout = h
              return -4
          else:
            nonsti+=1
            if (nonsti == 6):
              iasti = 0

        # -------------------------------------------
        #   Compute data for dense-output function
        # --------------------------------------------
        for k in xrange(n):
          yd0 = y[k]
          ydiff = self.yy1[k] - yd0
          bspl = h * self.kk[0,k] - ydiff
          self.rcont[0,k] = y[k]
          self.rcont[1,k] = ydiff
          self.rcont[2,k] = bspl
          self.rcont[3,k] = -h * self.kk[1,k] + ydiff - bspl

        # -----------------
        #   Advance x
        # ------------------
        # self.k1[:] = self.k2[:]
        self.kk[0,:] = self.kk[1,:]
        y[:] = self.yy1[:]
        self.xold = x
        x = xph
        # -----------------
        #   Output
        # ------------------
        self.hout = h
        self.xout = x
        if self.solout (self.naccpt + 1, self.xold, x, y, n, *pars):
          return 3

        #                         normal exit
        if (last):
          self.hout = hnew
          self.xout = x
          return 1
        #                          Adjust new h before new step
        if (abs (hnew) > hmax):
          hnew = posneg * hmax
        if (reject):
          hnew = posneg * min (abs (hnew), abs (h))
        reject = 0
      else: #                     step rejected
        hnew = h / min (facc1, fac11 / safe)
        reject = 1
        if (self.naccpt >= 1):
          self.naccpt = self.naccpt + 1   #  fprintf(stderr, "-- %ld -- step rejected\n", self.nstep)
        last = 0
      if self.FORCE_STEP:
        h = self.FORCE_STEP
        reject = 0
      else:
        h = hnew
    return 0
    #  ENDS: while

  def forceStepSize(self,h):
    """Force the integrator step size to the given value
       This is ONLY useful for integrator testing. NEVER use this for
       actual integration of an ODE

       Passing h=0 (or any other Python false) will revert to adaptive
       step size behaviour
    """
    self.FORCE_STEP = h

  def dope(self, y0, tEnd, pars=[], dt=None, nmax=100000):
    """Top-level function for integrator:

       integrate the ODE from initial condition y0
       y0[0] is the start time, with end time tEnd

       pars (default []) the parameters to the vector field function
       dt is maximal time step, defaulting to 1/100 of the duration
       nmax in maximal number of steps allowed, default 100000
    """
    if dt is None:
      dt = (tEnd - y0[0])/100
    ODE_DIM = len(y0)-1
    rtol = asarray([self.ODE_RTOL] * len(y0))
    atol = asarray([self.ODE_ATOL] * len(y0))
    self.dopri5(
      ODE_DIM,          #  unsigned n,       dimension of the system <= UINT_MAX-1
      y0[0],            #  double x,         initial x-value
      y0[1:],           #  double* y,        initial values for y
      pars,             #  double *pars,     parameters
      tEnd,             #  double xend,      final x-value (xend-x may be positive or negative)
      rtol,             #  double* rtoler,   relative error tolerance
      atol,             #  double* atoler,   absolute error tolerance
      0,                #  int itoler,       switch for rtoler and atoler: SCALARS
      2,                #  int iout,         switch for calling solout: DENSE
      stdout,       #  FILE* fileout,    messages stream
      2.3E-16,          #  double uround,    rounding unit
      0.9,              #  double safe,      safety factor
      0.2,              #  double fac1,      parameters for step size selection (lower ratio)
      10.0,             #  double fac2,      (upper ratio)
      0.04,             #  double beta,      for stabilized step size control
      dt,               #  double hmax,      maximal step size
      dt/2,             #  double h,         initial step size
      nmax,             #  long nmax,        maximal number of allowed steps
      self.ODE_MAX,     #  double magboud,   bound on magnitude of solution components
      1,                #  int meth,         switch for the choice of the coefficients
      -1,               #  long nstiff,      test for stiffness: NEVER
      ODE_DIM,          #  unsigned nrdens,  number of components for which dense outpout is required
      [],               #  unsigned* icont,  indexes of components for which dense output is required, >= nrdens
      0                 #  unsigned licont   declared length of icont
     )

  def dopri5 (self, n, x, y, pars,
             xend, rtoler, atoler, itoler,
             iout, fileout, uround,
             safe, fac1, fac2, beta, hmax,
             h, nmax, magbound, meth, nstiff,
             nrdens, icont, licont):
    """port of original dopri5 front end"""
    arret = False
    self.nfcn = 0
    self.nstep = 0
    self.naccpt = 0
    self.nrejct = 0

    # Figure out the correct datatype for arrays
    typ = type(x.flat[0]+y.flat[0])

    #  nmax, the maximal number of steps
    if ( not nmax):
      nmax = 100000
    elif (nmax <= 0):
      if (fileout):
        fileout.write( "Wrong input, nmax = %li\r\n" % nmax)
      arret = 1

    #  meth, coefficients of the method
    if ( not meth):
      meth = 1
    elif ((meth <= 0) or (meth >= 2)):
      if (fileout):
        fileout.write( "Curious input, meth = %i\r\n", meth)
      arret = 1

    # nstiff, parameter for stiffness detection
    if ( not nstiff):
      nstiff = 1000
    elif (nstiff < 0):
      nstiff = nmax + 10
    # iout, switch for calling solout
    if ((iout < 0) or (iout > 2)):
      if (fileout):
        fileout.write( "Wrong input, iout = %i\r\n", iout)
      arret = 1

    # nrdens, number of dense output components
    if (nrdens > n):
      if (fileout):
        fileout.write( "Curious input, nrdens = %u\r\n", nrdens)
      arret = 1
    elif (nrdens):
      # is there enough memory to allocate rcont12345&self.indir ?
      self.rcont = zeros((5,nrdens),dtype=typ)
      # control of length of icont
      if (nrdens == n):
        if (icont and fileout):
          fileout.write(
                   "Warning : when nrdens = n there is no need allocating memory for icont\r\n")
        self.nrds = n
      elif (licont < nrdens):
        if (fileout):
          fileout.write(
                   "Insufficient storage for icont, min. licont = %u\r\n"
                   % nrdens)
        arret = 1
      else:
        if ((iout < 2) and fileout):
          fileout.write( "Warning : put iout = 2 for dense output\r\n")
        self.nrds = nrdens

    # uround, smallest number satisfying 1.0+uround > 1.0
    if (uround == 0.0):
      uround = 2.3E-16
    elif ((uround <= 1.0E-35) or (uround >= 1.0)):
      if (fileout):
        fileout.write( "DOPRI 'uround' out-of-range (1e-35,1): %.16e\r\n" % uround)
      arret = 1

    # safety factor
    if (safe == 0.0):
      safe = 0.9
    elif ((safe >= 1.0) or (safe <= 1.0E-4)):
      if (fileout):
        fileout.write( "Curious input for safety factor, safe = %.16e\r\n" % safe)
      arret = 1

    #  fac1, fac2, parameters for step size selection
    if (fac1 == 0.0):
      fac1 = 0.2
    if (fac2 == 0.0):
      fac2 = 10.0

    # beta for step control stabilization
    if (beta == 0.0):
      beta = 0.04
    elif (beta < 0.0):
      beta = 0.0
    elif (beta > 0.2):
      if (fileout):
        fileout.write( "Curious input for beta : beta = %.16e\r\n", beta)
      arret = 1

    # maximal step size
    if (hmax == 0.0):
      hmax = xend - x

    # is there enough free memory for the method ?
    self.yy1 = empty(n,typ)
    self.kk = empty((6,n),typ)
    self.ysti = empty(n,typ)

    # when a failure has occured, we return -1
    if (arret):
      idid = -1
    else:
      idid = self.dopcor (n, x, y, pars, xend, hmax, h, rtoler, atoler, itoler,
                fileout, iout, nmax, magbound, uround, meth, nstiff,
                safe, beta, fac1, fac2, icont)
    self.yy1 = None
    self.ysti = None
    return idid

  # dense output function
  def contd5 (self,x,i=slice(None)):
    """Dense output function:

       Compute the trajectory at "time" x.
       If specified, i is an index of trajectory components to obtain
    """
    theta = (x - self.xold) / self.hout
    theta1 = 1.0 - theta
    coef = vstack([ones_like(theta),theta,theta1,theta,theta1]).cumprod(axis=0)
    return tensordot(coef,self.rcont,(0,0))[newaxis,i]

  def getTraj( self ):
    """Get a trajectory object representing the must recent timestep produced
       from the integrator. Returns a TrajPatch.
    """
    return TrajPatch( self.xold, self.hout, self.rcont.copy() )

  def refine( self, evtFun, *args ):
    """Use TrajPatch.refine the last timestep to a root of evtFun"""
    return self.trj.refine( evtFun,tol=self.ODE_EVTTOL, *args )

class odeDP5( Integro ):
  """Convenience interface to the Dopri5 integrator, allowing it to
    wrap a callable (e.g. lambda function)

    Events:
       You can use event detection to end the integration, by setting the
       .event attribute to an appropriate function.
       .event signature: t,y,trj,p --> bool
         t -- (t0,t1) a tuple, range of times covered in the last time-step
         y -- 2xD state at t0, t1
         trj-- TrajPatch trajectory patch for the time-step
         p -- the pars parameter to the odeDP5 constructor (defaults to [])
       .event is called every time a point is returned, until it returns
       something which converts to a boolean true, stopping integration.
       The ending time will be the t1 of the time-step that caused the event.
       To refine to an "exact" (within ODE_EVTTOL) ending time, use the
         refine() method. Typically, the dominant error in refined events
         will be due to integration and not refinement.
       To disable event detection, set .event to None

    Auxiliary output computation:
       You can add auxiliary outputs to the integration by assigning the .aux
       attribute to an appropriate function.
       .aux signature: t,y,trj,p --> x
       The value of x will be appended to every output point

    Display:
       If you wish to display integration state while the integration
         is in progress, do so from within .event.

  Example of use (from test_event(), below):
    >>> # Create an integrator for the linear "circle" system
    >>> o = odeDP5( lambda t,y,p : [-y[1]*p,y[0]*p], pars=(2.0,) )
    >>> o.event = lambda t,y,trj,p : y[0][1]<0.1 and y[1][1]>0.1
    >>> # Integrate the ODE starting from state [1,0], from time 0 to time 100
    >>> #   by "calling" the ODE object. Returns times and states in arrays.
    >>> t,y = o([1,0],0,100)
    >>> # Refine the endpoint to within ODE_EVTTOL
    >>> tt,yy = o.refine( lambda t,y : y[1]-0.1 )
    >>> # tt will be a pretty good approximation of 2*pi+arcsin(0.1)
  """
  def __init__(self, fun, pars=None):
    Integro.__init__(self)
    assert callable(fun),"function signature must be t,y,pars --> dy"
    self.fun = fun
    if pars is None:
      pars = []
    self.fpars = pars
    self.event = None
    self.aux = None
    self.sample = None
    self.si = None

  def safeOdeFunc(self,n,t,Y,p,dY):
    """PRIVATE: wrapper for vector field function"""
    dY[:] = self.fun(t,Y,self.fpars if p is None else p)

  def solout( self, n, t0, t1, yt, ny,*args ):
    assert False,"Never reached -- solout() is overwritten by __call__"

  def solout_sample( self, n, t0, t1, yt,ny, *args ):
    # Scan for sample points in new interval
    s0 = self.si
    s1 = self.si
    assert self.sample[s0]>=t0,"Next point is in (t0,t1) interval"
    while s1<len(self.sample) and self.sample[s1]<t1:
      s1+=1
    if s1>s0:
      # Collect times
      ts = self.sample[s0:s1]
      # Compute values from dense output
      yts = self.contd5(ts).squeeze()
      if len(yts.shape)==1:
        self.yOut.append(yts)
        self.tOut.append((t0,ts[-1]))
      else:
        for k in xrange(s1-s0):
          self.yOut.append(yts[k,:])
          if k==0:
            self.tOut.append((t0,ts[0]))
          else:
            self.tOut.append((ts[k-1],ts[k]))
      # Update sample index
      self.si = s1
    return self.test_events(*args)

  def solout_auto( self, n, t0, t1, yt,ny,*args ):
    """PRIVATE: process a new point emitted by integrator"""
    # Store in output log
    self.tOut.append((t0,t1))
    if self.aux is None:
      ytc = yt.copy()
    else:
      out = self.aux(t1,yt,self.fpars)
      ytc = concatenate([yt,out])
    self.yOut.append(ytc)
    return self.test_events(*args)

  def test_events(self,*args):
    """PRIVATE: test for events"""
    # Can't test for events before we have first two points
    if len(self.tOut)<2:
      self.trj = self.getTraj()
      return
    # Check for events
    if self.event is not None:
      evt = self.event(
        (self.tOut[-2][1],self.tOut[-1][0]),self.yOut[-2:], self.trj, *args )
      self.trj = self.getTraj()
      return evt
    return

  def refine( self, evtFun, *args ):
    t,y = Integro.refine( self, evtFun, *args )
    self.yOut[-1] = y.copy()
    self.tOut[-1] = (self.xold,t)
    return t,y
  refine.__doc__ = Integro.refine.__doc__

  def __call__(self,y0,t0,t1=None,dt=None, pars=()):
    """Integrate the ODE from initial condition y0 at time t0 to time t1
       Optional parameter dt can set maximal timestep.

       Alternatively, t0 can be a sequence of times for which the user wants
       to sample the resulting trajectory, with t0[-1] the end time.

       returns t,y arrays of sample times and states
    """
    if t1 is None:
      assert len(t0)>1,"Must have an interval"
      assert all(diff(t0)>0), "Sample times increase"
      self.sample = list(t0)
      self.si = 0
      t1 = t0[-1]
      t0 = t0[0]
      self.solout = self.solout_sample
      assert self.aux is None, "Sampling not supported with aux output"
    else:
      self.solout = self.solout_auto
    y0 = concatenate((
      (float(t0),),asarray(y0).flatten()
    ))
    self.tOut = []
    self.yOut = []
    self.trj = None
    self.dope( y0, tEnd = t1, dt=dt, pars=pars )
    self.tOut = array(self.tOut)
    self.yOut = vstack(self.yOut)
    return self.tOut[:,1], self.yOut

if __name__=="__main__":
  from sys import stderr
  from numpy import *
  from pylab import *
  from copy import deepcopy
  print("""Running unit tests:

             (1) Output variables and accuracy test
             (2) Sample test -- chosen times, dense sampling
             (3) Event detector
             (4) Rossler system
             (5) Order test
        """)
  class Rossler( Integro ):
    def __init__(self):
      Integro.__init__(self)
      self.par = None
      self.Y0 = None

    def safeOdeFunc(self,n,t,Y,p,dY):
      dY[0]=-Y[1]-Y[2]
      dY[1] = Y[0] + p[0]*Y[1]
      dY[2] = p[1] + Y[2]*(Y[0] - p[2])

    def solout (self,n,t0,t1,y,*arg):
      row = empty(len(y)+1,dtype=y.dtype)
      row[1:] = y
      row[0] = t1
      self.res.append(row)

    def run(self,tEnd=200,
        pars=array( ( 0.2,  0.2,  5.7 ) ),
        ics = array( ( 0.0, 5, -5, 2 ) ) ):
      self.res=[]
      self.dope( ics, tEnd = tEnd, pars = pars, dt = 0.5 )
      self.res = array(self.res)

      subplot(221)
      plot(self.res[:,1],self.res[:,2])
      xlabel('x')
      ylabel('y')
      subplot(222)
      plot(self.res[:,3],self.res[:,2])
      xlabel('z')
      ylabel('y')
      subplot(223)
      plot(self.res[:,1],self.res[:,3])
      xlabel('x')
      ylabel('z')
      # Compute a "3D" view
      vwSkw = mat(((0,1,-1),(-1,0,1),(1,-1,0)))
      u,s,v = svd(vwSkw)
      vw = dot(u,dot(diag(exp(s/2)),v)).real
      vu = dot(self.res[:,1:],vw[:,:2])
      subplot(224)
      plot(vu[:,0],vu[:,1])

  def test_rossler():
    """Run a Rossler system at a "pretty" set of parameters"""
    r = Rossler()
    clf()
    r.run()
    r.run(pars=array((0.2,0.1,5.7)))
    r.run(pars=array((0.2,0.3,5.7)))

  def test_sample():
    """Run an integrator on linear circle system (solutions cos,sin)
       and use sample mode to compute at chosen times, approximately
       x8 more dense than dt

       NOTE: requires matplotlib to display
    """
    N = 200
    o = odeDP5( lambda t,y,p : [-y[1],y[0]] )
    t0,y0 = o([1,0],0,4*pi,dt=4*pi/N)
    t1,y1 = o([1,0],linspace(0,4*pi,N*8),dt=4*pi/N)
    plot(t1,y1-c_[cos(t1),sin(t1)],'.')
    plot(t0,y0-c_[cos(t0),sin(t0)],'.-',linewidth=2)
    xlabel('time')
    ylabel('error')
    legend(['cos x8','sin x8','cos','sin'])
    title('Sample mode error, at x8 samples')

  def test_event():
    """Run an integrator on linear circle system (solutions cos,sin)
       detect and refine event of sin zero crossing and verify tolerance

       NOTE: requires matplotlib to display
    """
    o = odeDP5( lambda t,y,p : [-y[1],y[0]] )
    o.event = lambda t,y,p: (y[0][1]<-0.1) and (y[1][1]>-0.1)
    t,y = o([1,0],0,100)
    # Need to copy results because refine() adjusts the last point
    t=t.copy()
    y=y.copy()
    # Refine the endpoint to within ODE_EVTTOL
    tt,yy = o.refine( lambda t,y : y[1]+0.1 )

    tr = linspace(t[-2],t[-1],16)
    yr = o.getTraj()(tr).squeeze()
    def exact(t):
      return column_stack((cos(t),sin(t)))
    clf()
    plot(t,y-exact(t),'.-')
    plot(tr,yr-exact(tr),'-k')
    plot([tt,tt],(yy-exact([tt])).squeeze(),'+',markeredgewidth=2,markersize=10)
    legend(['cos','sin'])
    title("Event error %.2e" % (tt-2*pi-arcsin(-0.1)))

  def test_aux():
    """Run an integrator on linear circle system (solutions cos,sin)
       and compute an output. Based on trig we know that the outputs
       should be 1 and sin(2t). Plots the residual relative to ground truth

       NOTE: requires matplotlib to display
    """
    o = odeDP5( lambda t,y,p : [-y[1],y[0]] )
    o.aux = lambda t,y,p: [y[0]*y[0]+y[1]*y[1],2*y[0]*y[1]]
    t,y = o([1,0],0,100)
    r = c_[cos(t),sin(t),ones_like(t),sin(2*t)]
    semilogy(t,abs(y-r),'.-')
    legend(['cos(t)','sin(t)','1','sin(2t)'],loc=0)
    title("Output and accuracy test")

  def test_order():
    """Run an integrator order test and plot the results
       As of this writing, I get 6.2 which is reasonable for this scheme
       If the value drops from 6, something is broken in the numerical code

       NOTE: requires matplotlib to display
    """
    # define a function that does a single test
    def orbital_test( h ):
      """inner function -- run a single test for step size h"""
      # define the flow function
      def flow( t, v, *argv ):
        """flow used for test"""
        x,y,dx,dy = v
        r = x**2+y**2
        return array([dx,dy,-x/r,-y/r])
      # Create the integrator
      ode = odeDP5(flow)
      ode.forceStepSize( h )
      # Integrate over a "cycle"
      t,xn = ode( array([1,0,0,1.0]),0,2*3.14159265 )
      # Compute "ground truth values"
      x0 = array([cos(t),sin(t),-sin(t),cos(t)]).T
      # Return error and time
      return xn-x0,t

    # Loop over a range of step sizes
    h = logspace(0.5,-2.5,40)
    r = []
    for hi in h:
      stderr.write("Step is %-8.3e " % hi)
      stderr.flush()
      er,t = orbital_test( hi )
      r.append( sqrt(mean(sum(er*er.conj(),1))) )
      stderr.write(" --> %8.3e\n" % r[-1])
    # Range of results to fit regression line to
    rng = slice(7,-10)
    # Regress log-log values to obtain order
    p = polyfit(log(h[rng]),log(r[rng]),1)
    # Visualize results
    clf()
    loglog(1/h,exp(polyval(p,log(h))),'b-',linewidth=3)
    loglog(1/h[rng],exp(polyval(p,log(h[rng]))),'g',linewidth=10,color=(0,1,0))
    loglog(1/h,r,'o-r')
    grid('on')
    xlabel('1/step')
    ylabel('error RMS')
    title('Order test: %.2g' % p[0])
    return p[0]
  # Run tests
  figure()
  test_aux()
  figure()
  test_sample()
  figure()
  test_event()
  figure()
  test_rossler()
  figure()
  test_order()
  show()
