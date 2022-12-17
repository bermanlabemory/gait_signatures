"""
  This software is copyrighted material (C) Shai Revzen 2011-2020
  
  It is provided under the GNU Public License version 3.0,
  as found at http://www.gnu.org/licenses/gpl-3.0-standalone.html

  OVERVIEW:

  Implements a fixed step SDE integrator - the fixed step version of the
  R3 scheme from doi:10.1016/j.cam.2006.08.012 (Bastani et.al. 2006),
  which is itself taken from:
    P.M. Burrage "Runge-Kutta methods for stochastic differential equations"
    Ph.D. Thesis, The University of Queensland, Australia, 1999

  This integrates an SDE of the Stratonovich form:
    dX(t) = f (X(t)) dt + g(X(t)) o dW (t)
    X(0) = X0

  If you have matplotlib installed, running this file should generate
  plots of key unit tests.

  MAIN CLASSES:

    SDE -- typically the only thing you will use, integrates the SDE
"""

__all__=['SDE']

from numpy import (
  # array handling
  asarray, zeros, zeros_like, empty, newaxis, floating, 
  # array operations
  dot, linspace, mean, arange, any,
  # core math
  pi, nan, sin, cos, log, sqrt, exp, sinh, isnan,
  )

class SDE( object ):
  """Concrete class SDE
     
     Implements a fixed step SDE integrator - the fixed step version of the
     R3 scheme from doi:10.1016/j.cam.2006.08.012 (Bastani et.al. 2006),
     which is itself taken from:
        P.M. Burrage "Runge-Kutta methods for stochastic differential equations"
        Ph.D. Thesis, The University of Queensland, Australia, 1999

     This integrates an SDE of the Stratonovich form:
        dX(t) = f (X(t)) dt + g(X(t)) o dW (t)
        X(0) = X0
     
     USAGE:
     >>> sde = SDE( diffusion_function, drift_function, noise_dim ) 
     >>> t,y,w = sde( x0, t0, t1, dt )
       to integrate from t0 to t1 in steps of dt, with initial condition x0
     >>> sde.integrateAt( t, x0, X, W )
       to integrate and give trajectory values at times t, starting with 
       initial condition x0, and using X and W to store the trajectory data.

     See the test_XXX functions in this module for usage examples.
  """
  # Butcher array for explicit R3 scheme
  TAB = asarray([
      [0,    0,    0,     0],
      [0.5,  0.5,  0,     0],
      [0.75, 0,    0.75,  0],
      [nan,  2/9.0,1/3.0, 4/9.0]
  ])
  
  def __init__(self, d, s, sdim, dTab=TAB, sTab=TAB ):
    """
    INPUTS:
      d -- callable -- diffusion (ODE) part of the equation, mapping x --> dot x
      s -- callable -- drift (SDE) part of the equation, mapping x,dW --> dot x
        NOTE: this mapping *MUST* be linear in dW for the integration to be
              valid. It is in functional form to allow for code optimization.
      sdim -- dimension of dW
      dTab, sTab -- Butcher tables of the integration scheme.
        *WARNING* don't change these unless you really, REALLY, know what 
        it means. If you do -- make sure to run the tests and validate the
        convergent order.  
    """
    # store Butcher arrays
    self.bd = dTab
    self.bs = sTab
    assert callable(d),"diffusion flow is a callable"
    self.d = d
    assert callable(s),"stochastic (drift) flow is a callable"
    self.s = s
    assert int(sdim) >= 0, "dimension of noise >=0"
    self.sdim = int(sdim)
 
  def dW( self, t0, t1 ):
    """
    Generate stochastic step for time interval t0..t1.
    
    Subclasses may override this term if something other than a 
    Wiener process is needed.
    """
    return randn(self.sdim) * sqrt(t1-t0)
  
  def __call__(self,x0,t0,t1,dt=None):
    """ Integrate the SDE
    INPUTS:
      x0 -- D -- initial condition
      t0,t1 -- float -- start and end times
      dt -- float (optional) -- time step; default is (t1-t0)/100
      
    OUTPUTS:
      t -- N -- time points
      y -- N x D -- trajectory at times t
      w -- N x self.sdim -- random walk values at times t
    """
    if dt is None:
      dt = float(t1-t0)/100
    return self.integrateAt( arange(t0,t1,dt), x0 )
  
  def bisect( self, t0,t1,x0,x1,w0,w1, dtype=floating ):
    """Bisect the trajectory segment between times t0 and t1
    INPUTS:
      t0, t1 -- float -- times, t1>t0
      x0, x1 -- D -- trajectory points at t0, t1
      w0, w1 -- sdim -- random walk values at t0, t1
    OUTPUT:
      t -- 3 -- t0,(t0+t1)/2,t1 
      x -- 3 x D -- trajectory with mid-point
      w -- 3 x sdim -- random walk with mid-point 
    """
    assert t1>t0
    x0 = asarray(x0,dtype=dtype).squeeze()
    x1 = asarray(x1,dtype=dtype).squeeze()
    assert x0.shape==x1.shape
    w0 = asarray(w0,dtype=dtype).flatten()
    w1 = asarray(w1,dtype=dtype).flatten()
    assert w0.size==self.sdim
    assert w1.size==self.sdim
    # Define Brownian bridge function for the interval
    dW = w1-w0
    dT = t1-t0
    def bridge(_t0,_t1):
      assert _t0==t0 and _t1==(t1+t0)/2.0
      return dW/2.0 + randn(*w0.shape) * sqrt(dT/2.0)
    return self.integrateAt([t0,(t0+t1)/2.0,t1],x0)
    
  def refine( self, secfun, t0,t1,x0,x1,w0,w1, args=(), retAll = False, fTol = 1e-4, xTol=1e-8, tTol=1e-6, wTol = 1e-6, dtype=floating ):
    """Refine trajectory to a positive zero crossing of secfun
    INPUTS:
      t0,t1,x0,x1,w0,w1 -- as in self.bisect()
      secfun -- callable -- secfun(t,x,w,*args) is a function that crosses from
        negative to positive on the trajectory in the time interval t0..t1.
      args -- tuple -- extra arguments for secfun
      fTol -- tolerance for secfun values
      xTol -- tolerance for trajectory coordinates (sup norm)
      tTol -- tolerance for time values
      wTol -- tolerance for random walk step (sub norm)
      retAll -- return all points sampled
    OUTPUT: 
    (retAll false) t,x,w,y
      t -- float -- crossing time
      x -- D -- crossing point
      w -- sdim -- random walk term
      y -- float -- secfun value at crossing
    (retAll true) unsorted list of points with format t,x,w,y
    
    Uses repeated bisection to find a crossing point for a section function. 
    """
    y0 = secfun( t0, x0, w0, *args )
    y1 = secfun( t1, x1, w1, *args )
    # Check if we just got lucky...
    if abs(y0)<fTol:
      if abs(y1)<fTol:
        return ((t1+t0)/2, (x1+x0)/2, (w1+w0)/2, (y1+y0)/2)
      return (t0,x0,w0,y0)
    elif abs(y1)<fTol:
      return (t1,x1,w1,y1)
    if y0>0 or y1<=0 and abs(y1-y0)>fTol:
      raise ValueError("Section function values did not cross 0; were %g and %g" % (y0,y1)) 
    traj = [(t0,x0,w0,y0),(t1,x1,w1,y1)]
    while (abs(y1-y0)>fTol
          or all(abs(x1-x0)>xTol) 
          or all(abs(w1-w0)>wTol)) and (abs(t1-t0)>tTol):
      t,x,w = self.bisect( t0,t1,x0,x1,w0,w1 )
      y = secfun( t[1], x[1,:], w[1,:], *args )
      traj.append( (t[1], x[1,:], w[1,:], y) )
      #!!!print t0,y0,'--',t1,y1
      if y>0:
        t1,x1,w1,y1 = traj[-1]
      else:
        t0,x0,w0,y0 = traj[-1]
    if abs(t1-t0)<tTol:
      traj.append( ((t1+t0)/2, (x1+x0)/2, (w1+w0)/2, (y1+y0)/2) )
    if retAll:
      return traj
    return traj[-1]
    
  def integrateAt( self, t, x0, X=None, W=None, dW_fun=None, dtype=floating ):
    """ Integrate the SDE
    INPUTS:
      t -- N -- monotone increasing array with times at which to compute
         the integral
      x0 -- D -- initial condition
      X -- N x D (or None) -- storage for trajectory
      W -- N x self.sdim (or None) -- storage for random walk term
      dW_fun -- callable -- dW(t0,t1) gives the random walk step for times
        t0 to t1. For a Wiener process, this is randn()*sqrt(t1-t0).
    OUTPUTS: t,X,W
      if either X or W were None, a newly allocated array is returned with
      the data.
    """    
    x0 = asarray(x0).flatten()
    dim = len(x0)
    stp = self.bd.shape[0]-1
    assert dim == len(self.s(x0,zeros(self.sdim))), "Stochastic flow output must have same dimension as system"
    if X is None:
      X = empty( (len(t),dim), dtype=dtype )
    else:
      assert X.shape == (len(t),dim), "Has space for result"
    if W is None:
      W = empty((len(t),self.sdim), dtype=dtype)
    else:
      assert W.shape == (len(t),self.sdim), "Has space for noise"
    if dW_fun is None:
      dW_fun = self.dW  
    # Storage for step computations
    Xd = zeros( (stp+1,dim), dtype=dtype )
    Xs = zeros( (stp+1,dim), dtype=dtype )
    Yd = zeros( (stp,dim), dtype=dtype )
    Ys = zeros( (stp,dim), dtype=dtype )
    # Loop over all times
    t0 = t[0]
    X[0,:] = x0
    W[0,:] = 0
    for ti in range(1,len(t)):
      # Step forward in time
      t1 = t[ti]
      dt = t1-t0
      dW = dW_fun( t0, t1 )
      if any(isnan(dW)):
        print("%s(%g,%g) has NaN-s" % (repr(dW_fun),t0,t1))
        ##>>> good place for a breakpoint 
      # Init arrays for this step
      Xd[0,:] = 0
      Xs[0,:] = 0
      Yd[0,:] = self.d( x0 )
      Ys[0,:] = self.s( x0, dW )
      for k in range(1,stp):
        # Sums for Butcher row k
        Xd[k,:] = dot( self.bd[k,1:k+1], Yd[:k,:] )
        Xs[k,:] = dot( self.bs[k,1:k+1], Ys[:k,:] )
        # Evaluate at integration point
        xk = x0 + dt * Xd[k,:] + Xs[k,:]
        Yd[k,:] = self.d( xk ) 
        Ys[k,:] = self.s( xk, dW )
      # Compute next time step
      x1 = ( x0 
        + dt * dot(self.bd[-1,1:],Yd)
        + dot(self.bs[-1,1:],Ys)   
      )
      # Store results
      X[ti,:] = x1
      W[ti,:] = W[ti-1] + dW
      if any(isnan(x1)):
        print("integration step returned NaN-s")
        ##>>> good place for a breakpoint 
      x0 = x1
      t0 = t1
    # Return results
    return t,X,W

if __name__=="__main__": # Beginning of unit test code
  from pylab import (
    # plotting
    figure, subplot, semilogy, axis, title, ylabel, loglog, show,
    # math util functions
    nansum, randn, diff
  )
  import sys
  def test_diffusion_order():
    """
    Test the convergence order of the continuous integrator using
    the "circle" system
    """
    print("test_diffusion_order")
    sde = SDE( 
      lambda x: dot([[0,-1],[1,0]],x),
      lambda x,dw: zeros_like(x),
      2
      )
    ee = []
    for dt in 10**-linspace(0.5,4.5,11):
      print("dt=%6g"%dt)
      t,y,w = sde( [0,1],0,0.2*pi, dt )
      err = sqrt((y[-1,0]+sin(t[-1]))**2+(y[-1,1]-cos(t[-1]))**2)
      ee.append([dt,err])
    ee = asarray(ee)
    loglog(ee[:,0],ee[:,1],'o-')
    lee = log(ee)
    lee = lee - mean(lee,axis=0)[newaxis,:]
    lee = diff(lee,axis=0)
    title("Order %.2f" % mean(lee[:,1] / lee[:,0])) 

  def test_vs_sol( s, d, sol, beta, x0, N=5, rng = (0,10,0.001)):
    """
    Utility function for testing eqn. dX = s(x) dt + d(x,dW)
    INPUTS:
      s,d -- SDE specification
      beta -- parameter values to test; SDE-s are integrated in parallel, with
        on column for each parameter value (thus sdim = len(beta))
      sol -- function t,w --> x giving the exact solution for the integral
      N -- number of repetitions to run
      rng -- time range and sample points
    OUTPUTS:
      Y -- N x len(arange(*rng)) x len(beta) -- integrated outputs
      X -- Y.shape -- exact solutions given by sol
      
    Also plots the result in the current figure as subplots. Each row gives 
    a value of beta. The left column has the simulations; the right has the
    error w.r.t. the exact solution
    """
    def nanmean(x,axis=-1):
      return nansum(x,axis=axis)/nansum(~isnan(x),axis=axis)
    def nanstd(x,axis=-1):
      return sqrt(nanmean(x**2,axis)-nanmean(x,axis)**2)    
    beta = asarray(beta)
    sde = SDE(s,d,len(beta) )
    t = arange(*rng)
    w = None
    Y = []
    X = []
    sys.stdout.write("%d iterations:" % N)
    sys.stdout.flush()
    for k in range(N):
      t,y,w = sde.integrateAt( t, x0, W = w )
      Y.append(y)
      X.append( sol(t,w) )
      sys.stdout.write(" %d" % k)
      sys.stdout.flush()
    sys.stdout.write("\n")
    Y = asarray(Y)
    X = asarray(X)
    err = Y-X
    m = nanmean(err,0)
    s = nanstd(err,0)
    idx = linspace(0,len(t)-1,500).astype(int)
    tg = t[idx]
    for k in range(len(beta)):
      #
      #
      subplot(len(beta),2,2*k+1)
      cvg = abs(Y[:,idx,k]-Y[:,[-1],k])+1e-18
      semilogy(tg,cvg.T,alpha=0.3)
      axis([0,10,1e-18,10])
      ylabel('beta = %.2e' % beta[k])
      if k==0:
        title('Simulations')
      #
      #
      subplot(len(beta),2,2*k+2)
      semilogy(tg,abs(err[:,idx,k].T),alpha=0.3)
      semilogy(tg,abs(m[idx,k]),'k',lw=3)
      semilogy(tg,abs(m[idx,k]+s[idx,k]),'--k')
      semilogy(tg,abs(m[idx,k]-s[idx,k]),'--k')
      axis([0,10,1e-18,10])
      if k==0:
        title('Errors')
    return Y,X

  def test_bastani_6_4( beta = (0.1,1)):
    "test from Bastani, et.al. 2006 eqn. 6.4, 6.5"
    print("dX = -(1-x**2)dt + beta*(1-x**2)dW")
    beta = asarray(beta)
    def flow(x):
      return -(1-x**2) 
    def drift(x,dw):
      return dw*beta*(1-x**2)
    def exact(t,w):
      q = exp(-2*t[:,newaxis] + 2*beta*w)
      return (q-1)/(q+1)
    return test_vs_sol( flow, drift, exact, beta, (0,0) )
    
  def test_bastani_6_6( beta = (0.1,1)):
    "test from Bastani, et.al. 2006 eqn. 6.6, 6.7"
    print("dX = -x dt + beta*x dW")
    beta = asarray(beta)
    alpha = -1
    def flow(x):
      return alpha*x 
    def drift(x,dw):
      return dw*beta*x
    def exact(t,w):
      return exp(alpha*t[:,newaxis]+beta*w)
    return test_vs_sol( flow, drift, exact, beta, (1,1) )
  
  def test_bastani_6_8():
    "test from Bastani, et.al. 2006 eqn. 6.8, 6.9"
    print("dX = sqrt(1+x**2) dt + sqrt(1+x**2) dW")
    def flow(x):
      return sqrt(1+x**2) 
    def drift(x,dw):
      return dw*sqrt(1+x**2)
    def exact(t,w):
      return sinh(t+w)
    return test_vs_sol( flow, drift, exact, (0,), (0,), rng = (0,2,0.0005))
     
  def test_bastani_6_10( beta = (0.1,1), N=5):
    "test from Bastani, et.al. 2006 eqn. 6.10, 6.11"
    print("dX = -beta*sqrt(1-x.clip(-1,1)**2) dW")
    beta = asarray(beta)
    def flow(x):
      return zeros_like(x) 
    def drift(x,dw):
      return -beta*sqrt(1-x.clip(-1,1)**2)*dw
    def exact(t,w):
      return cos(beta*w+pi/2)
    return test_vs_sol( flow, drift, exact, beta, zeros_like(beta), N=N )

  # Actually run the tests    
  figure(1)
  test_diffusion_order()
  figure(4)    
  test_bastani_6_4()
  figure(6)
  test_bastani_6_6()
  figure(8)
  test_bastani_6_8()
  figure(10)
  test_bastani_6_10()
  show()

