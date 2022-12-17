"""
This file contains unit tests which require matplotlib to run. They are activated
if you run this file as a script. As a module, it defines a single function.

The jacobian_cdas function provides a tool for automatically computing Jacobians
of numpy array valued functions of numpy arrays. It implements an auto-scaling
mechanism to ensure that the central differences are scaled such that the
behaviour of the function is locally linear.

(c) 2008,...,2020 Shai Revzen

Use of this code is governed by the GNU General Public License, version 3
"""

def jacobian_cdas( func, scl, lint=0.9, tol=1e-12, eps = 1e-30, withScl = False ):
  """Compute Jacobian of a function based on auto-scaled central differences.
  NOTE: 

  INPUTS:
    func -- callable -- M:=(m1,m2,...) valued function of a N:=(n1,n2,...) input
    scl -- float or N -- initial scales central differences
    lint -- float -- linearity threshold, in range 0 to 1. 0 disables
         auto-scaling; 1 requires completely linear behavior from func
    tol -- float -- minimal step allowed
    eps -- float -- infinitesimal; must be much smaller than smallest change in
         func over a change of tol in the domain.
    withScl -- bool -- return scales together with Jacobian

  OUTPUTS: jacobian function
    jFun: x --> J (for withScale=False)
    jFun: x --> J,s (for withScale=True)

    x -- N -- input point
    J -- M x N -- Jacobian of func at x
    s -- N -- scales at which Jacobian holds around x
  """
  from numpy import (
      empty, empty_like, newaxis, diag, arange, sum, all, abs, any
  )
  def forceArray(x):
    """
    Force x to be an array; if scalar becomes array of length 1
    """
    x = asarray(x)
    if x.ndim == 0:
      return (1,),x.reshape((1,))
    return x.shape,x.flatten()
  _,scl = forceArray(scl)
  assert all(scl>0)
  lint = abs(lint)  
  def centDiffJacAutoScl( arg ):
    """
    Algorithm: use the value of the function at the center point
      to test linearity of the function. Linearity is tested by
      taking dy+ and dy- for each dx, and ensuring that they
      satisfy lint<|dy+|/|dy-|<1/lint
    """
    # Flatten inputs and outputs
    xsz,x0 = forceArray(arg)
    ysz,y0 = forceArray(func(arg))
    # Construct initial step sizes
    if scl.shape == xsz:
      s = scl.copy()
    else:
      s = empty(x0.shape,dtype=float)
      s[:] = scl
    # Initially, step along all coordinates
    idx = arange(x0.size,dtype=int)
    # Matrix from inputs to outputs
    dyp = empty((y0.size,x0.size),(x0[0]+y0[0]).dtype)
    dyn = empty_like(dyp)
    while True:
      assert not any(s==0)
      # Construct coordinate steps
      d0 = diag(s)
      # Apply in directions that are in idx
      for k in idx:
        dx = d0[k,:]
        dyp[:,k] = func((x0+dx).reshape(xsz)).flatten()-y0
        dyn[:,k] = func((x0-dx).reshape(xsz)).flatten()-y0 
      # Compute magnitude 
      adyp = abs(dyp)
      adyn = abs(dyn)
      dp = sum(adyp*adyp,axis=0) 
      dn = sum(adyn*adyn,axis=0)
      eps2 = eps*eps
      nz = (dp>eps2) | (dn>eps2) # directions where partial is zero
      nul = (dp<eps2) | (dn<eps2)
      if any(nul & nz): # Couldn't see enough change --> increase step
        s[nul & nz] *= 1.5
        continue
      rat = dp/(dn+eps)
      nl = ((rat<lint) | (rat>(1.0/lint)))
      # If no linearity violations found --> done
      if ~any(nl & nz):
        break
      # otherwise -- decrease steps in nonlinear directions
      idx, = nl.flatten().nonzero()
      s[idx] *= 0.75
      # Don't allow steps smaller than tol
      s[idx[s[idx]<tol]] = tol
      if all(s[idx] == tol):
        break
      # <--- loop back
    res = (dyp-dyn)/(2*s[newaxis,:])
    res = res.reshape(ysz+xsz)
    if withScl:
      return res, s.reshape(xsz)
    return res
  return centDiffJacAutoScl

if __name__=="__main__":
  print("Running unit tests")
  
  from numpy import sqrt, newaxis, asarray, linspace, sin, cos, log, sum, abs
  from pylab import figure, colorbar, imshow, show, close
  
  
  def test_scalar(f,Jf,cap=None):
    t = 2**linspace(-30,0,512)
    nj1 = jacobian_cdas(f,[0.001],tol=1e-20)
    nj1s = jacobian_cdas(f,[0.001],tol=1e-20,withScl=True)
    nj1f = jacobian_cdas(f,[0.001],tol=0.001)
    nj2 = jacobian_cdas(f,[1e-5],tol=1e-5)
    fig=figure(); fig.clf()
    ax = fig.add_subplot(211); ax.semilogx(t,[ f(ti) for ti in t ])
    if cap:
      ax.set_title(cap)
    # Jacobians
    ax = fig.add_subplot(212); 
    gt = asarray([ Jf(ti) for ti in t ]).squeeze()
    j1 = asarray([ nj1(ti).squeeze() for ti in t ])
    j1f = asarray([ nj1f(ti).squeeze() for ti in t ])
    s1 = asarray([ nj1s(ti)[1] for ti in t ])
    j2 = asarray([ nj2(ti).squeeze() for ti in t ])
    ax.loglog(t,abs(j1f-gt)/(1e-10+abs(gt))
        ,'.-',label='$\\varepsilon$ for $\\delta=10^{-3}$')
    ax.loglog(t,abs(j1-gt)/(1e-10+abs(gt))
        ,'.-',label='$\\varepsilon$ adaptive $\\delta=10^{-3}$')
    ax.loglog(t,abs(j2-gt)/(1e-10+abs(gt))
        ,'.-',label='$\\varepsilon$ for $\\delta=10^{-5}$')
    ax.loglog(t,s1,'-r',lw=2,alpha=0.6,label='auto $\\delta$')
    ax.hlines(0.001,t[0],t[-1],'m')
    ax.set_ylabel("Relative error $\\varepsilon$")
    ax.legend()
  
  def test_2D(f,Jf,cap='Function value'):
    t = linspace(-.5,.5,128)#r_[-(2**linspace(-20,0,32))[::-1],(2**linspace(-20,0,32))]
    nj = jacobian_cdas(f,1e-2,tol=1e-20,withScl=True)
    fig = figure(); fig.clf()
    vf = asarray([ [f([xi,yi]) for xi in t] for yi in t])
    gt = asarray([ [Jf([xi,yi]) for xi in t] for yi in t])
    lngt = log(1e-20+sum(gt*gt,-1))
    dat = [ [nj([xi,yi]) for xi in t] for yi in t]
    ej = asarray([[d[0] for d in row] for row in dat]).squeeze() - gt
    #ej2 = asarray([ [nj2([xi,yi])[0] for xi in t] for yi in t]).squeeze() - gt
    sc = log(asarray([[d[1].min() for d in row] for row in dat]))/log(10)
    ler = (log(sum(ej*ej,-1))-lngt)/(2*log(10))
    #ler2 = (log(sum(ej2*ej2,-1))-lngt)/(2*log(10))
    ax=fig.add_subplot(131); imshow(vf,extent=[t[0],t[-1],t[0],t[-1]]); colorbar()
    ax.axis('equal'); ax.set_title(cap)
    ax=fig.add_subplot(132); imshow(ler,extent=[t[0],t[-1],t[0],t[-1]]); colorbar()
    ax.axis('equal'); ax.set_title('$log_{10}(|\\varepsilon|/|f|)$ w/adaptive $\delta=10^{-2}$')
    ax=fig.add_subplot(133); imshow(sc,extent=[t[0],t[-1],t[0],t[-1]]); colorbar()
    ax.axis('equal'); ax.set_title('$log_{10}(\\delta)$ adaptive step')
    
  def test_rank3():
    """
    testing jacobian indexing by showing that it reproduces the rank 3
    transpose operation. 
    """
    from numpy import allclose, einsum, zeros, arange, all
    from numpy.random import rand
    # The transpose operation
    def fT(x):
      return x.T
    def JfT(x):
      # This gives the index permutation that produces a transpose
      idx = arange(x.size,dtype=int).reshape(x.shape).T.flatten()
      # Here is the proof
      assert all(x.flatten()[idx].reshape(x.T.shape) == x.T)
      # Build permutation matrix from this permutation
      P = zeros((x.size,x.size),int)
      P[range(idx.size),idx]=1
      return P.reshape(x.T.shape+x.shape)
    
    x = rand(2,3,4)
    # Since the transpose is linear, it should equal its jacobian
    assert all(einsum("ijkabc,abc->ijk",JfT(x),x)==x.T)
    nj = jacobian_cdas(fT,0.05)
    assert allclose(nj(x)-JfT(x),0)
  
  def f2d(x):
    return sqrt(abs(sin(x[0]*13)))+sqrt(abs(sin(x[1]*13)))
  def Jf2d(x):
    from numpy import sign
    x = asarray(x)
    s = sin(x*13); c = cos(x*13)
    return 13 * sign(s)*c/(2*sqrt(abs(s)))[newaxis,:]
  
  def f1(x):
    from numpy import asarray, sum
    x = asarray(x)
    return sum(x*x)**0.2
  def Jf1(x):
    from numpy import asarray, sum
    x = asarray(x)
    return f1(x)*0.4*x/sum(x*x)
  
  def f2(x):
    x = asarray(x)
    return sum(sin(log(x*x)))
  def Jf2(x):
    x = asarray(x)
    return 2*cos(log(x*x))/x
  
  print("Starting tests...")
  close('all')
  print("\tValidate multi-indexing (high rank operations)")
  test_rank3()
  print("\tScalar fractional power")
  test_scalar(f1,Jf1,"$|x|^{7/5}/x$") 
  print("\tScalar 'topologist`s sin function'")
  test_scalar(f2,Jf2,"$\sum_i \sin(\log(x_i^2))$") 
  print("\t2D fractional power")
  test_2D(f1,Jf1)  
  print("\t2D 'topologist`s sin function'")  
  test_2D(f2,Jf2)  
  print("...done")
  show()
  
  
  
  
  
  
  
  
  