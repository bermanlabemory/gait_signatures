"""
A general purpose FourierSeries class

TODO: more documentation

(c) 2008,...,2020 Shai Revzen and Simon Wilshin

Use of this code is governed by the GNU General Public License, version 3
"""

__all__=['FourierSeries']

from numpy import ( 
    pi, argsort, mod, newaxis, dot, exp, sum, asarray, hstack, allclose, 
    reshape, iscomplex, concatenate, c_, zeros, arange, empty, empty_like,
    diff, conj, ones
    )
from copy import deepcopy

class FourierSeries(object):
  def take(self,cols):
    """Get a FourierSeries that only has the selected columns"""
    other = self.copy()
    other.coef = other.coef[:,cols]
    other.m = other.m[cols]
    return other

  def val( self, phi ):
    """Evaluate fourier series at all the phases phi

       Returns rows corresponding to phi.flatten()
    """
    phi = asarray(phi).flatten()
    phi.shape = (len(phi.flat),1)
    th = phi * self.om
    return dot(exp(1j*th),self.coef)+self.m

  def integrate( self, z0=0 ):
    """Integrate fourier series, and set the mean values to z0
    """
    self.m[:] = asarray(z0)
    self.coef = -1j * self.coef / self.om.T
    return self

  def getDim( self ):
    """Get dimension of output"""
    return len(self.m)

  def getOrder( self ):
    """
    Get order of Fourier series

    If series is sparse, returns frequencies
    """
    if self.order:
      return self.coef.shape[0]/2
    return self.om.flatten()

  def extend( self, other ):
    """Extend a fourier series with additional output columns from
       another fourier series of the same order

       If fs1 is order k with c1 output colums, and
       fs2 is order k with c2 output columns then the following holds:

       fs3 = fs1.copy().extend(fs2)
       assert allclose( fs3.val(x)[:c1], fs1.val(x) )
       assert allclose( fs3.val(x)[c1:], fs2.val(x) )
    """
    assert allclose(other.om,self.om), "Must have same modes"
    self.m = hstack((self.m,other.m))
    self.coef = hstack((self.coef,other.coef))

  def diff( self ):
    """Differentiate the fourier series"""
    self.m[:] = 0
    self.coef = 1j * self.coef * self.om.T
    return self

  def copy( self ):
    """Return copy of the current fourier series"""
    return deepcopy( self )

  def fit( self, order, ph, data ):
    """Fit a fourier series to data at phases phi
       INPUT:
         order -- int   -- order of Fourier fit
               -- k     -- array of ints: uses a sparse fit to only those orders
         ph    -- n     -- phases of samples in data; internally wrapped at 2*pi
                           so input phases may be reals.
         data  -- d x n -- data to fit
       OUTPUT:
         overwrites internal state to model data with 2k complex
         frequency components. Fit is done using an Euler integration of the
         Fourier integrals. Output series is d dimensional.

         NOTE: if d == 1, data still needs to be a rank 2 array
    """
    ph = asarray(ph)
    try:
       phi = reshape( mod(ph + pi,2*pi) - pi, (1,len(ph.flat)) )
    except TypeError:
       if iscomplex(ph.flat[0]):
           raise TypeError("Phase must be real valued.")
       raise
    if phi.shape[1] != data.shape[1]:
      raise IndexError(
        "There are %d phase values for %d data points"
            % (phi.shape[1],data.shape[1]))
    # Sort data by phase
    idx = argsort(phi).flatten()
    dat = c_[data.take(idx,axis=-1),data[:,idx[0]]]
    phi = concatenate( (phi.take(idx),[phi.flat[idx[0]]+2*pi]) )

    # Compute means values and subtract them
    #self.m = mean(dat,1).T
    # mean needs to be computed by trap integration also
    dphi = diff(phi)
    self.m = sum((dat[:,:-1] + dat[:,1:]) * .5 * dphi[newaxis,:], axis = 1) / (max(phi) - min(phi))
    #PDB.set_trace()
    dat = (dat.T - self.m).T
    # Check for explicit frequencies
    if type(order) not in [int,float]:
      o = asarray(order).flatten()
      self.om = zeros((1,2*o.size))
      self.om[0,::2]=o
      self.om[0,1::2]=-o
      self.order = None
    else:
      # Allow 0th order (mean value) models
      order = max(0,int(order))
      self.order = order
      if order<1:
        order = 0
        self.coef = None
        return
      # Compute frequency vector
      om = zeros( 2*order )
      om[::2] = arange(1,order+1)
      om[1::2] = -om[::2]
      self.om = reshape(om,(1,order*2))
    # Compute measure for integral
    #if any(dphi<=0):
      #raise UserWarning,"Duplicate phase values in data"
    # Apply trapezoidal rule for data points (and add 2 pi factor needed later)
    zd = (dat[:,1:]+dat[:,:-1])/(2.0*2*pi) * dphi
    # Compute phase values for integrals
    th = self.om.T * (phi[1:]-dphi/2)
    # Coefficients are integrals
    self.coef = dot(exp(-1j*th),zd.T)
    return self

  def fromAlien( self, other ):
    self.order = int(other.order)
    self.m = other.m.flatten()
    if other.coef.shape[0] == self.order * 2:
      self.coef = other.coef
    else:
      self.coef = other.coef.T
    self.om = other.om
    self.om.shape = (1,len(self.om.flat))
    return self

  def filter( self, coef ):
    """Filter the signal by multiplication in the frequency domain
       Assuming an order N fourier series of dimension D,
       coef can be of shape:
        N -- multiply all D coefficients of order k by
            coef[k] and conj(coef[k]), according to their symmetry
        2N -- multiply all D coefficients of order k by
            coef[2k] and coef[2k+1]
        1xD -- multiply each coordinate by the corresponding coef
        NxD -- same as N, but per coordinate
        2NxD -- the obvious...
    """
    coef = asarray(coef)
    if coef.shape == (1,self.coef.shape[1]):
      c = coef
    elif coef.shape[0] == self.coef.shape[0]/2:
      if coef.ndim == 1:
        c = empty( (self.coef.shape[0],1), dtype=self.coef.dtype )
        c[::2,0] = coef
        c[1::2,0] = conj(coef)
      elif coef.ndim == 2:
        assert coef.shape[1]==self.coef.shape[1],"Same dimension"
        c = empty_like(self.coef)
        c[::2,:] = coef
        c[1::2,:] = conj(coef)
      else:
        raise ValueError("coef.ndim must be 1 or 2")
    try:
      self.coef *= c
    except NameError:
      raise IndexError("Coef shape %s does not match FS" % coef.shape)
    return self
  
  @staticmethod
  def bigSum( fts, wgt = None ):
    """[STATIC] Compute a weighted sum of FourierSeries models.
       All models must have the same dimension and order.

       INPUT:
         fts -- sequence of N models
         wgt -- sequence N scalar coefficients, or None for averaging

       OUTPUT:
         a new FourierSeries object
    """
    N = len( fts )
    if wgt is None:
      wgt = ones(N)/float(N)
    else:
      wgt = asarray(wgt)
      assert wgt.size==len(fts)

    fm = FourierSeries()
    fm.coef = zeros(fts[0].coef.shape,complex)
    fm.m = zeros(fts[0].m.shape,complex)
    for fs,w in zip(fts,wgt):
      fm.coef += w * fs.coef
      fm.m += w * float(fs.m)
    fm.order = fts[0].order
    fm.om = fts[0].om.copy()

    return fm

