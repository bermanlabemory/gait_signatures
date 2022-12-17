import warnings
from numpy import *
import scipy
from scipy.linalg import eig as geneig
from scipy.signal import lfilter
from scipy.stats import norm as gaussian

def fit_ellipse(xy):
  '''fit an ellipse to the points.

  INPUT:
    xy -- N x 2 -- points in 2D

  OUTPUT:
    A,B,C,D,E,F -- real numbers such that:
      A * x**2 + B * x * y + C * y**2 + D * x + E * y + F =~= 0

  This is an implementation of:

  "Direct Least Square Fitting of Ellipses"
  by Andrew Fitzgibbon, Maurizio Pilu, and Robert B. Fisher
  IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE,
  VOL. 21, NO. 5, MAY 1999

  Shai Revzen, U Penn, 2010
  '''
  xy = asarray(xy)
  N,w = xy.shape
  if w != 2:
    raise ValueError('Expected N x 2 data, got %d x %d' % (N,w))
  x = xy[:,0]
  y = xy[:,1]
  D = c_[ x*x, x*y, y*y, x, y, ones_like(x) ]
  S = dot(D.T,D)
  C = zeros((6,6),D.dtype)
  C[0,2]=-2
  C[1,1]=1
  C[2,0]=-2
  geval,gevec = geneig( S, C )
  idx = find( geval<0 & ~ isinf(geval) )
  return tuple(gevec[:,idx].real)

def smooth(x,window_len=10,window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=r_[2*x[0]-x[window_len:1:-1],x,2*x[-1]-x[-1:-window_len:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=ones(window_len,'d')
    else:
        w=eval(window+'(window_len)')

    y=convolve(w/w.sum(),s,mode='same')
    return y[window_len-1:-window_len+1]


def iqr( x ):
  """ x -- NxD"""
  N, D = x.shape
  s = sort(x, 0)
  return s[floor(N*.75),...] - s[floor(N*.25),...]

def xvPhase( x, v ):
  assert x.shape == v.shape
  N,D = x.shape
  alter = array([1,-1] * (D/2))
  x_score = dot( x, alter[:,newaxis] )
  v_score = dot( v, alter[:,newaxis] )
  x_score -= mean(x_score)
  v_score -= mean(v_score)
  x_score /= std(x_score)
  v_score /= std(v_score)
  phi = angle(x_score + 1j*v_score)
  if median(diff(phi)) < 0:
    phi = 0.0 - phi
  return phi

def MUL( *lst ):
  """Matrix multiply multiple matrices
     If matrices are not aligned, gives a more informative error
     message than dot(), with shapes of all matrices
  """
  try:
    return reduce( dot, lst )
  except ValueError(msg):
    sz = [ asarray(x).shape for x in lst ]
    raise ValueError("%s for %s" % (msg,sz))

def densityPlot1( data,
    returnX=False,
    bins=None,
    sigma=None,
    boundary=None,
    sd=2,
    ang=False):
  """Obtain a kernel smoothed density of the data
     INPUT:
       data -- DxN or N -- N data points of D dimensional data
       bins -- int -- number of bins (default 256)
       sigma -- float -- width of kernel (default 3 bins)
       boundary -- (float,float) -- range of plot (default all data)
       sd -- int -- number of significant std-divs to include in kernel
       ang -- bool -- set to true to treat data as angles in radians -pi..pi
          when ang is set, the data within 2*sigma from -pi and pi is
          repeated at +2*pi and -2*pi prior to kernel smoothing. This
          ensures that the smoothed result is periodic in most cases.
     OUTPUT:
       if returnX:
         returns x,d such that plot(x,d) will plot the density plot
       else:
         returns d.T
  """
  data = asarray(data)
  if data.ndim>2:
    raise IndexError("Data array must have at most two indices (ndim<=2)")
  if boundary == None:
    boundary = (data.min(), data.max())
  if bins is None:
    bins = 256
  if sigma is None:
    sigma = (boundary[1]-boundary[0])*3.0/bins
  if ang:
    boundary = (-pi,pi)
    data = (data + pi) % (2*pi) - pi
  if data.ndim == 1:
    data = (data,)
  # Compute data histograms
  h0 = array([
    histogram(x,range=boundary,bins=bins,normed=True)[0]
        for x in data
    ])
  # Smoothing kernel
  ker = exp(-linspace(-sd,sd,ceil(sigma * sd * bins))**2)
  ker/= sum(ker)
  # If doing an angle density, build up some padding on both sides
  if ang:
    ext = len(ker)
    h0 = hstack( (h0[...,-ext:], h0, h0[...,:ext]) )
    d = array([ convolve(hh,ker,'same') for hh in h0 ])
    d = d[:,ext-1:-ext]
  else: # not angle --
    d = array([ convolve(hh,ker,'same') for hh in h0 ])
  # Normalize
  nrm = trapz( d, dx = (boundary[1]-boundary[0])/float(d.shape[1]), axis=1 )
  d = d / nrm[:,newaxis]
  if returnX:
    if ang:
      return linspace(-pi,pi,d.shape[1])[:,newaxis],d.T
    return linspace(boundary[0],boundary[1],d.shape[1])[:,newaxis],d.T
  return d

def rigidFix( ref, dat ):
  """
  Find rigid transforms that fix dat to reference configuation
  INPUT:
    dat -- N x M x D -- N samples of M points in D dimensions
    ref -- M x D -- reference configuration for the points
  OUTPUT:
    rot -- N x D x D -- N orthogonal transformations
    ctr -- N x D -- N translations
  finds rot, ofs such that:
    dat[k,...] ~= dot( ref, rot[k,...] ) + ctr[k,newaxis,:]
  """
  dat = asarray(dat)
  N,M,D = dat.shape
  ctr = mean( dat, axis=1 ) - mean( ref, axis=0 )[newaxis,:]
  bof = dat - ctr[:,newaxis,:]
  rot = zeros( (N,D,D), dtype=dat.dtype )
  for k in xrange(N):
    R,_,_,_ = scipy.linalg.lstsq( bof[k,...], ref )
    U,_,V = svd(R)
    rot[k,...] = dot( U, V )
  return rot, ctr

class selUI( object ):
  """
  Allow user to click to highlight and identify line handles
  """
  def __init__(self, lines=None):
    if lines is None:
      lines = gca().get_lines()
    assert all([ isinstance( h, matplotlib.lines.Line2D ) for h in lines ])
    self.lh = lines
    self.fig = gcf()
    self.ax = gca()
    self.o_lw = []
    self.o_lal = []
    self.fig.set_visible(False)
    for lh in lines:
      self.o_lw.append( lh.get_lw() )
      self.o_lal.append( lh.get_alpha() )
      lh.set( lw = 1, alpha = 0.4, picker=2 )
    self.sel = None
    self.lbl = None
    self.fig.canvas.mpl_connect( 'pick_event', self._doPick )
    self.fig.set_visible(True)
    draw()

  def _lblUpdate( self, x, y, N ):
    if self.lbl is None:
      lbl = text( x, y, str(N) )
      lbl.set_backgroundcolor((1,1,1))
      self.lbl = lbl
    else:
      lbl = self.lbl
    lbl.set( x = x, y = y, text = str(N), visible=True )

  def _axesPicked( self ):
    if self.sel is not None:
      l = self.lh[self.sel]
      l.set( lw = 1, alpha = 0.4 )
      self.sel = None
    if self.lbl is not None:
      self.lbl.set_visible(False)

  def _linePicked( self, sel, x, y ):
    if self.sel is not None:
      l = self.lh[self.sel]
      l.set( lw = 1, alpha = 0.2 )
      self.sel = None
    self.sel = sel
    self.lh[sel].set(
      alpha = self.o_lal[ self.sel ],
      lw = self.o_lw[ self.sel ] + 1
      )
    self._lblUpdate( x,y, sel )

  def _doPick( self, event ):
    try:
      idx = self.lh.index( event.artist )
      self._linePicked( idx, event.mouseevent.xdata, event.mouseevent.ydata )
    except ValueError:
      self._axesPicked()
    draw()

def get_density( data, bins, boundary = None, num = 1 ):
  """Generate a density plot smoothed with a gaussian kernel"""

  #operates along the 0th dimension
  if boundary == None:
    boundary = (min(data), max(data))

  d = boundary[1] - boundary[0]
  length = 1000
  sigma = d / bins / 3.0
  res = zeros(length)
  c = 1 / sigma / sqrt(2*pi)
  t = linspace(boundary[0], boundary[1], length)
  for k in range(data.shape[0]):
    res += c * exp( -(t - data[k])**2 / 2 / sigma**2 )
    #PDB.set_trace()
  #print("ala")
  scl = sum(res) / num
  return t, res, scl

def real_eig( A ):
  """Do an eigenvalue decomposition over the reals for a real matrix A
     Does this by doing eig over the complex numbers, then replacing
     complex conjugate pairs by 2x2 blocks.

     Implementation will only work correctly for matrices without
     non-trivial Jordan blocks (i.e. only when geometric and algebraic
     multiplicities of eigenvalues are equal)
  """
  w,v = eig( A )
  sorted_index = argsort( w.real )
  w = w[sorted_index]
  w = diag(w)
  v = v[:,sorted_index]
  k = 0
  while k < w.shape[0]:
    if iscomplex(w[k,k]):
      v[:,k], v[:,k+1] = v[:,k].imag.copy(), v[:,k].real.copy()
      l = w[k,k]
      w[k:k+2,k:k+2] = array([[l.real, -l.imag],[l.imag, l.real]])
      k += 1
    k += 1
  return w.real,v.real

def cEigFromReal( D ):
  """Obtain the complex eigenvalues of a block-diagonal real matrix
  """
  return diag(D)+1j*r_[diag(D,1),[0]]+1j*r_[[0],diag(D,-1)]

def b2x2FromEig( E ):
  """Convert eigenvalue list to block 2x2 real matrix
  """
  return diag(E.real)+diag(E.imag[:-1],1)-diag(E.imag[:-1],-1)

def signFix( M ):
  """Fix signs of columns of M so that the real part with largest
     absolute value is always positive.

     When columns represent subspaces rather than vectors, this
     selects a canonical form

     Changes M in place!
  """
  sz = M.shape
  M.shape = M.shape[0],prod(M.shape[1:])
  s = array([sign(M[argmax(abs(M[:,k]))],k) for k in xrange(M.shape[1])])
  M *= s[newaxis,:]
  M.shape = sz

def eigv2real( u, allNorm=False ):
  """Take a (complex) right eigenvector matrix u of a matrix M and
     convert to a real-only form. With respect to the new form, A will be
     block diagonal real.

     OUTPUT:
       u_real -- the reals only, canonical form of u

     Given real matrix M such that dot(dot(pinv(u),M),u) is diagonal
     the matrix dot(dot(pinv(u_real),M),u_real) is block diagonal

     When allNorm is true, all columns are normalized, at the expense of
     making the off diagonal elements of A_diag no longer anti-symmetric

     (Thanks to Sam Burden for help in debugging the algebra)
  """
  ur = u.copy()
  mi = argmax(abs(ur.imag),axis=0)
  # Sign of largest real part
  sr = array([
    sign(ur[argmax(abs(ur[:,k].real)),k].real)
    for k in xrange(ur.shape[1])
  ])
  # Convert signs. This doesn't change the subspace spanned by each column
  ur *= sr[newaxis,:]
  # If allnorm isn't set --> Column norms come only from the real part...
  if not allNorm:
    nrm = sqrt(sum(ur.real**2,axis=0))[newaxis,:]
  # Sign of largest imaginary part
  si = array([sign(ur[mi[k],k].imag) for k in xrange(ur.shape[1])])
  # Columns with si<0 are converted to real part
  neg = find(si<0)
  ur[:,neg] = ur[:,neg].real
  # Columns with si>0 are converted to imaginary part
  pos = find(si>0)
  ur[:,pos] = ur[:,pos].imag
  # Renormalize columns
  # If allnorm is set --> Renormalize all the columns
  if allNorm:
    nrm = sqrt(sum(ur.real**2,axis=0))[newaxis,:]
  u_real = ur.real / nrm
  # Return results
  return u_real

def matMaxPermute( M ):
  """Find permutations that maximize the diagonal of M, as follows:

     INPUT:
       M -- RxC -- real matrix

     OUTPUT:
       r -- R -- permutation of rows
       c -- C -- permutation of columns

     Such that M[r,:][:,c] is a matrix for which M[k,k] is maximal
     in the sub-matrix M[k:,k:]

     This is useful given two orthogonal matrices U and V, since
     the r,c permutations of the product M=dot(U,V.T) give a greedy
     matching of the rows of U to the rows of V based on angle cosine
  """
  r = range(M.shape[0])
  c = range(M.shape[1])
  ro = []
  co = []
  while r and c:
    Ms = M[r,:][:,c]
    ri = argmax(Ms.max(axis=1))
    ci = argmax(Ms.max(axis=0))
    ro.append( r.pop(ri) )
    co.append( c.pop(ci) )
  return array(ro), array(co)



def svd_inv( x ):
  """Computes the inverse of any matrix using svd
  """
  M,N = x.shape
  U,S,V = svd( x )

  if M < N:
    Si = vstack( [diag( 1/S ), array( [[0] * M] * (N-M) )] )
  else:
    Si = hstack( [diag( 1/S ), array( [[0] * (M-N)] * N ) ] )

  return reduce( dot, [V.T, Si, U.T] )

def princomp( x ):
  """Principal component analysis

  INPUT:
    x -- N_1 x ... x N_k x D -- observations of a D dimensional variable
  OUTPUT:
    L -- D -- Latent value in each principal axis
    V -- D x D -- Matrix of principal components
    S -- N_1 x ... x N_k x D -- Scores of x along principal axes
    Latent values are sorted by size
  """
  #dat = x - mean( x, -1 ).reshape(( x.shape[0], 1 ))
  #PDB.set_trace()
  dat = asarray(x)
  dat = dat.reshape( (prod(dat.shape[:-1]),dat.shape[-1]) ).T
  if not allclose(mean(dat,axis=1),0):
    warnings.warn( "Mean of data MUST be the zero vector" )
  p,n = dat.shape
  U, s, V = svd(dat / sqrt(n-1.0), full_matrices=False)
  score = dot(dat.T,U.T)
  latent = s**2
  return latent,U.T,score.reshape(x.shape)

def tlstsq(A,B,inverse = linalg.inv):
    """Total least squares regression
         A -- m x n
         B -- m x r
         inverse -- choice of matrix inversion function to use
       returns X
         X -- n x r -- such that X minimizes E_A**2 + E_B**2 in the
            equation dot((A+E_A), X) = B+E_B
    """
    # n is the width of A (A is m by n)
    n = A.shape[1]
    # C is A augmented with B
    C = hstack([A,B])
    # find the SVD of C.
    U,s,V = svd(C,full_matrices = 0)
    # Take the block of V with rows of A and columns of B
    VAB = V.T[:n,n:]
    # Take the block of V between B and itself
    VBB = V.T[n:,n:]
    return -dot(VAB,inverse(VBB))

def hilbert( x ):
  """Discrete-time analytic signal via Hilbert transform.

    Operates along the last dimension

    Computes the so-called discrete-time analytic signal X = Xr + i*Xi
    such that Xi is the Hilbert transform of real vector Xr.

    From:
      "Computing the discrete-time analytic signal via FFT",
        by S. Lawrence Marple, Jr.
        in IEEE Transactions on Signal Processing,
        Vol. 47, No. 9, 1999, pp.2600--2603.

    By Shai Revzen, Berkeley 2008
  """
  n = x.shape[-1]
  global fft,ifft
  if not callable(fft):
    fft = fft.fft
    ifft = fft.ifft
  f = fft(x)
  h = zeros(f.shape)
  if n % 2 == 0:
    h[...,0] = 1
    h[...,n/2] = 1
    h[...,1:n/2] = 2
  else:
    h[...,0] = 1
    h[...,1:(n+1)/2]= 2
  return ifft( f * h )

def lfilter_zi(b,a):
    #compute the zi state from the filter parameters. see [Gust96].

    #Based on:
    # [Gust96] Fredrik Gustafsson, Determining the initial states in forward-backward
    # filtering, IEEE Transactions on Signal Processing, pp. 988--992, April 1996,
    # Volume 44, Issue 4

    n=max(len(a),len(b))

    zin = (  eye(n-1) - hstack( (-a[1:n,newaxis],
                                 vstack((eye(n-2), zeros(n-2))))))

    zid=  b[1:n] - a[1:n]*b[0]

    zi_matrix=linalg.inv(zin)*(matrix(zid).transpose())
    zi_return=[]

    #convert the result into a regular array (not a matrix)
    for i in range(len(zi_matrix)):
      zi_return.append(float(zi_matrix[i][0]))

    return array(zi_return)

def filtfilt(b,a,x):
    """Apply a linear filter in zero phase-shift configuration

       Runs the filter "forwards and backwards" causing phase shifts to
       cancel. This can only be usen with filters that do not rely on phase
       shift to function properly. For example, this works well with Butterworth
       filters but not with Chebyshev filters.

       Shai Revzen, Berkeley 2009
    """
    # If applied to multidimensional array -- vectorize, applied to first index
    if x.ndim != 1:
      s = x.shape
      x.shape = (x.shape[0],prod(x.shape[1:]))
      y = zeros(x.shape,dtype=x.dtype)
      for c in xrange(x.shape[1]):
        y[:,c] = filtfilt(b,a,asarray(x[:,c]).flatten())
      y.shape=s
      return y

    ntaps=max(len(a),len(b))
    edge=ntaps*3

    #x must be bigger than edge
    if x.size < edge:
        raise ValueError("Input vector needs to be bigger than 3 * max(len(a),len(b).")

    if len(a) < ntaps:
      a=r_[a,zeros(len(b)-len(a))]

    if len(b) < ntaps:
      b=r_[b,zeros(len(a)-len(b))]

    zi=lfilter_zi(b,a)

    #Grow the signal to have edges for stabilizing
    #the filter with inverted replicas of the signal
    s=r_[2*x[0]-x[edge:1:-1],x,2*x[-1]-x[-1:-edge:-1]]
    #in the case of one go we only need one of the extrems
    # both are needed for filtfilt

    (y,zf)=lfilter(b,a,s,-1,zi*s[0])

    (y,zf)=lfilter(b,a,flipud(y),-1,zi*y[-1])

    return flipud(y[edge-1:-edge+1])

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
    """Get order of Fourier series"""
    return self.coef.shape[0]/2

  def extend( self, other ):
    """Extend a fourier series with additional output columns from
       another fourier series of the same order

       If fs1 is order k with c1 output colums, and
       fs2 is order k with c2 output columns then the following holds:

       fs3 = fs1.copy().append(fs2)
       assert allclose( fs3.val(x)[:c1], fs1.val(x) )
       assert allclose( fs3.val(x)[c1:], fs2.val(x) )
    """
    assert len(other.om) == len(self.om), "Must have same order"
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

       data is a row or two-dimensional array, with data points in columns
    """

    phi = reshape( mod(ph + math.pi,2*math.pi) - math.pi, (1,len(ph.flat)) )
    if phi.shape[1] != data.shape[1]:
      raise IndexError(
        "There are %d phase values for %d data points"
            % (phi.shape[1],data.shape[1]))
    # Sort data by phase
    idx = argsort(phi).flatten()
    dat = c_[data.take(idx,axis=-1),data[:,idx[0]]]
    phi = concatenate( (phi.take(idx),[phi.flat[idx[0]]+2*math.pi]) )

    # Compute means values and subtract them
    #self.m = mean(dat,1).T
    # mean needs to be computed by trap integration also
    dphi = diff(phi)
    self.m = sum((dat[:,:-1] + dat[:,1:]) * .5 * dphi[newaxis,:], axis = 1) / (max(phi) - min(phi))
    #PDB.set_trace()
    dat = (dat.T - self.m).T
    # Allow 0th order (mean value) models
    order = max(0,order)
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
    zd = (dat[:,1:]+dat[:,:-1])/(2.0*2*math.pi) * dphi
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
    self.coef *= c
    return self

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
    fm.coef = zeros_like(fts[0].coef)
    fm.m = zeros_like(fts[0].m)
    for fs,w in zip(fts,wgt):
      fm.coef += w * fs.coef
      fm.m += w * fs.m
    fm.order = fts[0].order
    fm.om = fts[0].om

    return fm
  bigSum = staticmethod( bigSum )

def extarray( a, sz ):
  """Extend array to specified shape. Array shape must be a prefix of sz"""
  N = len(a.shape)
  # Allow trailing indices==1 to be skipped
  while a.shape[N-1]==1:
    N=N-1
    assert sz[:N]==a.shape[:N]
  return reshape(repeat( a, prod(sz[N:]), -1 ),sz)

def section( dat, seq, lim=None, arg=False ):
  """Compute Poincare section of dat at positive zero crossings of seq

     Uses linear interpolation.
     If lim is defined, steps larger than lim in seq are ignored.
     If arg is set to True, returns tuple: idx,y
       idx -- indices of sample before section
       y -- value at section
  """
  seq = asarray(seq).ravel()
  dat = asarray(dat)
  assert len(seq) == len(dat)
  s = ravel((seq[1:]>0) & (seq[:-1]<=0))
  idx = arange(seq.shape[0])[s]
  if not lim is None:
    idx = idx[ ravel((abs(seq[idx]) < lim) & (abs(seq[idx+1]) < lim))]

  y0 = dat[idx]
  y1 = dat[idx+1]
  #x0 = extarray( seq[pre], y0.shape )
  #x1 = extarray( seq[post], y1.shape )
  #PDB.set_trace()
  x0 = seq[idx,:]
  x1 = seq[idx+1,:]
  #PDB.set_trace()
  y = (y0*x1[:,newaxis]-y1*x0[:,newaxis])/(x1-x0)[:,newaxis]
  if arg:
    return idx, y
  return y

import pdb as PDB

def section01( dat, slc0, slc1, arg=False ):
  """Compute two sections and match them into input-output pairs
     Returns section with each input-output pair as a column of
     dimension twice that of dat.

     If arg is true, return idx,res where res is as above, and idx
     are the indices of the closest entries in dat.
  """
  i0,s0 = section( dat, slc0, arg=True)
  i1,s1 = section( dat, slc1, arg=True)
  if i0[0] >= i1[0]:
    i1 = i1[1:]
    s1 = s1[1:]
  print(i1, i0)
  sec = array([arg2 for val,arg2 in genAlternating( i0, i1 )])
  res = c_[s0[sec[:,0]],s1[sec[:,1]]]
  if arg:
    idx = c_[i0[sec[:,0]],i1[sec[:,1]]]
    return idx, res
  return res

def rotN( ang, xi, yi, vec ):
  """Rotate the vectors vec (given as rows) through the angle ang in
     components xi and yi

     If vec is an integer N, returns a NxN rotation matrix R such that

     dot(vec,R) == rotN( ang, xi, yi, vec )

     rotN is a "generalized rotation" in arbitrary dimension.
     In dimensions higher than 3 there are several ways to think of rotations:
     (1) As orthogonal matrices; these are of dimension N*(N-1)/2
     (2) As products of two-dimensional rotations, each of which affects only
         coordinates within a two-dimensional hyperplane.
     (3) Same as (2), but restrict your attention only to the N*(N-1)/2
         "standard basis" of 2-forms, i.e. only to planes that are spanned
         by two coordinates of the standard basis.

     rotN takes the last approach; xi and yi are coordinate indices. For example
     rotN(pi/6,3,5,9) would return a matrix that rotates pi/6 radians in the
     hyperplane spanned by coordinates 3 and 5 of 9-dimensional space.

     Furthermore, rotN can be used in two ways:
     (1) perform the rotations efficiently -- by giving the vectors in vec
     (2) produce the matrix representing the rotation -- giving vec = dimension

     By Shai Revzen, Berkeley 2008
  """
  if type(vec)==int:
    vec = eye(vec)
  if xi>yi:
    xi,yi=yi,xi
    ang = -ang
  v = vec[:,[xi,yi]]
  c = cos(ang) * v
  s = sin(ang) * v
  res = c_[vec[:,:xi],c[:,0]-s[:,1],vec[:,(xi+1):yi],c[:,1]+s[:,0],vec[:,yi+1:]]
  return res

def bsBalance( dats, N ):
  """Return a bootstrap sample from rows of elements of dats.
     Each sample will have exactly N columns.

     INPUT:
       dats -- sequence containing arrays with identical shape[1:]
       N -- integer >0

     OUTPUT: pair (S,V)
       S -- a list of len(dats) arrays of shape N,dat[0].shape[1:]
            To combine into a single array use r_.__getitem__(tuple(S))
            NOTE: S may well have duplicate rows!
       V -- a sequence of the unselected portions of the dats
            (i.e. parts not in S)
  """
  assert N==int(N) and N>0
  if len(dats)==0:
    return array([]),[]
  S = []
  V = []
  sz = asarray(dats[0]).shape[1:]
  for dat in dats:
    a = asarray(dat)
    assert a.shape[1:] == sz, "dats elements have matching shapes"
    idx = randint( a.shape[0], size=N )
    sel = zeros( a.shape[:1], dtype=bool )
    sel[idx] = True
    S.append( a[idx] )
    V.append( a[~sel] )
  return S,V



def bsRegStep( x, y, N, pr=0, smo=None ):
  """Perform a single bootstrap regression step

     Given input columns x, output columns y and proposed predictive
     relationship between then pr, generate a bootstrap xb,yb of size N
     by resampling the prediction pr residuals.

     If pr is not callable, it is assumed to be a matrix.

     If smo is not None, it is used to recondition the array of residuals prior
     to the resampling step. If it is callable it is applied to residuals,
     otherwise it is treated as a matrix, applied to standard Gaussian data and
     the result added to the residuals (allowing Kernel smoothing to be applied)

     Also returns validation data indices -- the indices of columns that were
     not used for generating bootstrap samples.

     returns xb,yb,vdi
  """
  if len(x.shape)<2:
    x.shape = (1,x.shape[0])
    y.shape = (1,y.shape[0])
  assert x.shape[1] == y.shape[1]
  L = x.shape[1]
  idx = randint( L, size=N )
  ridx = randint( L, size=N )
  vd = ones( L, dtype=bool )
  vd[idx] = False
  vd[ridx] = False

  # Compute model predictions
  if callable(pr):
    y0 = pr(x)
  else:
    y0 = dot(pr,x)
  # Compute bootstrapped residuals
  resid = (y-y0)[:,ridx]
  # Recondition residuals
  if smo is None:
    pass
  elif callable(smo):
    resid = smo(resid)
  else:
    assert smo.shape[0] == resid.shape[0] and smo.shape[0]==smo.shape[1]
    resid = resid + dot( smo, randn(*resid.shape) )
  return x[:,idx],y0[:,idx]+resid,nonzero(vd)[0]

def genAlternating( l, u ):
  """Given sorted sequences l, u, generate alternating pairs
       (l[n],u[m]),(n,m) for monotone increasing n,m such that
     pairs are increasing, i.e.
          u[m-1] < l[n] <= u[m], l[n] <= u[m] < l[n+1]
     and "close", i.e. l[n],u[m] are the closest pair of all valid
     monotone alternatives.

     This is useful when l,u are positions of events expected to occur
     in an alternating positions, but were detected independently and may
     have false positives and/or negatives.
  """
  gl = (x for x in zip(l,xrange(len(l))))
  gu = (y for y in zip(u,xrange(len(u))))
  u0 = gu.next()
  l0 = gl.next()
  # If head of l is too large --> go down u until ok
  while l0>=u0:
    u0 = gu.next()
  # Emit pairs
  while True:
    try:
      l1 = gl.next()
      while l1<u0:
        l0 = l1
        l1 = gl.next()
    except StopIteration:
      yield tuple(zip(l0,u0))
      return

    yield tuple(zip(l0,u0))
    l0 = l1
    u0 = gu.next()
    while u0<=l0:
      u0 = gu.next()

import time, glob, os
from pylab import *
class Animation(object):
  """ Animation can be used to generate animations of matplotlib figures
      both at runtime and as avi files using ffmpeg compression.

      The same interface is used, with only one parameter changin the
      mode from "live" to "recorded" animations.

      Typical useage:
      >>> A = Animation(0.1)
      >>> for foo in fooSequence:
      ...    plot( *foo )
      ...    A.step()
      ... A.stop()
      >>> A = Animation("fooSequence.avi")
      >>> for foo in fooSequence:
      ...    plot( *foo )
      ...    A.step()
      ... A.stop()
  """
  def __init__( self, pth, fps=2):
    """
    Create an Animation object
    INPUT:
      pth : string -- pathname to an .avi file (must end with .avi)
        or
      pth : float -- delay between frames in "live" animation
      fps : number (optional) -- fps of recorded animation
    """
    self.n = 1
    self.fig = gcf()
    if type(pth)==str:
      pfx = pth[:-4]
      if pth[-4:].lower() != ".avi":
        raise TypeError("Filename pattern must end with .avi")
      if glob.glob(pfx):
        raise KeyError("A file/directory by the name '%s' exists -- bailing out" % pfx)
      os.system('mkdir -p %s' % pfx)
      self.pfx = pfx
      self.pth = pth
      self.fps = fps
      self.step = self._step_AVI
      self.stop = self._stop_AVI
    else:
      self.dur = float(pth)
      self.step = self._step_PAUSE
      self.stop = self._stop_PAUSE
    self.fig.set_visible(False)

  def _step_PAUSE(self):
    self.fig.set_visible(True)
    draw()
    time.sleep(self.dur)
    self.fig.set_visible(False)

  def _stop_PAUSE(self):
    self.fig.set_visible(True)
    draw()

  def _step_AVI(self):
    self.fig.set_visible(True)
    self.n = self.n + 1
    draw()
    savefig( "%s/fr-%04d.png" % (self.pfx,self.n) )
    self.fig.set_visible(False)

  def _stop_AVI(self):
    self.fig.set_visible(True)
    draw()
    exc = None
    try:
      os.system("mencoder mf://%s/fr-*.png "
        "-mf fps=%d:type=png -ovc lavc "
        "-lavcopts vcodec=mpeg4:mbd=2:trell"
        " -oac copy -o %s" % (self.pfx,self.fps,self.pth))
    except ex:
      exc = ex
    os.system('rm -rf %s' % self.pfx )
    if exc is not None:
      raise exc

class VisEig:
  """VisEig is a tool for visualizing distributions of complex numbers
  that are, generally speaking, around 0. It is typically used to show
  eigenvalue distributions of ensembles of matrices.

  Typical useage:
  >>> vi = VisEig()
  >>> [ vi.add(eigvals(M)) for M in listOfMatrices ]
  >>> vi.vis()
  """
  def __init__(self, N=63, fishEye=True):
    self.N = N
    self.H = zeros((self.N,self.N))
    self.Hr = zeros(self.N)
    self.scl = 1
    self.fishy = fishEye

  def clear(self):
    self.H[:] = 0
    self.Hr[:] = 0

  def fishEye( self, z ):
    z = asarray(z).copy().flatten() * self.scl
    if self.fishy:
      r = abs(z)
      b = r>1
      z[b]= z[b]/r[b]*(2-1/r[b])
    return z.real, z.imag

  def add( self, ev ):
    x,y = self.fishEye(ev.flatten())
    H = histogram2d( y,x,self.N, range=[[-2.0,2.0],[-2.0,2.0]] )[0]
    if any(y==0):
      Hr = histogram( x[y==0], bins=self.N, range=(-2.0,2.0) )[0]
      self.Hr = self.Hr + Hr
    self.H = self.H + H

  def plt( self ):
    """
    Plot marginal density of real parts, and marginal density
    of real parts of numbers which were real to begin with

    return plot object handles
    """
    x = linspace(-2,2,self.N)
    tot = sum(self.H.flat)
    h = plot(x,sum(self.H,0)/tot,color=[0,0,1],linewidth=3)
    h.extend(plot(x,self.Hr/tot,color=[0.5,0.5,0.5],linewidth=2))
    return h

  def vis( self, rlabels = [0.5,1], N=None):
    "Visualize density with color-coded bitmap"
    fig = gcf()
    vis = fig.get_visible()
    fig.set_visible(False)
    h = [imshow(log(1+self.H),interpolation='nearest',extent=[-2,2,-2,2])]
    self._decorate( rlabels, dict(color=[1,1,1],linestyle='--') )
    fig.set_visible(vis)
    return h

  def vis1( self, rlabels = [0.5,1], N=8, **kw ):
    "Visualize density with contours"
    fig = gcf()
    vis = fig.get_visible()
    fig.set_visible(False)
    z = log( 1 + self.H )
    h = [contour( z, N, extent=[-2,2,-2,2], **kw )]
    self._decorate( rlabels, dict(color=[0,0,0],linestyle='--') )
    fig.set_visible(vis)
    return h

  def _decorate( self, rlabels, lt,  ):
    axis('equal')
    rlabels = asarray(rlabels).flatten()
    t = linspace(-3.1415,3.1415,64)
    t[-1]=t[0]
    d = exp(1j*pi/4)
    self.h_lbl = []
    self.h_circ = []
    for r in rlabels:
      x, y = self.fishEye( exp(1j * t)*r )
      self.h_circ.append(
        plot( x, y, **lt )[0]
      )
      x, y = self.fishEye( r*d )
      self.h_lbl.append(
        text(x,y,"%g" % r, color=lt['color'])
      )
    # Prepare labels and label positions
    l = concatenate([-rlabels[::-1],[0],rlabels])
    v0 = self.fishEye( rlabels )[0]
    v = concatenate([-v0[::-1],[0],v0])
    gca().set(xticks=v,xticklabels=l,yticks=v,yticklabels=l)

def glueColumns( seq ):
  return c_.__getitem__( tuple(seq) )

def qqplot( Sy, Sx = gaussian.isf ):
    """Quantile-quantile plot the samples against each other
       If Sx is a function, it is assumed to be the
       inverse survival function to be used as a reference
    """
    Sy = asarray(Sy).flatten()
    Sy.sort()
    n = len(Sy)
    if callable(Sx):
      Sx = -Sx(linspace(0.1/n,1-0.1/n,n))
    else:
      Sx = asarray(Sx).flatten()
      Sx.sort()
    plot( Sx, Sy, '.' )

def planAxes( grid, bw=0.1, withAxes=True ):
    """
    Convenience function for creating a figure layout
    INPUT:
      grid -- ASCII art layout (see example below)
      bw -- border width in grids; either scalar or (horizontal,vertical)
      withAxes -- if True, create axes and return them; else return boxes
    OUTPUT:
      Axes for regions, sorted by region id chars.
      Regions marked with space, '-' or '|' are ignored
    Example:
      p='''
      xxxxx yyy
      xxxxx yyy
      xxxxx yyy

      zzzzzzzzz
      zzzzzzzzz
      '''
      ax_x,ax_y,ax_z = planAxes(p,bw=0.5)
    """
    # Convert border to 2 entry array
    bw = asarray(bw)
    if bw.size==1:
      bw = array([bw,bw])
    else:
      assert bw.size==2
    # Split lines
    g0 = grid.strip().split("\n")
    l = max([len(r) for r in g0])
    # Pad lines to constant length
    pad = " "*l
    # Create grid of chars
    g = array([ [y for y in (x+pad)[:l]] for x in g0][::-1])
    xs = 1.0/g.shape[1]
    ys = 1.0/g.shape[0]
    lst = unique(g.flatten())
    res = []
    bx,by = bw
    for nm in lst:
      if nm in ' -|':
        continue
      ind = (g==nm)
      xi = find(any(ind,axis=0))
      yi = find(any(ind,axis=1))
      box = (
        (xi[0]+bx)*xs,(yi[0]+by)*ys,
        xs*(xi[-1]-xi[0]+1-2*bx),ys*(yi[-1]-yi[0]+1-by)
      )
      if withAxes:
        res.append(axes(box))
      else:
        res.append(box)
    return res

class ecdf( object ):
    def __init__(self,data):
      """Construct empirical cumulative distribution function from data sample"""
      self.tbl = asarray(data).flatten().copy()
      self.tbl.sort()

    def remap( self, func, *args, **kwargs ):
      """Remap probability distribution from p(x) to p(func(x))

         e.g. ecdf(randn(1000)).remap( lambda x: x+1 )
         will give a normal distribution with mean 1
      """
      ntbl = fromiter( (func(x,*args,**kwargs) for x in self.tbl ), dtype=tbl.dtype )
      ntbl.sort()
      self.tbl = ntbl

    def resize( self, N ):
      """Resize internal representation to N entries"""
      self.tbl = self.inv(linspace(0,1.0,N))

    def __add__( self, other ):
      if isinstance(other,ecdf):
        if self.tbl.size > other.tbl.size:
          return ecdf( concatenate( (self.tbl, other.inv(linspace(0,1,self.tbl.size))) ))
        else:
          return ecdf( concatenate( (other.tbl, self.inv(linspace(0,1,other.tbl.size))) ))
      other = asarray(other)
      assert other.size == 1
      return ecdf( self.tbl + other )

    def __mul__( self, other ):
      other = asarray(other)
      assert other.size == 1
      res = ecdf([])
      if other>0:
        res.tbl = self.tbl.copy() * other
      else:
        res.tbl = self.tbl.copy()[::-1] * other
      return res

    def __call__(self, x):
      """Compute the cumulative probability of each of the values of x"""
      return self.tbl.searchsorted(asarray(x)) / float(self.tbl.size)

    def support( self ):
      """Range in which probability density is non-zero"""
      return self.tbl.take((0,-1))

    def inv( self, p ):
      """Evaluate the inverse CDF at points p"""
      p = asarray(p)
      return self.tbl[array(self.tbl.size * p.clip(0,1-1e-10),dtype=int)]

    def dist(self, cdf2 ):
      """Compute the difference between two ecdf-s
      OUTPUT: X,delta
        X -- sorted vector of independent parameter
        delta -- difference at corresponding X
      This allows various norms to be computed, e.g.
        x,d = a.dist(b)
        mag = trapz( d**2, x )
      Will give the familiar L2 norm, to second order precision, while
        a.dist(b)[1].max()
      Gives the L-infinity norm (a.k.a. sup norm)
      """
      assert isinstance( cdf2, ecdf )
      if len(self)>len(cdf2):
        a,b = self,cdf2
      else:
        a,b = cdf2,self

      p = b(a.tbl)
      return a.tbl.copy(),(linspace(0,1,len(a)) - p)

    def pdf( self, N=None ):
      """
      Return a (2nd order accurate) probability distribution function
      If N is given, the PDF is sampled at N points
      """
      if N is None:
        x,p = self.asXY()
      else:
        x,p = self.asXY(N+1)
      return (x[1:]+x[:-1])/2.0, diff(p)/diff(x)

    def asXY( self, N=None ):
      """
      Return X,Y that allow ecdf to be plotted at N points, e.g. plot(*a.asXY(100))
      """
      if N is None:
        N = len(self.tbl)
      x = linspace(self.tbl[0],self.tbl[-1],N)
      return x,self(x)

def imsc( img, **kw ):
  """Wrapper for imshow that defaults to nerest neighbour interpolation"""
  if not kw.has_key('interpolation'):
    kw['interpolation']='nearest'
  return imshow( img, **kw )

def matPow( M, p, rTrunc = 0, negTol = 1e-3 ):
    """Compute real powers of a square matrix
       Using the eigendecomposition of M, this function computes
       any real powers of M. For matrix roots, *a* root will be returned,
       but this root is only one of the possible solutions.
       M must not have any real negative eigenvalues. The code treats an
       eigenvalue as negative real if its polar angle is pi to within negTol
       INPUT:
         M -- N x N real matrix
         p -- any numerical array / sequence
         rTrunc -- truncate any eigenvalues smaller than this
           NOTE: for negative powers the results may be quite strange when
           the matrix M has small eigenvalues. Dividing by zero must be done
           carefully...
         negTol -- tolerance for real negative eigenvalue test (assertion only)
       OUTPUT: Mp
         Mp -- p.shape x N x N array of real valued results
    """
    M = asarray(M)
    assert M.ndim == 2, "2D array"
    assert M.shape[-2]==M.shape[-1], "Square"
    assert allclose(M.imag,0), "No imaginary parts allowed"
    p = asarray(p)
    if p.size==1:
      p = array([p])
    szP = p.shape
    # Compute eigendecomposition of M
    E,U = eig(M)
    a = angle(E)
    r = abs(E)
    # Tiny eigenvalues are assumed to be positive reals
    a[r<rTrunc] = 0
    assert all(cos(a)>cos(negTol-pi)), "No negative real eigenvalues allowed"
    # Prepare for results -- compute powers using the polar decomposition
    #  of the eigenvalues
    pa = a[:,newaxis] * p[newaxis,:]
    pr = r[:,newaxis] ** p[newaxis,:]
    # Discard (truncate) any reciprocals of "small" eigenvalues
    pr[ (r<rTrunc),: ] = 0
    # Build eigenvalue array for results
    pe = pr * exp(1j*pa)
    # Construct results
    Ui = pinv(U)
    res = [ dot(dot(U,diag(pe[:,k])),Ui).real for k in xrange(pe.shape[1]) ]
    return array(res).reshape( p.shape+M.shape  )

class SampleAndHold( object ):
  def __init__( self ):
    self.tbl = None
    self.order = 0

  def val( self, t ):
    """Evaluate at the parameter values t"""
    t = asarray(t)
    sz = t.shape
    # Find entries for t
    idx = self.tbl[0,:].searchsorted(t.flatten())
    # Bounds of (-order,+order) windows around nerest sample
    ub = (idx+self.order).clip(0,self.tbl.shape[1]-1)
    lb = (idx-self.order).clip(0,self.tbl.shape[1]-1)
    # Compute mean of each window
    res = array([ mean(self.tbl[1:,slice(l,u)],axis=1)
        for l,u in zip(lb,ub) ])
    # Reshape the output
    res.reshape(sz+res.shape[1:])
    return res

  def fit( self, order, t, data ):
    """Fit multidimensional data; data points are in columns

       INPUT:
         t -- sequence of length N
         data -- array of N x D

       OUTPUT:
         updated self
    """
    t = asarray(t).flatten()
    idx = argsort(t)
    self.tbl = vstack( [t[idx],asarray(data)[:,idx]] )
    self.order = order
    return self

  def copy( self ):
    return deepcopy(self)

  def reindexTo( self, N ):
    """Reindex the interpolator to have exactly N entries

      INPUT: positive integer N
      OUTPUT: (updated) self
    """
    idx = linspace(0,self.tbl.shape[1],N+1).round()
    ntbl = array([ mean(self.tbl[:,l:max(u,l+1)],axis=1)
      for l,u in zip(idx[:-1],idx[1:])
    ]).T
    self.order = ceil(self.order * float(N) / self.tbl.shape[1] )
    self.tbl = ntbl
    return self

  def bigSum( lst ):
    assert all([
      a.tbl.shape[1] == b.tbl.shape[1]
        for a,b in zip(lst[:-1],lst[1:])
      ]), "Sizes must be compatible -- use .reindexTo()"
    t = concatenate( [ o.tbl[0,:] for o in lst] )
    d = hstack([ o.tbl[1:,:] for o in lst])
    return lst[0].copy().fit( lst[0].order, t, d )
  bigSum = staticmethod(bigSum)

class SampleHoldByAngle( SampleAndHold ):
  def wrap( t ):
    return (array(t)+pi) % (2*pi) - pi
  wrap = staticmethod(wrap)

  def val( self, t ):
    return SampleAndHold.val(self,self.wrap(t))

  def fit( self, order, t, data ):
    t0 = self.wrap(asarray(t).flatten())
    idx0 = argsort(t0)
    # Final and initial windows large enough to ensure wrapping
    fw = idx0[:order+2]
    lw = idx0[-order-1:]
    # Adjust indices to duplicate wrapped windows
    idx = concatenate( [lw,idx0,fw] )
    # Duplicate parameter with 2*pi shift
    t1 = concatenate( [t0[lw]-2*pi,t0[idx0],t0[fw]+2*pi] )
    # Build lookup table with extra rows
    self.tbl = vstack( [t1,asarray(data)[:,idx]] )
    self.order = order
    self.coreSlice = (order+1,-order-2)
    return self
  fit.__doc__ = SampleAndHold.fit.__doc__

def angPlotPlan( ang, *lst ):
  """Prepare angle variables for plotting by inserting nan-s where
     they wrap around (+/-)pi.

     INPUT:
       ang -- N x ... -- angle variable
       lst -- M arrays whose shape[0] is N

     OUTPUT:
       A list of M+1 lists, corresponding to contiguous segments of
       the a variable along the first index. If ang is multicolumn,
       segments end whenever *any* column of ang wraps.
       Wrapping is defined as a jumps of more than pi, when taken
       mod 2*pi.

       The first list contains the wrapped version of ang.

     EXAMPLE:
       a = c_[linspace(0,8*pi,256),linspace(1,7*pi,256)]
       c,s = cos(a[:,0]*1.1),sin(a[:,1]*0.9)
       for a0,c0,s0 in zip(*angPlotPlan( a, c, s )):
          figure(1) # clean plot on the 2-torus
          plot(a0[:,0],a0[:,1],'.-' )
          figure(2) # clean w.r.t. to a[:,0], breaks for a[:,1]
          plot(a0[:,0],c0,'.-b')
          figure(3) # clean w.r.t. to a[:,1], breaks for a[:,0]
          plot(a0[:,1],s0,'.-r')
  """
  a = (ang+pi)%(2*pi)-pi
  sz = a.shape
  da = diff(a.reshape(sz[0],prod(sz[1:])),axis=0)
  sk = concatenate( [find(any(abs(da)>pi,axis=1)),[len(da)]] )
  plan = [[] for k in  xrange(1+len(lst)) ]
  pk = 0
  for k in sk:
    plan[0].append( a[pk:k+1,...] )
    for p,l in zip(plan[1:],lst):
      p.append(l[pk:k+1,...])
    pk = k+1
  return plan

def plotTraj( t, x, y, NT=5, NMT=25, fmt="%.2f" ):
  """plot a trajectory, with time markers
  INPUTS:
    t -- N -- time
    x,y -- N x M -- M time series each
    NT -- int -- number of major "ticks" on trajectory; default 5
    NMT -- int -- number of minor ticks; default 25
    fmt -- string -- format string for time values, or false for
      no time labels
  OUTPUT:
    dictionary of all line handles. Keys are:
      major, minor -- major and minor tics
      lines -- trajectory lines
      text -- text object handles
  """
  t,x,y = map(asarray, (t,x,y))
  assert x.shape == y.shape
  assert t.ndim == 1
  N = t.size
  x = x.reshape(x.size / N,N).T
  y = y.reshape(y.size / N,N).T
  assert x.shape[0] == N
  h = plot( x, y, '-', alpha=0.5 )
  mts = max(len(t)/NMT,1)
  minor = plot( x[::mts,...], y[::mts,...], '.' )
  bg = gca().get_axis_bgcolor()
  for mh,lh in zip(minor,h):
    mh.set(mec=bg,mfc=lh.get_color())
  step = max(len(t)/NT,1)
  cj = cm.jet((t[::step]-t[0])/(t[-1]-t[0]))
  tics = []
  txt = []
  for k,fc in zip(xrange(0,len(t),step),cj):
    h2 = plot( x[[k],...], y[[k],...], 'o' )
    for mh,lh in zip(h2,h):
      c = lh.get_color()
      mh.set(mec=c, mfc=fc)
      if fmt:
        txt.append(text(
          mh.get_xdata(), mh.get_ydata(), fmt % t[k]
          ))
    tics.extend(h2)
  return dict(
    lines = h,
    minor = minor,
    text = txt,
    major = tics )

def fillnans( dat ):
  """
  Interpolate all nan entries along the first index of the array

  NOTE:
    first and last entries MUST NOT be nan
  """
  dat = dat.reshape(dat.shape[0],prod(dat.shape[1:]))
  for k in xrange(dat.shape[1]):
    d = dat[:,k]
    n = isnan(d)
    idx = find(n)
    d[idx] = interp(idx,find(~n),d[~n])

def itermatch( idx, dat, match=None, withKey = False ):
  """
  Iterate data items with matching indices.
  INPUT:
    idx -- sequence of len N -- N sorted iterables
    dat -- sequence of len N -- N iterables with 'data'
    match -- callable -- vectorized function:
      key,keys --> bool array of len(keys)
    withKey -- bool -- set to emit key together with data

  OUTPUT:
    yields tuples from the N dat iterables whose idx values match.
  """
  idx = map(iter,idx)
  dat = map(iter,dat)
  t = array([ ti.next() for ti in idx ])
  d = [ di.next() for di in dat ]
  if match is None:
    match = lambda k,lst : all(lst==k)
  while True:
    tmi = min(t)
    if match(tmi,t):
      if withKey:
        yield tmi,tuple(d)
      else:
        yield tuple(d)
      t = array([ ti.next() for ti in idx ])
      d = [ di.next() for di in dat ]
    for k in xrange(len(t)):
      if not match(tmi,t[k]):
        continue
      t[k] = idx[k].next()
      d[k] = dat[k].next()
  return
