from numpy import *
from util import *
from shai_util import *
from scipy import signal
import warnings
import numpy as np

"""
The phaser module provides an implementation of the phase estimation algorithm
of "Estimating the phase of synchronized oscillators";
   S. Revzen & J. M. Guckenheimer; Phys. Rev. E; 2008, v. 78, pp. 051907
   doi: 10.1103/PhysRevE.78.051907

Phaser takes in multidimensional data from multiple experiments and fits the
parameters of the phase estimator, which may then be used on new data or the
training data. The output of Phaser is a phase estimate for each time sample
in the data. This phase estimate has several desirable properties, such as:
(1) d/dt Phase is approximately constant
(2) Phase estimates are robust to measurement errors in any one variable
(3) Phase estimates are robust to systematic changes in the measurement error

The top-level class of this module is Phaser.
An example is found in test_sincos(); it requires matplotlib
"""
print("Running phaser")
class ZScore( object ):
  """
  Class for finding z scores of given measurements with given or computed
  covarance matrix.

  This class implements equation (7) of [Revzen08]

  Properties:
    y0 -- Dx1 -- measurement mean
    M -- DxD -- measurement covariance matrix
    S -- DxD -- scoring matrix
  """

  def __init__( self, y = None, M = None ):
    """Computes the mean and scoring matrix of measurements
    INPUT:
      y -- DxN -- N measurements of a time series in D dimensions
      M -- DxD (optional) -- measurement error covariance for y
        -- If M is missing, it is assumed to be diagonal with variances
    -- given by 1/2 variance of the second order differences of y
    """

    # if M given --> use fromCovAndMean
    # elif we got y --> use fromData
    # else --> create empty object with None in members
    if M is not None:
      self.fromCovAndMean( mean(y, 1), M)
    elif y is not None:
      self.fromData( y )
    else:
      self.y0 = None
      self.M = None
      self.S = None
    

  def fromCovAndMean( self, y0, M ):
    """
    Compute scoring matrix based on square root of M through svd
    INPUT:
      y0 -- Dx1 -- mean of data
      M -- DxD -- measurement error covariance of data
    """
    self.y0 = y0
    self.M = M
    (D, V) = linalg.eig( M )
    self.S = dot( V.transpose(), diag( 1/sqrt( D ) ) )

  def fromData( self, y ):
    """
    Compute scoring matrix based on estimated covariance matrix of y
    Estimated covariance matrix is geiven by 1/2 variance of the second order
    differences of y
    INPUT:
      y -- DxN -- N measurements of a time series in D dimensions
    """
    self.y0 = mean( y, 1 )
    self.M = diag( std( diff( y, n=2, axis=1 ), axis=1 ) )
    self.S = diag( 1/sqrt( diag( self.M ) ) )

  def __call__( self, y ):
    """
    Callable wrapper for the class
    Calls self.zScore internally
    """
    return self.zScore( y )

  def zScore( self, y ):
    """Computes the z score of measurement y using stored mean and scoring
    matrix
    INPUT:
      y -- DxN -- N measurements of a time series in D dimensions
    OUTPUT:
      zscores for y -- DxN
    """
    return dot( self.S, y - self.y0.reshape( len( self.y0 ), 1 ) )


def _default_psf(x):
  """Default Poincare section function
     by rights, this should be inside the Phaser class, but pickle
     would barf on Phaser objects if they contained functions that
     aren't defined in the module top-level.
  """
  return signal.lfilter(
    array([0.02008336556421, 0.04016673112842,0.02008336556421] ),
    array([1.00000000000000,-1.56101807580072,0.64135153805756] ),
  x[0,:] )

class Phaser( object ):
  """
  Concrete class implementing a Phaser phase estimator

  Instance attributes:
    sc -- ZScore object for converting y to z-scores
    P_k -- list of D FourierSeries objects -- series correction for correcting proto-phases
    prj -- D x 1 complex -- projector on combined proto-phase
    P -- FourierSeries object -- series correction for combined phase
    psf -- callable -- callback to psecfun
  """

  def __init__( self, y = None, C = None, ordP = None, psecfunc = None ):
    """
    Initilizing/training a phaser object
    INPUT:
      y -- DxN or [ DxN_1, DxN_2, DxN_3, ... ] -- Measurements used for training
      C -- DxD (optional) -- Covariance matrix of measurements
      ordP -- 1x1 (optional) -- Orders of series to use in series correction
      psecfunc -- 1x1 (optional) -- Poincare section function
    """

    # if psecfunc given -> use given
    if psecfunc is not None:
      self.psf = psecfunc
    else:
      self.psf = _default_psf

    # if y given -> calls self.phaserTrain
    if y is not None:
      self.phaserTrain( y, C, ordP )

  def __call__( self, dat ):
    """
    Callable wrapper for the class. Calls phaserEval internally
    """
    return self.phaserEval( dat )


  def phaserEval( self, dat ):
    """
    Computes the phase of testing data
    INPUT:
      dat -- DxN -- Testing data whose phase is to be determined
    OUTPUT:
      Returns the complex phase of input data
    """

    # compute z score
    z = self.sc.zScore( dat )

    # compute Poincare section
    p0 = self.psf( dat )

    # compute protophase using Hilbert transform
    zeta = self.mangle * hilbert( z )
    z0, ido0 = Phaser.sliceN( zeta, p0 )

    # Compute phase offsets for proto-phases
    ofs = exp(-1j * angle(mean(z0, axis = 1)).T)

    # series correction for each dimision using self.P_k
    th = Phaser.angleUp( zeta * ofs[:,newaxis] )

    # evaluable weights based on sample length
    p = 1j * zeros( th.shape )
    for k in range( th.shape[0] ):
      p[k,:] = self.P_k[k].val( th[k,:] ).T + th[k,:]

    rho = mean( abs( zeta ), 1 ).reshape(( zeta.shape[0], 1 ))
    # compute phase projected onto first principal components using self.prj
    ph = Phaser.angleUp( dot( self.prj.T, vstack( [cos( p ) * rho, sin( p ) * rho] ) ))

    # return series correction of combined phase using self.P
    phi = real( ph + self.P.val( ph ).T )
    pOfs2 = (p0[ido0+1] * exp(1j * phi.T[ido0+1]) - p0[ido0] * exp(1j * phi.T[ido0] )) / (p0[ido0+1] - p0[ido0])
    return phi - angle(sum(pOfs2))

  def phaserTrain( self, y, C = None, ordP = None ):
    """
    Trains the phaser object with given data.
    INPUT:
      y -- DxN or [ DxN_1, DxN_2, DxN_3, ... ] -- Measurements used for training
      C -- DxD (optional) -- Covariance matrix of measurements
    """

    # if given one sample -> treat it as an ensemble with one element
    if y.__class__ is ndarray:
      y = [y]
    # Copy the list container
    y = [yi for yi in y]
    # check dimension agreement in ensemble
    if len( set( [ ele.shape[0] for ele in y ] ) ) is not 1:
      raise( Exception( 'newPhaser:dims','All datasets in the ensemble must have the same dimension' ) )
    D = y[0].shape[0]

    # train ZScore object based on the entire ensemble
    self.sc = ZScore( hstack( y ), C )

    # initializing proto phase variable
    zetas = []
    cycl = zeros( len( y ))
    svm = 1j*zeros( (D, len( y )) )
    svv = zeros( (D, len( y )) )

    # compute protophases for each sample in the ensemble
    for k in range( len( y ) ):
      # hilbert transform the sample's z score
      zetas.append( hilbert( self.sc.zScore( y[k] ) ) )

      # trim beginning and end cycles, and check for cycle freq and quantity
      cycl[k], zetas[k], y[k] = Phaser.trimCycle( zetas[k], y[k] )

      # Computing the Poincare section
      sk = self.psf( y[k] )
      (sv, idx) = Phaser.sliceN( zetas[k], sk )
      if idx.shape[-1] == 0:
        raise Exception( 'newPhaser:emptySection', 'Poincare section is empty -- bailing out' )

      svm[:,k] = mean( sv, 1 )
      svv[:,k] = var( sv, 1 ) * sv.shape[1] / (sv.shape[1] - 1)


    # computing phase offset based on psecfunc
    self.mangle, ofs = Phaser.computeOffset( svm, svv )

    # correcting phase offset for proto phase and compute weights
    wgt = zeros( len( y ) )
    rho_i = zeros(( len( y ), y[0].shape[0] ))
    for k in range( len( y ) ):
      zetas[k] = self.mangle * exp( -1j * ofs[k] ) * zetas[k]
      wgt[k] = zetas[k].shape[0]
      rho_i[k,:] = mean( abs( zetas[k] ), 1 )

    # compute normalized weight for each dimension using weights from all samples
    wgt = wgt.reshape(( 1, len( y )))
    rho = ( dot( wgt, rho_i ) / sum( wgt ) ).T
    # if ordP is None -> use high enough order to reach Nyquist/2
    if ordP is None:
      ordP = ceil( max( cycl ) / 4 )

    # correct protophase using seriesCorrection
    self.P_k = Phaser.seriesCorrection( zetas, ordP )


    # loop over all samples of the ensemble
    q = []
    for k in range( len( zetas ) ):
      # compute protophase angle
      th = Phaser.angleUp( zetas[k] )

      phi_k = 1j * ones( th.shape )

      # loop over all dimensions
      for ki in range( th.shape[0] ):
        # compute corrected phase based on protophase
        phi_k[ki,:] = self.P_k[ki].val( th[ki,:] ).T + th[ki,:]

      # computer vectorized phase
      q.append( vstack( [cos( phi_k ) * rho, sin( phi_k ) * rho] ) )

    # project phase vectors using first two principal components
    W = hstack( q[:] )
    W = W - mean( W, 1 )[:,newaxis]
    pc = svd( W, False )[0]
    self.prj = reshape( pc[:,0] + 1j * pc[:,1], ( pc.shape[0], 1 ) )

    # Series correction of combined phase
    qz = []
    for k in range( len( q ) ):
      qz.append( dot( self.prj.T, q[k] ) )

    # store object members for the phase estimator
    self.P = Phaser.seriesCorrection( qz, ordP )[0]

  def computeOffset( svm, svv ):
    """
    """
    # convert variances into weights
    svv = svv / sum( svv, 1 ).reshape( svv.shape[0], 1 )

    # compute variance weighted average of phasors on cross section to give the phase offset of each protophase
    mangle = sum( svm * svv, 1)
    if any( abs( mangle ) ) < .1:
      b = np.where( abs( mangle ) < .1 )
      raise Exception( 'computeOffset:badmeasureOfs', len( b ) + ' measuremt(s), including ' + b[0] + ' are too noisy on Poincare section' )

    # compute phase offsets for trials
    mangle = conj( mangle ) / abs( mangle )
    mangle = mangle.reshape(( len( mangle ), 1))
    svm = mangle * svm
    ofs = mean( svm, 0 )
    if any( abs( ofs ) < .1 ):
      b = np.where( abs( ofs ) < .1 )
      raise Exception( 'computeOffset:badTrialOfs', len( b ) + ' trial(s), including ' + b[0] + ' are too noisy on Poincare section' )

    return mangle, angle( ofs )

  computeOffset = staticmethod( computeOffset )

  def sliceN( x, s, h = None ):
    """
    Slices a D-dimensional time series at a surface
    INPUT:
      x -- DxN -- data with colums being points in the time series
      s -- N, array -- values of function that is zero and increasing on surface
      h -- 1x1 (optional) -- threshold for transitions, transitions>h are ignored
    OUPUT:
      slc -- DxM -- positions at estimated value of s==0
      idx -- M -- indices into columns of x indicating the last point before crossing the surface
    """

    # checking for dimension agreement
    if x.shape[1] != s.shape[0]:
      raise Exception( 'sliceN:mismatch', 'Slice series must have matching columns with data' )
    #print(s)
    #print(s.shape)
    idx = np.where(( s[1:] > 0 ) & ( s[0:-1] <= 0 )) #changed here
    #print(idx[0])
    #print(x.shape[1])
    indices = np.where(idx[0] < x.shape[1])
    #print(indices[0])
    #print(type(indices[0]))
    #ind = np.array(indices[0].int)
    #idx = idx[[indices[0]]] #changed here
    idx = np.take(idx, indices[0])
    #print(idx)


# array([4, 3, 6])
    

    if h is not None:
      idx = idx( abs( s[idx] ) < h & abs( s[idx+1] ) < h );

    N = x.shape[0]

    if len( idx ) is 0:
      return zeros(( N, 0 )), idx

    wBfr = abs( s[idx] )
    wBfr = wBfr.reshape((1, len( wBfr )))
    wAfr = abs( s[idx+1] )
    wAfr = wAfr.reshape((1, len( wAfr )))
    slc = ( x[:,idx]*wAfr + x[:,idx+1]*wBfr ) / ( wBfr + wAfr )

    return slc, idx

  sliceN = staticmethod( sliceN )

  def angleUp( zeta ):
    """
    Convert complex data to increasing phase angles
    INPUT:
      zeta -- DxN complex
    OUPUT:
      returns DxN phase angle of zeta
    """
    # unwind angles
    th = unwrap( angle ( zeta ) )

    # reverse decreasing sequences
    bad = th[:,0] > th[:,-1]
    if any( bad ):
      th[bad,:] = -th[bad,:]
    return th

  angleUp = staticmethod( angleUp )

  def trimCycle( zeta, y ):
    """
    """
    # compute wrapped angle for hilbert transform
    ph = Phaser.angleUp( zeta )

    # estimate nCyc in each dimension
    nCyc = abs( ph[:,-1] - ph[:,0] ) / 2 / pi
    cycl = int(ceil( zeta.shape[1] / max( nCyc ) ))

    # if nCyc < 7 -> warning
    # elif range(nCyc) > 2 -> warning
    # else truncate beginning and ending cycles
    if any( nCyc < 7 ):
      warnings.warn( "PhaserForSample:tooShort" )
    elif max( nCyc ) - min( nCyc ) > 2:
      warnings.warn( "PhaserForSample:nCycMismatch" )
    else:
      zeta = zeta[:,cycl:-cycl]
      y = y[:,cycl:-cycl]

    return cycl, zeta, y

  trimCycle = staticmethod( trimCycle )

  def seriesCorrection( zetas, ordP ):
    """
    Fourier series correction for data zetas up to order ordP
    INPUT:
      zetas -- [DxN_1, DxN_2, ...] -- list of D dimensional data to be corrected using Fourier series
      ordP -- 1x1 -- Number of Fourier modes to be used
    OUPUT:
      Returns a list of FourierSeries object fitted to zetas
    """

    # initialize proto phase series 2D list
    proto = []

    # loop over all samples of the ensemble
    wgt = zeros( len( zetas ) )
    for k in range( len( zetas ) ):
      proto.append([])
      # compute protophase angle (theta)
      zeta = zetas[k]
      N = zeta.shape[1]
      theta = Phaser.angleUp( zeta )

      # generate time variable
      t = linspace( 0, 1, N )
      # compute d_theta
      dTheta = diff( theta, 1 )
      # compute d_t
      dt = diff( t )
      # mid-sampling of protophase angle
      th = ( theta[:,1:] + theta[:,:-1] ) / 2.0

      # loop over all dimensions
      for ki in range( zeta.shape[0] ):
        # evaluate Fourier series for (d_theta/d_t)(theta)
        # normalize Fourier coefficients to a mean of 1
        fdThdt = FourierSeries().fit( ordP * 2, th[ki,:].reshape(( 1, th.shape[1])), dTheta[ki,:].reshape(( 1, dTheta.shape[1])) / dt )
        fdThdt.coef = fdThdt.coef / fdThdt.m
        fdThdt.m = array([1])

        # evaluate Fourier series for (d_t/d_theta)(theta) based on Fourier
        # approx of (d_theta/d_t)
        # normalize Fourier coefficients to a mean of 1
        fdtdTh = FourierSeries().fit( ordP, th[ki,:].reshape(( 1, th.shape[1])), 1 / fdThdt.val( th[ki,:].reshape(( 1, th.shape[1] )) ).T )
        fdtdTh.coef = fdtdTh.coef / fdtdTh.m
        fdtdTh.m = array([1])

        # evaluate corrected phsae phi(theta) series as symbolic integration of
        # (d_t/d_theta), this is off by a constant
        proto[k].append(fdtdTh.integrate())

      # compute sample weight based on sample length
      wgt[k] = zeta.shape[0]

    wgt = wgt / sum( wgt )

    # return phase estimation as weighted average of phase estimation of all samples
    proto_k = []
    for ki in range( zetas[0].shape[0] ):
      proto_k.append( FourierSeries.bigSum( [p[ki] for p in proto], wgt ))

    return proto_k

  seriesCorrection = staticmethod( seriesCorrection )

def test_sincos():
  """
  Simple test/demo of Phaser, recovering a sine and cosine

  Demo courtesy of Jimmy Sastra, U. Penn 2011
  """
  from numpy import sin,cos,pi,array,linspace,cumsum,asarray,dot,ones
  from pylab import plot, legend, axis, show, randint, randn, std,lstsq
  # create separate trials and store times and data
  dats=[]
  t0 = []
  period = 55 # i
  phaseNoise = 0.5/sqrt(period)
  snr = 20
  N = 10
  print(N,"trials with:")
  print("\tperiod %.2g"%period,"(samples)\n\tSNR %.2g"%snr,"\n\tphase noise %.2g"%phaseNoise,"(radian/cycle)")
  print("\tlength = [",)
  for li in xrange(N):
    l = randint(400,2000) # length of trial
    dt = pi*2.0/period + randn(l)*0.07 # create noisy time steps
    t = cumsum(dt)+randn()*2*pi # starting phase is random
    raw = asarray([sin(t),cos(t)]) # signal
    raw = raw + randn(*raw.shape)/snr # SNR=20 noise
    t0.append(t)
    dats.append( raw - raw.mean(axis=1)[:,newaxis] )
    print(l,)
  print("]")
  # use points where sin=cos as poincare section
  phr = Phaser( dats, psecfunc = lambda x : dot([1,-1],x) )
  phi = [ phr.phaserEval( d ) for d in dats ] # extract phase
  reg = array([linspace(0,1,t0[0].size),ones(t0[0].size)]).T
  tt = dot( reg, lstsq(reg,t0[0])[0] )
  plot(((tt-pi/4) % (2*pi))/pi-1, dats[0].T,'x')
  plot( (phi[0].T % (2*pi))/pi-1, dats[0].T,'.')#plot data versus phase
  legend(['sin(t)','cos(t)','sin(phi)','cos(phi)'])
  axis([-1,1,-1.2,1.2])
  show()

if __name__=="__main__":
  test_sincos()
