# -*- coding: utf-8 -*-


# -----------------------------------------------------------------------------
#
# importation of libraries
#
# ----------------------------------------------------------------------------- 
from scipy.interpolate import interp1d      # one-dimensional interpolation
from scipy.interpolate import interp2d      # two-dimensional interpolation
from scipy.integrate import simps           # simpson's rule for integration

import numpy as np                          # array manipulation
import numpy.ma as ma                       # masked arrays

from math import sin                        # sine operation on scalars
from math import cos                        # cosine operation on scalars
from math import exp                        # exponential on scalars    
from math import sqrt                       # square root on scalars
from math import acos                       # acos on scalars

from numba import jit                       # just-in-time compilation

#------------------------------------------------------------------------------
#
# external jitted functions
#
#------------------------------------------------------------------------------
@jit(nopython=True)
def f(cx, cy, x, y, t, N, DT):
    """ fills Cx and Cy according to the prescribed relative coordinates """
    
    NS = 100        # number of subdivisions in each sector
    DP = DT/NS      # subdivision width
    p = 0.00        # subdivision angle
    
    # loop through control points:
    for j in range(N):
        
        # loop through sectors:
        for i in range(N):
            cx[j, i] = 0.00     # set each element as an accumulator
            cy[j, i] = 0.00     # set each element as an accumulator
            
            # loop through subdivisions:
            for k in range(NS+1):
                # subdivision angle:
                p = t[i] - (1.0/2.0)*DT + k*DP 
                # numerator and denominator for cx:
                above =  -(x[j] + sin(p))*sin(p) + (y[j] - cos(p))*cos(p)
                below = (x[j] + sin(p))**2 + (y[j] - cos(p))**2
                cx[j, i] += (above/below)*DP*(1/(-2.0*np.pi))
                # numerator for cy:
                above =  -(x[j] + sin(p))*cos(p) - (y[j] - cos(p))*sin(p)
                cy[j, i] += (above/below)*DP*(1/(-2.0*np.pi))

@jit(nopython=True)
def jit_get_wx(x, y, phi, qn, wxgrid):
    """ returns a flattened array of wx due to the current turbine """
    
    NGP = x.size        # number of flattened grid points

    for j in range(NGP):
        # x & y are arrays!
        # up: numerator of quotient (array of size NS)
        # dw: denominator of quotient (array of size NS)
        # integrand: array of size NS
        up = -(x[j]+np.sin(phi))*np.sin(phi) + (y[j]-np.cos(phi))*np.cos(phi)
        dw = (x[j]+np.sin(phi))**2 + (y[j]-np.cos(phi))**2
        f = qn*(up/dw)                  # array to be integrated
        h = np.diff(phi)[0]             # difference
        n = f.size - 1                  # number of intervals
        s = (h/3)*(f[0] + 2*np.sum(f[2:n:2]) + 4*np.sum(f[1:n:2]) + f[n])
        wxgrid[j] = s
        
    wxgrid = -(1/(2*np.pi))*wxgrid
    
    return wxgrid

@jit(nopython=True)
def jit_get_wy(x, y, phi, qn, wygrid):
    """ returns a flattened array of wx due to the current turbine """
    
    NGP = x.size        # number of flattened grid points
    
    for j in range(NGP):
        # x & y are arrays!
        # up: numerator of quotient (array of size NS)
        # dw: denominator of quotient (array of size NS)
        # integrand: array of size NS
        up = -(x[j]+np.sin(phi))*np.cos(phi) - (y[j]-np.cos(phi))*np.sin(phi)
        dw = (x[j]+np.sin(phi))**2 + (y[j]-np.cos(phi))**2
        f = qn*(up/dw)                  # array to be integrated
        h = np.diff(phi)[0]             # difference
        n = f.size - 1                  # number of intervals
        s = (h/3)*(f[0] + 2*np.sum(f[2:n:2]) + 4*np.sum(f[1:n:2]) + f[n])
        wygrid[j] = s
        
    wygrid = -(1/(2*np.pi))*wygrid
    
    return wygrid
    
# -----------------------------------------------------------------------------
#
# turbine class, constructor, class methods and instance methods
#
# -----------------------------------------------------------------------------
class Turbine(object):
    """ parameters common to all turbines """
    
    N = 36              # number of control points
    F = 1.01            # outward offset factor for the control points
    E = 1e-2            # relative tolerance for convergence
    K = 0.25            # under-relaxation factor
    DT = 0.00           # slice width in radians
    PI = np.pi          # pi constant
    
    TSR = 0.00          # tip-speed ratio
    RPM = 0.00          # revolutions per minute
    RAD = 0.00          # turbine's radius in meters
    BLA = 0             # number of blades
    CHR = 0.00          # blade's chord
    SEC = 0             # wing section number
    
    NT = 0              # number of turbines
    
    SOL = 0.00          # turbine's solidity
    REG = 0.00          # global reynolds number
    NU = 1.5e-5         # kinematic viscosity (default is air)
             
    def __init__(self, x0, y0, k):
        """ instance constructor """
        
        # x0: turbine's x center coordinate
        # y0: turbine's y center coordinate
        #  k: turbine's rotation vector
        self.x0 = x0
        self.y0 = y0
        self.k = k
        
        """ call other initialization methods """
        self.set_global_coordinates()
        self.set_influence_matrices()
        self.set_velocities()
        self.set_sign()
    
    @classmethod
    def set_parameters(cls):
        """ load parameters common to all turbines """
        """ EXECUTE THIS FUNCTION IN THE FARM CLASS! """
        cls.TSR, cls.RPM, cls.RAD, cls.BLA, cls.CHR, cls.SEC, cls.NT = \
        np.loadtxt('settings.csv', delimiter=',', skiprows=1, unpack=True)
    
    @classmethod
    def set_constants(cls):
        """ solidity, global reynolds and slice width are initialized """
        """ EXECUTE THIS FUNCTION IN THE FARM CLASS! """
        cls.SOL = (cls.BLA*cls.CHR)/(2.0*cls.RAD)
        cls.REG = (cls.RPM*cls.RAD*cls.CHR*cls.PI)/(30*cls.NU)
        cls.DT = 2*cls.PI/cls.N
        
    @classmethod
    def set_coefficients(cls):
        """import lift and drag coefficients from available airfoils"""
        """ EXECUTE THIS FUNCTION IN THE FARM CLASS! """

        # select file names according to airfoil section:
        if cls.SEC == 1:
            clfile, cdfile = 'naca0012cl.csv', 'naca0012cd.csv'
        elif cls.SEC == 2:
            clfile, cdfile = 'naca0015cl.csv', 'naca0015cd.csv'
        elif cls.SEC == 3:
            clfile, cdfile = 'naca0018cl.csv', 'naca0018cd.csv'
        elif cls.SEC == 4:
            clfile, cdfile = 'naca0021cl.csv', 'naca0021cd.csv'
        elif cls.SEC == 5:
            clfile, cdfile = 'du06w200cl.csv', 'du06w200cd.csv'
        else:
            raise Exception('Input error: invalid airfoil section number!')
            
        # load arrays of coefficients:
        CL = np.loadtxt(clfile, delimiter=',')
        CD = np.loadtxt(cdfile, delimiter=',')

        # angle of attack and reynolds tables:
        if cls.SEC != 5:
            AA = np.loadtxt('nacaaa.csv', unpack=True)
            RE = np.loadtxt('nacare.csv', unpack=True)
        else:
            AA = np.loadtxt('du06w200aa.csv', unpack=True)
            RE = np.loadtxt('du06w200re.csv', unpack=True)        
        
        # create functions for lift and drag coefficients:
        fCL = interp2d(RE, AA, CL, kind='cubic')
        fCD = interp2d(RE, AA, CD, kind='cubic')
        
        # vectorize lift and drag functions:
        cls.v_fCL, cls.v_fCD = np.vectorize(fCL), np.vectorize(fCD) 
        
    @classmethod
    def set_control_points(cls):
        """ creates an array of angular coordinates """
        """ EXECUTE THIS FUNCTION IN THE FARM CLASS! """
        
        # t: consists of N points
        # p: is a finer grid made from t (default is 10 times N points)
        cls.t = np.fromfunction(lambda j : (1.0/2.0 + j)*cls.DT, (cls.N, ))
        cls.p = np.linspace(cls.t[0], cls.t[-1], num=10*cls.N)
    
    def get_cl(self, re, aa):
        """ returns an array of lift coefficients """
        return Turbine.v_fCL(re, np.degrees(aa))

    def get_cd(self, re, aa):
        """ returns an array of drag coefficients """
        return Turbine.v_fCD(re, np.degrees(aa))
    
    def set_global_coordinates(self):
        """ sets the global dimensionless coordinates of the control points """

        # alias:
        F = Turbine.F
        t = Turbine.t
        
        self.x = -F*np.sin(t) + self.x0
        self.y = +F*np.cos(t) + self.y0 
        
    def set_relative_coordinates(self, *turbines):
        """ sets the relative coordinates of the control points"""
        """ EXECUTE THIS FUNCTION IN THE FARM CLASS! """
        self.xi = []
        self.yi = []

        # alias:
        NT = Turbine.NT
        N = Turbine.N
        
        # append an empty array of N garbage elements:
        for i in range(int(NT)):
            self.xi.append(np.zeros((N, ), dtype=float))
            self.yi.append(np.zeros((N, ), dtype=float))
            
        # fill each element of the list with the relative coordinates respect 
        # to the i-th turbine:
        for i, turbine in enumerate(turbines):
            self.xi[i] = self.x - turbine.x0
            self.yi[i] = self.y - turbine.y0
            
    def set_influence_matrices(self):
        """ initializes the influence matrices Cx and Cy """
        self.cx = []
        self.cy = []

        # alias:
        NT = Turbine.NT
        N = Turbine.N
        
        for i in range(int(NT)):
            self.cx.append(np.zeros((N, N), dtype=float))
            self.cy.append(np.zeros((N, N), dtype=float))
          
    def fill_influence_matrices(self):
        """ fillls the current turbine's influence matrices, each one with
            respect to the i-th turbine """
        """ EXECUTE THIS FUNCTION IN THE FARM CLASS! """
        
        # import the jiited function from outside the class:
        global f

        # alias:
        NT = Turbine.NT
        N = Turbine.N
        t = Turbine.t
        DT = Turbine.DT

        for i in range(int(NT)):
            f(self.cx[i], self.cy[i], self.xi[i], self.yi[i], t, N, DT)
     
    @classmethod
    def set_own_wake(cls):
        """ fills the wake terms coming from the current matrix """
        """ EXECUTE THIS FUNCTION IN THE FARM CLASS! """

        # alias:
        N = Turbine.N
        F = Turbine.F
        t = Turbine.t
        
        cls.wn = np.zeros((N, N), dtype=float)
        cls.wt = np.zeros((N, N), dtype=float)
        
        # fill the normal loads wake matrix (lower half):
        right_index = (np.arange(N/2, N)).astype(int)
        left_index = (np.arange(N/2 - 1, -1, -1)).astype(int)
        cls.wn[right_index, right_index] = 1.00
        cls.wn[right_index, left_index] = -1.00
        
        # fill the tangential loads wake matrix (lower half):
        Y = F*np.cos(t[N//2+1: N-1]) 
        cls.wt[right_index[1:-1], right_index[1:-1]] = -Y/np.sqrt(1.0 - Y**2)
        cls.wt[right_index[1:-1], left_index[1:-1]] = -Y/np.sqrt(1.0 - Y**2)
        
        # the [1:-1] means that the head and tail are omitted due to the fact
        # that Y would yield values greater than 1 in the poles, thereby 
        # leading to singularities (-Y/(1 - Y^2)^(1/2)).
    
    @classmethod
    def set_auxiliaries(cls):
        """ zero N*N auxiliary matrix and N*N template matrix """
        """ EXECUTE THIS FUNCTION IN THE FARM CLASS! """

        # alias:
        N = Turbine.N
        
        # zeros: a template for Qt.
        # diags: a template for Qn. It represents a matrix containing all of 
        # the wake terms from some other turbine blocking partially or
        # totally the current turbine. 
        cls.zeros = np.zeros((N, N), dtype=float)
        cls.diags = np.zeros((N, N), dtype=float)
        
        # auxiliary indices:
        cls.row_index = np.linspace(N - 1, 0, num=N, dtype=int)
        cls.col_index = np.linspace(0, N - 1, num=N, dtype=int)
        
        # cross consisting of [-1, -1, ..., ..., +1, +1]:
        cross = np.hstack((np.full((N//2,), -1), np.full((N//2,), 1)))
        np.fill_diagonal(cls.diags, cross)

        # assign cross to transposed diagonal:
        cls.diags[cls.row_index, cls.col_index] = cross   
        
    def set_templates(self, other):
        """ computes the templates for the full obstruction matrices """
        self.fullQn = np.copy(Turbine.diags)
        self.fullQt = np.copy(Turbine.zeros)
        
        # masking the y-relative coordinates with respect to the other turbine:
        y = ma.masked_where((other <= -1.0) | (other >= 1.0), other)
        y = y.filled(0.00)
        cross = -y/np.sqrt(1.0 - y**2)

        # fill the Qt template matrix:
        np.fill_diagonal(self.fullQt, cross)

        # assign cross to transpoed diagonal:
        self.fullQt[Turbine.row_index, Turbine.col_index] = cross
        
    def offset_templates(self, i, turbine):
        """ fills the obstruction matrix with respect to the n-th turbine """

        # alias:
        N = Turbine.N
        
        # new matrices' rows will be either pushed inside or outside.
        self.newQn = np.copy(Turbine.zeros)
        self.newQt = np.copy(Turbine.zeros)
        
        if self.y0 < turbine.y0:
            # differences between the top coordinate point and the other 
            # turbine's upwind coordinate points:
            diff = np.abs(self.yi[i][0] - turbine.yi[i][0:N//2])
            # displacement given in number of control points:
            s = int((np.argwhere(diff <= np.min(diff)))[0])
            self.do = 'push outside'
            
        if self.y0 > turbine.y0:
            # differences between the bottom coordinate point and the other 
            # turbine's upwind coordinate points:
            diff = np.abs(self.yi[i][N//2 - 1] - turbine.yi[i][0:N//2])
            # displacement given in number of control points:
            s = N//2 - 1 - int((np.argwhere(diff <= np.min(diff)))[0])
            self.do = 'push inside'
            
        elif self.y0 == turbine.y0:
            s = 0
            self.do = 'push not'
            
        # offset matrices according to the displacement s:
        self.nodisplacements.append(s)
        self.procedures.append(self.do)
        
        if self.do == 'push outside':
            self.newQn[0:N//2-s, :] = self.fullQn[0+s:N//2, :]
            self.newQn[N//2+s:N, :] = self.fullQn[N//2:N-s, :]
            self.newQt[0:N//2-s, :] = self.fullQt[0+s:N//2, :]
            self.newQt[N//2+s:N, :] = self.fullQt[N//2:N-s, :]
            
        elif self.do == 'push inside':
            self.newQn[0+s:N//2, :] = self.fullQn[0:N//2-s, :]
            self.newQn[N//2:N-s, :] = self.fullQn[N//2+s:N, :]
            self.newQt[0+s:N//2, :] = self.fullQt[0:N//2-s, :]
            self.newQt[N//2:N-s, :] = self.fullQt[N//2+s:N, :]
            
        else:
            self.newQn = np.copy(self.fullQn)
            self.newQt = np.copy(self.fullQt)
        
    def other_wakes(self, current, *turbines):
        """ fills the wake terms coming from other turbines """
        """ EXECUTE THIS FUNCTION IN THE FARM CLASS! """
        self.nodisplacements = []
        self.procedures = []
        
        # blockage matrices:
        self.bn = []
        self.bt = []
        
        for i, turbine in enumerate(turbines):
            # append the own wake matrices when the current turbine is 
            # compared to itself:
            
            if i == current:
                self.bn.append(Turbine.wn)
                self.bt.append(Turbine.wt)
            elif i != current:
                # it is shadowed when at least one control point of the current
                # turbine lies in the direct wake of the i-th turbine.
                self.shadowed = np.any((self.yi[i]>=-1) & (self.yi[i]<=1))
                self.behind = self.x0 > turbine.x0
                
                if (self.shadowed and self.behind):
                    # compute obstruction matrices:
                    self.set_templates(self.yi[i])
                    self.offset_templates(i, turbine)
                    
                    # offsetted block matrices are appended to the list:
                    self.bn.append(self.newQn)
                    self.bt.append(self.newQt)
                else:
                    # add empty blockage matrices if there is no obstruction:
                    self.bn.append(np.copy(Turbine.zeros))
                    self.bt.append(np.copy(Turbine.zeros))
                    
    def set_contribution(self):
        """ current turbine contributing rows of X and Y """
        
        # WX = X*Q
        # WY = Y*Q
        # see how these are assembled in the Farm class.
        
        # x-row of big X:
        self.x = np.hstack((np.hstack((\
        [xi + ni for xi, ni in zip(self.cx, self.bn)]\
        )), np.hstack((\
        [yi + ti for yi, ti in zip(self.cy, self.bt)]\
        ))))
        
        # y-row of big Y:
        self.y = np.hstack((np.hstack((self.cy)), np.hstack((\
        [-xi for xi in self.cx]))))
        
    @classmethod
    def zero_vector(cls):
        """ makes an auxiliary N*1 column vector full of zeros """
        """ EXECUTE THIS FUNCTION IN THE FARM CLASS! """
        cls.wzero = np.zeros((Turbine.N, ), dtype=float)
        
    def set_velocities(self):
        """ sets the perturbation velocities to zero """
        self.wx = np.copy(Turbine.wzero)
        self.wy = np.copy(Turbine.wzero)
        
    def update_velocities(self, wx, wy):
        """ updates the perturbation velocities after one iteration """
        self.wx = wx
        self.wy = wy
        
    def set_sign(self):
        """ sets the sign due to either ccw or cw used in certain equations """
        # sign is just an auxiliary coefficient used in some equations:
        if self.k == 1: self.sign = -1
        if self.k == 0: self.sign = +1
        
    def loads(self):
        """ computes loads for each control point in the turbine """
        
        # the operations are done element-wise.
        
        # vx: dimensionless streamwise velocity
        # vy: dimensionless cross-stream velocity
        # vn: dimensionless normal velocity (normal to the chord)
        # vt: dimensionless tangential velocity (tangent to the chord)
        # vr: relative velocity
        # aa: angle of attack
        # re: local Reynolds number
        # cl: lift coefficient
        # cd: drag coefficient
        # cn: normal force coefficient
        # ct: tangent force coefficient
        # qn: blades' normal force coefficient (one-revolution average)
        # qt: blades' tangential force coefficient (one-revolution average)

        # alias:
        t = Turbine.t
        TSR = Turbine.TSR
        REG = Turbine.REG
        SOL = Turbine.SOL
        PI = Turbine.PI
        
        self.vx = 1.0 + self.wx                                     # vx
        self.vy = self.wy                                           # vy
        self.vn = self.vx*np.sin(t) - self.vy*np.cos(t)             # vn
        self.vt = -self.sign*self.vx*np.cos(t) - \
        self.sign*self.vy*np.sin(t) + TSR                           # vt       
        self.vr = np.sqrt(self.vn**2 + self.vt**2)                  # vr
        self.aa = np.arctan(self.vn/self.vt)                        # aa
        self.re = REG*self.vr/(TSR*1e6)                             # re
        self.cl = self.get_cl(self.re, self.aa)                     # cl
        self.cd = self.get_cd(self.re, self.aa)                     # cd
        self.cn = self.cl*np.cos(self.aa) + self.cd*np.sin(self.aa) # cn
        self.ct = self.cl*np.sin(self.aa) - self.cd*np.cos(self.aa) # ct
        self.qn = (SOL/(2.*PI))*(self.vr**2)*self.cn                # qn
        self.qt = self.sign*(SOL/(2.*PI))*(self.vr**2)*self.ct      # qt

    def correction(self):
        """ computes the correction factors for the perturbation velocities """
        
        # empirical coefficients:
        k3, k2, k1, k0 = 0.0892, 0.0544, 0.2511, -0.0017
        
        # thrust as a function of the azimuth angle and the loads:
        thrust = self.qn*np.sin(Turbine.t) + self.qt*np.cos(Turbine.t)
        
        # interpolator function for the thrust:
        function = interp1d(Turbine.t, thrust, kind='cubic')
        
        # vectorize the function so that it takes an array of angles:
        __function__ = np.vectorize(function)
        
        # thrust coefficient integrating according to phi:
        self.cth = simps(__function__(Turbine.p), Turbine.p)
        
        # induction factor:
        self.a = k3*self.cth**3 + k2*self.cth**2 + k1*self.cth + k0
        
        # correction factor:
        if self.a <= 0.15:
            self.ka = 1.0/(1.0 - self.a)
        else:
            self.ka = (1./(1 - self.a))*(0.65 + 0.35*exp(-4.5*(self.a - 0.15)))
            
    def power(self):
        """ self explanatory """

        # alias:
        TSR = Turbine.TSR
        p = Turbine.p
        
        function = interp1d(Turbine.t, self.qt, kind='cubic')
        __function__ = np.vectorize(function)
        self.cp = self.sign*TSR*simps(__function__(p), p)

    def smooth_loads(self):
        """ makes smooth functions for the Qn and Qt loads """
        self.fqn = interp1d(Turbine.t, self.qn, kind='cubic')

    def get_relative_grid(self, xgbl, ygbl):
        """ coordinates of the grid relative to the current turbine """
        self.rx = xgbl - self.x0
        self.ry = ygbl - self.y0

        # flatten the coordinates:
        self.rx = self.rx.ravel()
        self.ry = self.ry.ravel()



        
        
        

        

        
        
        
        
    
    

        
        





         
