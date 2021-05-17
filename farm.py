#------------------------------------------------------------------------------
#
# farm class, composition is employed by virtue of the turbine class
#
#------------------------------------------------------------------------------

from math import sin                        # sine operation on scalars
from math import cos                        # cosine operation on scalars
from math import exp                        # exponential on scalars    
from math import sqrt                       # square root on scalars
from math import acos                       # acos on scalars
import numpy as np                          # array manipulation

from turbine import Turbine                 # Turbine class
from turbine import jit_get_wx              # just-in-time function
from turbine import jit_get_wy              # just-in-time function

class Farm(object):
    
    def __init__(self):
        """ farm constructor based on coordinates """
        
        # empty array of Turbine objects (turbines):
        self.turbines = []

        # turbines' coordinates:
        # x0: dimensionless x coordinate of the center.
        # y0: dimensionless y coordinate of the center.
        # ro: rotation vector.

        # call initialization methods from the Turbine class:
        Turbine.set_parameters()
        Turbine.set_constants()
        Turbine.set_coefficients()
        Turbine.set_control_points()
        Turbine.set_own_wake()
        Turbine.set_auxiliaries()
        Turbine.zero_vector()
        
        # import coordinates:
        x0, y0, ro = np.loadtxt('coordinates.csv', delimiter=',', unpack=True)
        self.xc = x0
        self.yc = y0
        self.ro = ro
        
        # append a Turbine instance to the array of turbines:
        for i in range(int(Turbine.NT + 1)):
            self.turbines.append(Turbine(x0[i], y0[i], ro[i]))
            
        # delete dummy turbine:
        self.turbines.pop()
        self.xc = x0[:-1]
        self.yc = y0[:-1]
        self.ro = ro[:-1]     

        # fill various matrices:
        [t.set_relative_coordinates(*self.turbines) for t in self.turbines]
        [t.fill_influence_matrices() for t in self.turbines]
        [t.other_wakes(i, *self.turbines) for i, t in enumerate(self.turbines)]
        [t.set_contribution() for t in self.turbines]
        self.set_blocks(self.turbines)
        
    def set_blocks(self, turbines):
        """ make the X and Y matrices: wx = X*Q; wy = Y*Q """
        self.X = np.vstack([t.x for t in turbines])
        self.Y = np.vstack([t.y for t in turbines])
        
    def calculate(self):
        """ computes loads for all turbines """
        its = int(0)

        # alias:
        N = Turbine.N
        E = Turbine.E
        K = Turbine.K
        NT = Turbine.NT
        
        while True:
            # compute loads throughout each turbine's control points.
            # compute thrust, induction factor and correction factor.
            [t.loads() for t in self.turbines]
            [t.correction() for t in self.turbines]
            
            # column vector made of wx of all turbines.
            # column vector made of wy of all turbines.
            WX_OLD = np.hstack([t.wx for t in self.turbines])
            WY_OLD = np.hstack([t.wy for t in self.turbines])
            
            # column vector made of qn of all turbines.
            # column vector made of qt of all turbines.
            QN = np.hstack([t.qn for t in self.turbines])
            QT = np.hstack([t.qt for t in self.turbines])
            
            # column vector made by stacking QN an QT:
            Q = np.hstack((QN, QT))
            
            # new pile of wx perturbation velocities.
            # new pile of wy perturbation velocities.
            WX = self.X.dot(Q)
            WY = self.Y.dot(Q)
            
            # pile of correction factors, each segment contains N elements:
            KA = np.hstack(([np.full((N, ), t.ka) for t in self.turbines]))
            
            # apply correction factors for WX and WY:
            WX = KA*WX
            WY = KA*WY
            
            # all elements need to be close to the relative tolerance in order
            # for the method to converge:
            wx_condition = np.allclose(WX, WX_OLD, rtol=E)
            wy_condition = np.allclose(WY, WY_OLD, rtol=E)
            
            if (wx_condition and wy_condition) or (its > 100):
                break
            else:
                # update according to the relaxation factor:
                WX_OLD = K*WX + (1.0 - K)*WX_OLD
                WY_OLD = K*WY + (1.0 - K)*WY_OLD
                
                # slices of WX and WY:
                U = np.hsplit(WX_OLD, NT)
                V = np.hsplit(WY_OLD, NT)
                
                # update each turbine's perturbation velocities:
                [t.update_velocities(u,v) for t,u,v in zip(self.turbines,U,V)]
                
                # increase count:
                its += 1
        
        # compute power coefficient for each turbine:
        [t.power() for t in self.turbines]

        # return power coefficients:
        return [t.cp for t in self.turbines]
 
    def plots(self, num):
        """ show the various plots of a particular turbine """
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
        
        # create figure with 6 axes:
        fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(8,8), dpi=100)
        
        # setup:
        labels = [(r'$w_x$',r'$w_y$'), \
                  (r'$Q_n$',r'$Q_t$'), \
                  (r'$AOA$',r'$v_{r}$')]
        steps = [[0.1, 0.05], [0.05, 0.01], [5.0, 0.5]]
        r = self.turbines[num - 1]
        subs = [(r.wx, r.wy), (r.qn, r.qt), (np.degrees(r.aa), r.vr)]
        theta = np.degrees(Turbine.t)
        clr = 'midnightblue'
        
        # plot each variable:
        for i in range(3):
            for j in range(2):
                ax = axes[i, j]
                ax.plot(theta, subs[i][j], label=labels[i][j], c=clr, \
                        markerfacecolor='white', \
                        markeredgecolor=clr, \
                        marker='o')
                ax.legend()
                ax.set_title(labels[i][j])
                ax.set_xlabel(r'$\theta$')
                ax.set_ylabel(labels[i][j])
                ax.xaxis.grid(True)
                ax.yaxis.grid(True)
                ax.set_xticks([k for k in np.linspace(0, 360, 9)])
                ax.yaxis.set_major_locator(ticker.MultipleLocator(steps[i][j]))
                ax.set_facecolor('lavender')
                
        # plot to console:
        fig.tight_layout()
        plt.show()

    def maps(self):
        """ show the absolute velocity color maps """
        # set figure parameters:
        dx = max(self.xc) - min(self.xc)    # x width
        dy = max(self.yc) - min(self.yc)    # y width
        dl = max(dx, dy)                    # max(dx, dy)
        xm = max(self.xc) - dx/2            # x middle point
        ym = max(self.yc) - dy/2            # y middle point
        xlow = xm - dl/2 - 1.5              # x lower limit
        xhigh = xm + dl/2 + 1.5             # x higher limit
        ylow = ym - dl/2 - 1.5              # y lower limit
        yhigh = ym + dl/2 + 1.5             # y higher limit

        # global grid:
        POINTS = 100
        self.xspace = np.linspace(xlow, xhigh, POINTS)
        self.yspace = np.linspace(yhigh, ylow, POINTS)

        # mesh grid for x and y:
        xgbl, ygbl = np.meshgrid(self.xspace, self.yspace)

        # create the relative grid for each turbine:
        [t.get_relative_grid(xgbl, ygbl) for t in self.turbines]

        # create the functions for the loads of each turbine:
        [t.smooth_loads() for t in self.turbines]

        # function that gets the wake terms:
        
        def wake(rx, ry, fqn):
            """ rx & ry are the relative grid coordinates """
            # logical array conditioners:
            shadowed = (ry > -1.0) & (ry < 1.0)
            outside = np.sqrt(rx**2 + ry**2) > 1.0
            downwind = rx > 0.0

            def get_term(array):
                # unpacking:
                c1 = array[0]
                c2 = array[1]
                c3 = array[2]
                x = array[3]
                y = array[4]
                
                term = 0.
                # applying conditions:
                if c1 and c2 and c3:
                    term = -fqn(acos(y)) + fqn(2*np.pi - acos(y))
                elif c1 and not c2:
                    term = -fqn(acos(y))
                else:
                    term = 0.
                return term

            # piling:
            shadowed.resize(POINTS**2, 1)
            outside.resize(POINTS**2, 1)
            downwind.resize(POINTS**2, 1)
            rx.resize(POINTS**2, 1)
            ry.resize(POINTS**2, 1)

            array =  np.hstack((shadowed, outside, downwind, rx, ry))

            return np.apply_along_axis(get_term, 1, array)

        # alias:
        t = Turbine.t
        N = Turbine.N
        
        # phi array of angular values to be passed to functions:
        phi = np.linspace(t[0], t[-1], num=2*N + 1)

        # perturbation velocities flattened array initialization:
        wx_grid = np.zeros((POINTS**2, ), dtype=float)
        wy_grid = np.zeros((POINTS**2, ), dtype=float)

        # import the global jitted functions:
        global jit_get_wx
        global jit_get_wy
        
        # contribution to wx & wy from all of the turbines:
        for t in self.turbines:
            # vectorize the normal loads function:
            __fqn__ = np.vectorize(t.fqn)
            
            # take the whole phi array as an argument and retrieve an array:
            qn = __fqn__(phi)

            # add-up:
            arr = np.zeros((POINTS**2, ), dtype=float)
            wx_grid += jit_get_wx(t.rx, t.ry, phi, qn, arr)\
                       + wake(t.rx, t.ry, t.fqn)

            arr = np.zeros((POINTS**2, ), dtype=float)
            wy_grid += jit_get_wy(t.rx, t.ry, phi, qn, arr)

        # non-dimensional velocities:
        wx_grid += 1                                # vx = wx + 1
                                                    # vy = wy
        wz_grid = np.sqrt(wx_grid**2 + wy_grid**2)  # |v| = sqrt(vx^2+vy^2)

        # resize velocities before plotting them in 2D:
        wz_grid.resize(POINTS, POINTS)

        # image plots:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle

        # list of circles:
        circles = []
        for j in range(int(Turbine.NT)):
            col = 'blue' if (self.ro[j] == 1) else 'red'
            circles.append(Circle((self.xc[j], self.yc[j]),\
                                  1.0,\
                                  color=col,\
                                  fill=False))
                      
        # figure will contain the image plot and the turbines:
        fig = plt.figure(figsize=(6,6), dpi=300)               
        ax = fig.add_subplot(1,1,1)
        img = ax.imshow(wz_grid, interpolation='bicubic',\
                        extent=[self.xspace[0], self.xspace[-1],\
                                self.yspace[-1], self.yspace[0]],\
                        cmap='jet')
        ax.set_xlabel('$x/R$')
        ax.set_ylabel('$y/R$')
        ax.set_title('Módulo de velocidad adimensional')
        cbar = fig.colorbar(ax=ax, mappable=img, orientation='vertical')

        # add circles:
        [ax.add_artist(c) for c in circles]
        
        # adjust fontsize:
        # inches per circle radius:
        ipc = 6/(self.xspace[-1] - self.xspace[0])
        fsize = round(2*ipc*300/72)

        
        # add power coefficients:
        for j, t in enumerate(self.turbines):
            ax.text(self.xc[j] - 0.45, self.yc[j],\
                    '{:4.3f}'.format(t.cp),\
                    weight='bold',\
                    color='white',\
                    fontsize=fsize)
        plt.show()
