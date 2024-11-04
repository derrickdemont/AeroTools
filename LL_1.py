# MAE 5500
# WIBERG, DERRICK

# Beginner LL - no sweep, dihedral, geometric or aerodynamic twist
import math

import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import json

pi = np.pi
def LL(filename,alpha):
        # 1. - 2. Read json file for wing geometry and assign nodes with cosine clustering
        # Import geometry information
        #filename = input("Enter json filename:")
        json_string = open(filename).read()
        input_dict = json.loads(json_string)

        # Calculate input variables
        elliptic = False
        plantype = input_dict["wing"]["planform"]["type"]
        if plantype == "tapered":
            Rt = float(input_dict["wing"]["planform"]["taper_ratio"])
        else:
            elliptic = True
            Rt = 1

        Cla = float(input_dict["wing"]["airfoil_lift_slope"])           # Section Lift slope
        Ra = int(input_dict["wing"]["planform"]["aspect_ratio"])        # Aspect Ratio

        nodespersemi = int(input_dict["wing"]["nodes_per_semispan"])
        N = nodespersemi * 2 - 1                                             # Number of nodes


        # Locate first, last and intermediate sections
        theta = np.zeros((N, 1))
        zb = np.zeros((N, 1))
        cb = np.zeros((N, 1))
        for i in range(N):
            theta[i] = i*pi / (N-1)
            zb[i] = -0.5*np.cos(theta[i])
            if elliptic:
                cb[i] = (4 / (pi*Ra)) * np.sin(theta[i])
            else:
                cb[i] = (2 / (Ra * (1 + Rt))) * (1 - (1 - Rt) * abs(np.cos(theta[i])))


        # Calculate C matrix (system of equations)
        C = np.zeros((N, N))

        for i in range(N):
            for j in range(N):
                if i == 0:
                    C[i, j] = (j+1)**2
                elif i == N-1:
                    C[i, j] = ((-1)**j)*((j+1)**2)
                else:
                    C[i, j] = ((4/(Cla*cb[i]) + ((j+1)/math.sin(theta[i]))) * math.sin((j+1)*theta[i]))


        # Calculate inverse of C
        Cinv = inv(C)


        # Calculate a coefficients
        uni = np.ones((N, 1))
        a = np.matmul(Cinv, uni)

        # Calculate b coefficients
        omega = np.zeros((N, 1))
        for i in range(N):
            omega[i] = 1 - math.sin(theta[i]) / (1-(1-Rt)*abs(math.cos(theta[i])))

        b = np.matmul(Cinv, omega)

        # Calculate Kappa coefficients
        KL = float((1-(1+pi*Ra/Cla)*a[0]) / ((1+pi*Ra/Cla)*a[0]))

        KD = 0
        for i in range(2, N+1):
            KD = KD + i*(a[i-1] ** 2) / (a[0] ** 2)
        KD = float(KD)

        KDL = 0
        for i in range(2, N+1):
            KDL = KDL + (i * a[i-1] / (a[0])) * ((b[i-1] / b[0]) - (a[i-1] / a[0]))
        KDL = 2 * (b[0] / a[0]) * KDL

        KDomega = 0
        for i in range(2, N+1):
            KDomega = KDomega + (i * ((b[i-1] / b[0]) - (a[i-1] / a[0])) ** 2)
        KDomega = KDomega * ((b[0] / a[0]) ** 2)

        KDnot = KD - (KDL ** 2) / (4 * KDomega)


        # Calculate span efficiency factor
        e_s = 1 / (1+KD)

        # Calculate Lift slope
        if elliptic:
            CLa = Cla / (1 + (Cla / (pi * Ra)))
        else:
            CLa = Cla / ((1 + (Cla/(pi*Ra))) * (1+KL))


        # Calculate Lift and drag coefficients
        # alpha = float(input("Enter an Aoa (deg):"))
        alpha = alpha * np.pi / 180

        CL = CLa*alpha

        if elliptic:
            CDi = (CL ** 2) / (pi * Ra)
        else:
            CDi = (CL ** 2) / (pi*Ra*e_s)

        # 4. Return KL, KD, e_s, Cla, Cl, Cd
        # if not elliptic:
            # print("KL = ", KL)
            # print("KD = ", KD)
            # print("e_s (span efficiency factor) =", e_s)


        """
        # 5. Plot planform area
        cb0 = np.zeros((N, 1))
        cb14 = np.zeros((N, 1))
        cb34 = np.zeros((N, 1))
        
        for i in range(N):
            cb14[i] = cb[i] / 4         #c/b  quarterchord line
            cb34[i] = cb[i] * -0.75     #c/b 3 quarters below quarterchord line
        
        fig, ax = plt.subplots()
        ax.plot(zb, cb14, '-', color="k")
        ax.plot(zb, cb34, '-', color="k")
        ax.plot(zb, cb0, '-', color="k")
        
        # create vertical lines along planform where nodes are solved
        solx = np.zeros((2, 1))
        soly = np.zeros((2, 1))
        for i in range(N):
            solx[0] = zb[i]
            solx[1] = zb[i]
            soly[0] = cb14[i]
            soly[1] = cb34[i]
            ax.plot(solx, soly, '-', color="tab:blue")
        
        plt.axis("equal")
        plt.xlabel('z/b')
        plt.ylabel('c/b')
        plt.title('Planform Area')
        plt.show()
        """

        # Calculate circulation distribution
        Gamma = np.zeros((N, 1))
        for i in range(N):
            for j in range(N):
                Gamma[i] = Gamma[i] + a[j]*np.sin((j+1)*theta[i])
        Gamma = Gamma*alpha

        return [Gamma, zb]


[G1, zb] = LL(filename='bwing.json',alpha=5)
[G2, zb] = LL(filename='bwing.json',alpha=10)
[G3, zb] = LL(filename='taperwing.json',alpha=10)


fig, ax = plt.subplots()
ax.plot(zb, G1, '-', color="k", label='elliptic, a = 5')
ax.plot(zb, G2, '-', color="tab:blue", label='elliptic, a = 10')
ax.plot(zb, G3, '-', color="tab:red", label='tapered, a = 10')
plt.xlabel('z/b')
plt.ylabel('Gamma')
plt.title('Circulation Distribution')
ax.legend(loc='upper right', frameon=False)
plt.show()



