# MAE 5500
# WIBERG, DERRICK

# the series solution to Prandtl’s lifting-line equation to
# numerically predict the lift and induced-drag coefficients for unswept elliptic and tapered wings
# with twist.
import math
import sys
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import json

def LL(filename):
    pi = np.pi
    # From thin airfoil theory, alpha_l0 = 0 and has therefore been omitted.

    # 1. - 2. Read json file for wing geometry and assign nodes with cosine clustering
    # Import geometry information
    json_string = open(filename).read()
    input_dict = json.loads(json_string)

    # Calculate input variables
    # Planform type and taper ratio
    plantype = input_dict["wing"]["planform"]["type"]
    elliptic = False
    tapered = False
    if plantype == "tapered":
        tapered = True
        Rt = float(input_dict["wing"]["planform"]["taper_ratio"])       # Taper Ratio
        Ra = int(input_dict["wing"]["planform"]["aspect_ratio"])  # Aspect Ratio
    elif plantype == "elliptic":
        elliptic = True
        Rt = 1                                                              # No taper
        Ra = int(input_dict["wing"]["planform"]["aspect_ratio"])        # Aspect Ratio
    else:
        planfile = input_dict["wing"]["planform"]["filename"]
        token = open(planfile, 'r')
        linestoken = token.readlines()
        Zplanf = []
        cplanf = []

        for x in linestoken:
            Zplanf.append(x.split()[0])
            cplanf.append(x.split()[1])

        token.close()
        # Convert each list of strings into lists of floats
        Zplan = []
        cplan = []
        for x in range(len(Zplanf)-1):
            Zplan.append(float(Zplanf[x+1]))
            cplan.append(float(cplanf[x+1]))

        n = len(Zplan)
        if cplan[n-1] == 0:
            cplan[n-1] = 0.001

        # Integrate over planform to find Aspect Ratio
        area = 0
        for i in range(n-1):
            z2 = Zplan[i+1]
            z1 = Zplan[i]
            c2 = cplan[i+1]
            c1 = cplan[i]
            area = area + (z2-z1)*(0.5*(c2+c1))
        Ra = 1 / (2*area)
        print("Ra is", Ra)


    CLat = float(input_dict["wing"]["airfoil_lift_slope"])           # Section Lift slope

    nodespersemi = int(input_dict["wing"]["nodes_per_semispan"])
    N = nodespersemi * 2 - 1                                             # Number of nodes

    # Ailerons
    zb1 = float(input_dict["wing"]["aileron"]["begin[z/b]"])
    zb2 = float(input_dict["wing"]["aileron"]["end[z/b]"])
    cf1 = float(input_dict["wing"]["aileron"]["begin[cf/c]"])
    cf2 = float(input_dict["wing"]["aileron"]["end[cf/c]"])
    hinge = float(input_dict["wing"]["aileron"]["hinge_efficiency"])
    defl = 1.0              # Assume deflectin efficiency of 1.0

    # Condition
    alpha = input_dict["condition"]["alpha_root[deg]"]           # Angle of attack or CL
    if alpha == 'CL':                                               # AoA will be back calculated from operating CL
        L = 21
        CL = np.zeros((L, 1))
        for i in range(L):
            CL[i] = -1.0 + i*0.1
        GivenCL = True
    else:
        alpha = float(alpha)
        alpha = alpha * np.pi / 180                                         # Convert to radians
        GivenCL = False
    delta = float(input_dict["condition"]["aileron_deflection[deg]"])           # Angle of attack (degrees)
    delta = delta * np.pi / 180                                         # Convert to radians
    pbar = input_dict["condition"]["pbar"]

    # Washout Distribution
    WashDist = input_dict["wing"]["washout"]["distribution"]
    if WashDist == "linear":
        linear = True
        twist = True
        optimum = False
    elif WashDist == "optimum":
        optimum = True
        twist = True
        linear = False
    elif WashDist == "none":
        linear = False
        optimum = False
        twist = False
    else:
        print("ERROR, please define washout distribution using available options")
        sys.exit()


    # Locate first, last and intermediate sections using cosine clustering
    theta = np.zeros((N, 1))
    Cos = np.zeros((N, 1))
    omega = np.zeros((N, 1))
    zb = np.zeros((N, 1))
    cb = np.zeros((N, 1))
    for i in range(N):
        theta[i] = i*pi / (N-1)
        Cos[i] = np.cos(theta[i])
        zb[i] = -0.5*np.cos(theta[i])

        # c/b values
        if elliptic:
            cb[i] = (4 / (pi*Ra)) * np.sin(theta[i])
            if cb[i] == 0:
                cb[i] = 0.001
        elif tapered:
            cb[i] = (2 / (Ra * (1 + Rt))) * (1 - (1 - Rt) * abs(np.cos(theta[i])))
            if cb[i] == 0:
                cb[i] = 0.001
        else:
            for j in range(n):
                if Zplan[j] > abs(zb[i]):
                    upperz = Zplan[j]
                    lowerz = Zplan[j-1]
                    upperc = cplan[j]
                    lowerc = cplan[j-1]
                    cb[i] = ((upperc-lowerc)/(upperz-lowerz))*(abs(zb[i]) - lowerz) + lowerc
                    break
                if Zplan[j] == abs(zb[i]):
                    cb[i] = cplan[j]
                    break


    for i in range(N):
        # Washout distribution
        if twist:
            if linear:
                omega[i] = abs(np.cos(theta[i]))
            elif optimum:
                if elliptic or tapered:
                    omega[i] = 1 - (np.sin(theta[i]) / (cb[i] / cb[nodespersemi-1]))
                else:
                    omega[i] = 1 - (np.sin(theta[i]) / (cb[i] / cplan[0]))


    print("omega is", omega)

    cb34 = np.zeros((N, 1))                                 # calculates y-coords of bottom planform
    if elliptic:
        cb1 = (4 / (pi * Ra)) * np.sqrt(1 - (2 * zb1) ** 2)
        cb2 = (4 / (pi * Ra)) * np.sqrt(1 - (2 * zb2) ** 2)
    elif tapered:
        cb1 = (2 / (Ra * (1 + Rt))) * (1 - (1 - Rt) * 2*zb1)
        cb2 = (2 / (Ra * (1 + Rt))) * (1 - (1 - Rt) * 2*zb2)
    else:
        for j in range(n):
            if Zplan[j] > abs(zb1):
                upperz = Zplan[j]
                lowerz = Zplan[j - 1]
                upperc = cplan[j]
                lowerc = cplan[j - 1]
                cb1 = ((upperc - lowerc) / (upperz - lowerz)) * (abs(zb1) - lowerz) + lowerc
                break
        for j in range(n):
            if Zplan[j] > abs(zb2):
                upperz = Zplan[j]
                lowerz = Zplan[j - 1]
                upperc = cplan[j]
                lowerc = cplan[j - 1]
                cb2 = ((upperc - lowerc) / (upperz - lowerz)) * (abs(zb2) - lowerz) + lowerc
                break

        # creates arrays of y-coords for the aileron
    cbout = np.zeros((2, 1))
    cbout[0] = cb2 * -0.75          # bottom  outer aileron y-coord
    cbout[1] = cbout[0] + cf2 * cb2     # top  outer aileron y-coord

    cbin = np.zeros((2, 1))
    cbin[0] = cb1 * -0.75           # bottom  inner aileron y-coord
    cbin[1] = cbin[0] + cf1 * cb1       # top inner aileron y-coord

    ycord = np.zeros((N, 1))            # for interpolating along hinge
    cfc = np.zeros((N, 1))              # local cf/c value
    thetaf = np.zeros((N, 1))           # local theta_f value
    efi = np.zeros((N, 1))              # local ideal efficiency
    chi_theta = np.zeros((N, 1))
    m = ((cbout[1] - cbin[1]) / (zb2 - zb1))            # slope of hinge line
    b = zb1*-m+cbin[1]                                  # y-intcpt of hinge line
    for i in range(N):
        if -zb2 < zb[i] < -zb1 or zb1 < zb[i] < zb2:
            cb34[i] = cb[i] * -0.75     # c/b 3 quarters below quarterchord line
            if zb[i] < 0:
                ycord[i] = -m * zb[i] + b
            else:
                ycord[i] = m * zb[i] + b
            cfc[i] = (ycord[i] - cb34[i]) / cb[i]
            thetaf[i] = np.arccos(2*cfc[i] - 1)
            efi[i] = 1 - (thetaf[i] - np.sin(thetaf[i])) / pi
            if zb[i] > 0:                               # Account for opposite sign for right aileron
                chi_theta[i] = -efi[i] * hinge * defl        # Assumes deflection efficiency = 1
            else:
                chi_theta[i] = efi[i] * hinge * defl

    # Calculate C matrix (system of equations)
    C = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == 0:
                C[i, j] = (j+1)**2
            elif i == N-1:
                C[i, j] = ((-1)**j)*((j+1)**2)
            else:
                C[i, j] = ((4/(CLat*cb[i]) + ((j+1)/math.sin(theta[i]))) * math.sin((j+1)*theta[i]))

    # Calculate inverse of C
    Cinv = inv(C)

    # Calculate a coefficients
    uni = np.ones((N, 1))

    a = np.matmul(Cinv, uni)

    # Calculate b coefficients
    # if twist:
    b = np.matmul(Cinv, omega)

    # Calculate c coefficients
    c = np.matmul(Cinv, chi_theta)

    # Calculate d coefficients
    d = np.matmul(Cinv, Cos)

    # Calculate change in rolling moment coefficient wrt aileron deflection (da) and rolling rate (p)
    Cl_da = float(-pi*Ra*c[1] / 4)

    Cl_pbar = float(-pi*Ra*d[1] / 4)


    # Calculate Pbar_steady:
    if pbar == "steady":
        pbar = -delta*(Cl_da / Cl_pbar)
    else:
        pbar = float(pbar)

    ##############################################
    # Calculate epsilon-omega

    epsilon = float(b[0] / a[0])

    # Calculate Kappa coefficients
    KL = float((1-(1+pi*Ra/CLat)*a[0]) / ((1+pi*Ra/CLat)*a[0]))

    KD = 0
    for i in range(2, N+1):
        KD = KD + i*(a[i-1] ** 2) / (a[0] ** 2)
    KD = float(KD)
    if twist:
        KDL = 0
        for i in range(2, N+1):
            KDL = KDL + (i * a[i-1] / (a[0])) * ((b[i-1] / b[0]) - (a[i-1] / a[0]))
        KDL = float(2 * (b[0] / a[0]) * KDL)

        KDomega = 0
        for i in range(2, N+1):
            KDomega = KDomega + (i * ((b[i-1] / b[0]) - (a[i-1] / a[0])) ** 2)
        KDomega = float(KDomega * ((b[0] / a[0]) ** 2))

        KDnot = float(KD - (KDL ** 2) / (4 * KDomega))
    else:
        KDL = 0
        KDomega = 0

    # Calculate span efficiency factor
    e_s = 1 / (1+KD)

    # Calculate Lift slope
    if elliptic:
        CLa = CLat / (1 + (CLat / (pi * Ra)))
    else:
        CLa = CLat / ((1 + (CLat/(pi*Ra))) * (1+KL))

    ################### 4 #######################
    # Properties at design operating condition
    CL_design = float(input_dict["wing"]["washout"]["CL_design"])
    if twist:
        WashAmt = input_dict["wing"]["washout"]["amount[deg]"]
        Omega_opt_des = KDL * CL_design / (2 * KDomega * CLa)
        Omega = Omega_opt_des
    else:
        Omega = 0
    alpha = (CL_design/CLa) + epsilon*Omega
    print("Part 4: At Design Operating Condition of CL =", CL_design)
    print("Root angle of attack [deg]:", alpha*180/pi)
    print("Optimum Washout magnitude [deg]", Omega * 180/pi)


    # Calculate A Matrix
    A = np.zeros((N, 1))
    alpha = (float(input_dict["condition"]["CL"])/ CLa) + epsilon * Omega
    print("CL is ", float(input_dict["condition"]["CL"]))
    for i in range(N):
        A[i] = a[i] * alpha - b[i] * Omega + c[i] * delta + d[i] * pbar  # Eq. 1.8.50

    # Calculate circulation distribution
    Gamma = np.zeros((N, 1))
    for i in range(N):
        for j in range(N):
            Gamma[i] = Gamma[i] + A[j] * np.sin((j + 1) * theta[i])

    return [Gamma, zb]


[G1, zb] = LL(filename='ewing.json')
[G2, zb] = LL(filename='ewing2.json')
[G3, zb] = LL(filename='ewing3.json')

fig, ax = plt.subplots()
ax.plot(zb, G1, '-', color="k", label='rec, optimum twist at CL= 0.2, CL = 0.5')
ax.plot(zb, G2, '-', color="tab:red", label='rec, optimum twist at CL= 0.2, CL = 0.2')
ax.plot(zb, G3, '-', color="tab:blue", label='rec, optimum twist at CL= 0.2, CL = 0.0')
plt.xlabel('z/b')
plt.ylabel('Gamma')
plt.title('Circulation Distribution')
ax.legend(loc='upper right', frameon=False)
plt.show()













