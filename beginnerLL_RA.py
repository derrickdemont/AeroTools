# MAE 5500
# WIBERG, DERRICK

# WING 1: Write a Python 3 computer program that uses the series solution to Prandtl’s lifting-line equation to
# numerically predict the lift and induced-drag coefficients for unswept elliptic and tapered wings having
# no geometric or aerodynamic twist.
import math

import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import json

pi = np.pi
# 1. - 2. Read json file for wing geometry and assign nodes with cosine clustering
# Import geometry information
filename = input("Enter json filename:")
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

# Ra = int(input_dict["wing"]["planform"]["aspect_ratio"])        # Aspect Ratio


nodespersemi = int(input_dict["wing"]["nodes_per_semispan"])
N = nodespersemi * 2 - 1                                             # Number of nodes


# Loop through range of Aspect Ratios
RAS = np.linspace(2, 20, 10)
CLAS = np.zeros((len(RAS), 1))
print("RAS is",RAS)
for rr in range(len(RAS)):

    Ra = RAS[rr]

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
        CLAS[rr] = Cla / (1 + (Cla / (pi * Ra)))
    else:
        CLAS[rr] = Cla / ((1 + (Cla/(pi*Ra))) * (1+KL))




# 4. Plot CLa for various RAS
fig, ax = plt.subplots()
ax.plot(RAS, CLAS, 'o', color="k")
plt.xlabel('RA')
plt.ylabel('CLa')
plt.title('Lift Slope with Aspect Ratio')
plt.show()



