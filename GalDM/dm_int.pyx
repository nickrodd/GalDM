###############################################################################
# dm_int.pyx
###############################################################################
#
# Calculate the galactic dark matter (DM) map for three profiles (NFW, Burkert, 
# and Einasto), for either decay or annihilation
#
###############################################################################


cimport cython

# C functions
cdef extern from "math.h":
    double pow(double x, double y) nogil

# Conversion constant
cdef double cmperkpc = 3.08567758e21 # kpc to cm

#####################################
# Navarro-Frenk-White (NFW) Profile #
#####################################

from .dm_profiles cimport rhoNFW, NFWnorm

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cpdef double dm_NFW(double b, double l,
                    double rsun, double rs, double gamma,
                    double rhosun, int decay,
                    double xmax, int xsteps) nogil:
    """ Calculate a map of DM following the NFW profile

    :param b: latitude in galactic coordinates [rad]
    :param l: longitude in galactic coordinates [rad]
    :param rsun: sun-GC distance [kpc], common value 8.5
    :param rs: NFW scale radius [kpc], common value 20
    :param gamma: generalised NFW parameter, 1.0 gives canonical profile
    :param rhosun: DM density at the location of the sun [GeV/cm^3]
    :param decay: 0 for annihilation map, 1 for decay
    :param xmax: maximum x to integrate to, formally infinity [kpc]
    :param xsteps: number of steps to use in the integral

    :returns: DM J/D factor for given (b,l) coordinate
              units: [GeV^2/cm^5] for annihilation
                     [GeV/cm^2] for decay
    """


    cdef double rhosNFW = NFWnorm(rsun, rs, gamma, rhosun)

    cdef double JDval, bval, lval, xval
    cdef double dx = xmax/float(xsteps)

    cdef double rhopow = 2.
    if decay == 1:
        rhopow = 1.
    
    cdef Py_ssize_t xi
    JDval = 0.
    xval = 0.
    
    for xi in range(xsteps):
        JDval += dx*pow(rhoNFW(xval, b, l, rsun, rs, gamma, rhosNFW), rhopow)
        xval += dx

    # Integrated over kpc, convert to cm
    JDval *= cmperkpc

    return JDval


###################
# Burkert Profile #
###################

from dm_profiles cimport rhoBurkert, Burkertnorm

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cpdef double dm_Burkert(double b, double l,
                        double rsun, double rs,
                        double rhosun, int decay,
                        double xmax, int xsteps) nogil:
    """ Calculate a map of DM following the Burkert profile

    :param b: latitude in galactic coordinates [rad]
    :param l: longitude in galactic coordinates [rad]
    :param rsun: sun-GC distance [kpc], common value 8.5
    :param rs: Burkert scale radius [kpc], common value 14
    :param rhosun: DM density at the location of the sun [GeV/cm^3]
    :param decay: 0 for annihilation map, 1 for decay
    :param xmax: maximum x to integrate to, formally infinity [kpc]
    :param xsteps: number of steps to use in the integral

    :returns: DM J/D factor for given (b,l) coordinate
              units: [GeV^2/cm^5] for annihilation
                     [GeV/cm^2] for decay
    """


    cdef double rhosBurkert = Burkertnorm(rsun, rs, rhosun)

    cdef double JDval, bval, lval, xval
    cdef double dx = xmax/float(xsteps)

    cdef double rhopow = 2.
    if decay == 1:
        rhopow = 1.

    cdef Py_ssize_t xi
    JDval = 0.
    xval = 0.

    for xi in range(xsteps):
        JDval += dx*pow(rhoBurkert(xval, b, l, rsun, rs, rhosBurkert), rhopow)
        xval += dx

    # Integrated over kpc, convert to cm
    JDval *= cmperkpc

    return JDval


###################
# Einasto Profile #
###################

from dm_profiles cimport rhoEinasto, Einastonorm

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cpdef double dm_Einasto(double b, double l,
                        double rsun, double rs, double alpha,
                        double rhosun, int decay,
                        double xmax, int xsteps) nogil:
    """ Calculate a map of DM following the Einasto profile

    :param b: latitude in galactic coordinates [rad]
    :param l: longitude in galactic coordinates [rad]
    :param rsun: sun-GC distance [kpc], common value 8.5
    :param rs: Burkert scale radius [kpc], common value 20
    :param alpha: degree of curvature, common value 0.17
    :param rhosun: DM density at the location of the sun [GeV/cm^3]
    :param decay: 0 for annihilation map, 1 for decay
    :param xmax: maximum x to integrate to, formally infinity [kpc]
    :param xsteps: number of steps to use in the integral

    :returns: DM J/D factor for given (b,l) coordinate
              units: [GeV^2/cm^5] for annihilation
                     [GeV/cm^2] for decay
    """


    cdef double rhosEinasto = Einastonorm(rsun, rs, alpha, rhosun)

    cdef double JDval, bval, lval, xval
    cdef double dx = xmax/float(xsteps)

    cdef double rhopow = 2.
    if decay == 1:
        rhopow = 1.

    cdef Py_ssize_t xi
    JDval = 0.
    xval = 0.

    for xi in range(xsteps):
        JDval += dx*pow(rhoEinasto(xval, b, l, rsun, rs, alpha, rhosEinasto),
                        rhopow)
        xval += dx

    # Integrated over kpc, convert to cm
    JDval *= cmperkpc

    return JDval
