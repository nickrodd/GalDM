###############################################################################
# dm_profiles.pyx
###############################################################################
#
# Define three galactic dark matter (DM) profiles: NFW, Burkert, and Einasto
#
###############################################################################


cimport cython

# C functions
cdef extern from "math.h":
    double pow(double x, double y) nogil
    double exp(double x) nogil
    double cos(double x) nogil
    double sqrt(double x) nogil

#####################################
# Navarro-Frenk-White (NFW) Profile #
#####################################

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef double rhoNFW(double x, double b, double l,
                   double rsun, double rs, double gamma, double rhosNFW) nogil:
    """ Generalized NFW DM density

    :param x: distance from Earth [kpc]
    :param b: latitude in galactic coordinates [rad]
    :param l: longitude in galactic coordinates [rad]
    :param rsun: sun-GC distance [kpc], common value 8.5
    :param rs: NFW scale radius [kpc], common value 20
    :param gamma: generalised NFW parameter, 1.0 gives canonical profile
    :param rhosNFW: normalising density [GeV/cm^3]

    :returns: DM density [GeV/cm^3]
    """

    cdef double rval = rGC(x, rsun, b, l)

    return rhosNFW * pow(rval/rs,-gamma) * pow(1.+rval/rs,gamma-3.)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef double NFWnorm(double rsun, double rs, double gamma, double rhosun) nogil:
    """ Normalising density for the NFW profile

    :param rsun: sun-GC distance [kpc], common value 8.5
    :param rs: NFW scale radius [kpc], common value 20
    :param gamma: generalised NFW parameter, 1.0 gives canonical profile
    :param rhosun: DM density at the location of the sun [GeV/cm^3]

    :returns: NFW normalising density [GeV/cm^3]
    """

    NFW_prenorm = rhoNFW(0., 0., 0., rsun, rs, gamma, 1.)

    return rhosun / NFW_prenorm


###################
# Burkert Profile #
###################

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef double rhoBurkert(double x, double b, double l,
                       double rsun, double rs, double rhosBurkert) nogil:
    """ Burkert DM density

    :param x: distance from Earth [kpc]
    :param b: latitude in galactic coordinates [rad]
    :param l: longitude in galactic coordinates [rad]
    :param rsun: sun-GC distance [kpc], common value 8.5
    :param rs: Burkert scale radius [kpc], common value 14
    :param rhosBurkert: normalising density [GeV/cm^3]

    :returns: DM density [GeV/cm^3]
    """

    cdef double rval = rGC(x, rsun, b, l)

    return rhosBurkert * pow(1.+rval/rs,-1.) * pow(1.+pow(rval/rs,2.),-1.)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef double Burkertnorm(double rsun, double rs, double rhosun) nogil:
    """ Normalising density for the Einasto profile

    :param rsun: sun-GC distance [kpc], common value 8.5
    :param rs: Burkert scale radius [kpc], common value 14
    :param rhosun: DM density at the location of the sun [GeV/cm^3]

    :returns: Burkert normalising density [GeV/cm^3]
    """

    Burkert_prenorm = rhoBurkert(0., 0., 0., rsun, rs, 1.)

    return rhosun / Burkert_prenorm


###################
# Einasto Profile #
###################

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef double rhoEinasto(double x, double b, double l,
                       double rsun, double rs, double alpha, 
                       double rhosEinasto) nogil:
    """ Einasto DM density

    :param x: distance from Earth [kpc]
    :param b: latitude in galactic coordinates [rad]
    :param l: longitude in galactic coordinates [rad]
    :param rsun: sun-GC distance [kpc], common value 8.5
    :param rs: Einasto scale radius [kpc], common value 20
    :param alpha: degree of curvature, common value 0.17
    :param rhosEinasto: normalising density [GeV/cm^3]

    :returns: DM density [GeV/cm^3]
    """

    cdef double rval = rGC(x, rsun, b, l)

    return rhosEinasto * exp( (-2./alpha) * (pow(rval/rs,alpha) - 1.) )

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef double Einastonorm(double rsun, double rs, double alpha, 
                        double rhosun) nogil:
    """ Normalising density for the Einasto profile

    :param rsun: sun-GC distance [kpc], common value 8.5
    :param rs: Einasto scale radius [kpc], common value 20
    :param alpha: degree of curvature, common value 0.17
    :param rhosun: DM density at the location of the sun [GeV/cm^3]

    :returns: Einasto normalising density [GeV/cm^3]
    """

    Einasto_prenorm = rhoEinasto(0., 0., 0., rsun, rs, alpha, 1.)

    return rhosun / Einasto_prenorm


####################
# Common Functions #
####################

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef double rGC(double x, double rsun, double b, double l) nogil:
    """ Distance from the Galactic Center (GC)

    :param x: distance from Earth [kpc]
    :param rsun: sun-GC distance [kpc], common value 8.5
    :param b: latitude in galactic coordinates [rad]
    :param l: longitude in galactic coordinates [rad]

    :returns: distance from GC [kpc]
    """

    return sqrt(pow(x,2.) - 2.*rsun*x*cos(b)*cos(l) + pow(rsun,2.))
