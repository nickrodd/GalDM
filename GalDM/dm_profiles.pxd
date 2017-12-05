###############################################################################
# dm_profiles.pxd
###############################################################################
#
# Predefine functions for optimal compilation
#
###############################################################################


cdef double rhoNFW(double x, double b, double l,
                   double rsun, double rs, double gamma, double rhosNFW) nogil

cdef double NFWnorm(double rsun, double rs, double gamma, double rhosun) nogil

cdef double rhoBurkert(double x, double b, double l,
                       double rsun, double rs, double rhosBurkert) nogil

cdef double Burkertnorm(double rsun, double rs, double rhosun) nogil

cdef double rhoEinasto(double x, double b, double l,
                       double rsun, double rs, double alpha,
                       double rhosEinasto) nogil

cdef double Einastonorm(double rsun, double rs, double alpha,
                        double rhosun) nogil

cdef double rGC(double x, double rsun, double b, double l) nogil
