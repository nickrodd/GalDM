###############################################################################
# gal_dm.py
###############################################################################
#
# Class for creating galactic dark matter (DM) maps and calculated J/D factors
#
# An example of how do define the ROIs to calculate J/D factors is found here:
# https://github.com/bsafdi/NPTFit/blob/master/examples/Example2_Creating_Masks.ipynb
#
###############################################################################


# Import basic functions
import os
import numpy as np
import healpy as hp
from tqdm import *
from .prop_mod import mod
from .create_mask_forJD import make_mask_total as mmt

# Import modules for calculating J/D factors
from .dm_int import dm_NFW, dm_Burkert, dm_Einasto


class dm_maps():
    def __init__(self, maps_dir=None, base_nside=1024, jd_nside=2048, 
                 xmax=200., xsteps=10000):
        """ Class for calculating DM maps and J/D factors

        :param maps_dir: where the maps will be stored
        :param base_nside: nside default maps are created with
        :param jd_nside: nside to use when calculating J/D factors
        :param xmax: maximum x to integrate to, formally infinity [kpc]
        :param xsteps: number of steps to use in the integral
        """

        self.base_nside = base_nside
        self.jd_nside = jd_nside
        self.base_npix = hp.nside2npix(base_nside)
        self.coords_setup = False
        
        self.xmax = xmax
        self.xsteps = xsteps

        # Create maps_dir if does not already exist
        self.maps_dir = maps_dir
        self.set_map_dir()


    def nfw(self, nside=128, rsun=8.5, rs=20., gamma=1., rhosun=0.4, decay=0):
        """ Create an NFW DM map if it doesn't exist, or load otherwise
            Map has units of [GeV^2/cm^5.sr] for annihilation or [GeV/cm^2.sr] 
            for decay

        :param nside: nside you want the map returned in
        :param rsun: sun-GC distance [kpc]
        :param rs: NFW scale radius [kpc]
        :param gamma: generalised NFW parameter
        :param rhosun: DM density at the location of the sun [GeV/cm^3]
        :param decay: 0 for annihilation map, 1 for decay
        """
        
        nfw_str = 'NFW_J_'
        if decay:
            nfw_str = 'NFW_D_'
        nfw_str += 'rsun' + str(rsun) + '_rs' + str(rs) + '_g' + str(gamma) + \
                   '_rhosun' + str(rhosun) + '_nside' + str(self.base_nside) + \
                   '.npy'

        # If file exists load, if not create
        if os.path.exists(self.maps_dir + nfw_str):
            base_nfw_map = np.load(self.maps_dir + nfw_str) 
        else:
            print "Map does not exist, creating..."
            
            self.galactic_coords()
            base_nfw_map = np.zeros(self.base_npix)
            for p in tqdm(range(self.base_npix)):
                base_nfw_map[p] = dm_NFW(self.bval[p], self.lval[p], rsun, rs,
                                         gamma, rhosun, decay, 
                                         self.xmax, self.xsteps)

            # Multiply by area of pixel to add sr unit
            base_nfw_map *= self.pixarea
            
            np.save(self.maps_dir + nfw_str, base_nfw_map)
            print "...done!"

        return hp.ud_grade(np.array(base_nfw_map), nside, power=-2)

    
    def burkert(self, nside=128, rsun=8.5, rs=14., rhosun=0.4, decay=0):
        """ Create an Burkert DM map if it doesn't exist, or load otherwise
            Map has units of [GeV^2/cm^5.sr] for annihilation or [GeV/cm^2.sr]
            for decay

        :param nside: nside you want the map returned in
        :param rsun: sun-GC distance [kpc]
        :param rs: Burkert scale radius [kpc]
        :param rhosun: DM density at the location of the sun [GeV/cm^3]
        :param decay: 0 for annihilation map, 1 for decay
        """

        bur_str = 'Bur_J_'
        if decay:
            bur_str = 'Bur_D_'
        bur_str += 'rsun' + str(rsun) + '_rs' + str(rs) + \
                   '_rhosun' + str(rhosun) + '_nside' + str(self.base_nside) + \
                   '.npy'

        # If file exists load, if not create
        if os.path.exists(self.maps_dir + bur_str):
            base_bur_map = np.load(self.maps_dir + bur_str)
        else:
            print "Map does not exist, creating..."

            self.galactic_coords()
            base_bur_map = np.zeros(self.base_npix)
            for p in tqdm(range(self.base_npix)):
                base_bur_map[p] = dm_Burkert(self.bval[p], self.lval[p], rsun, 
                                             rs, rhosun, decay,
                                             self.xmax, self.xsteps)

            # Multiply by area of pixel to add sr unit
            base_bur_map *= self.pixarea

            np.save(self.maps_dir + bur_str, base_bur_map)
            print "...done!"

        return hp.ud_grade(np.array(base_bur_map), nside, power=-2)


    def einasto(self, nside=128, rsun=8.5, rs=20., alpha=0.17, rhosun=0.4, 
                decay=0):
        """ Create an Einasto DM map if it doesn't exist, or load otherwise
            Map has units of [GeV^2/cm^5.sr] for annihilation or [GeV/cm^2.sr]
            for decay

        :param nside: nside you want the map returned in
        :param rsun: sun-GC distance [kpc]
        :param rs: Ein scale radius [kpc]
        :param alpha: degree of curvature, common value 0.17
        :param rhosun: DM density at the location of the sun [GeV/cm^3]
        :param decay: 0 for annihilation map, 1 for decay
        """

        ein_str = 'Ein_J_'
        if decay:
            ein_str = 'Ein_D_'
        ein_str += 'rsun' + str(rsun) + '_rs' + str(rs) + '_a' + str(alpha) + \
                   '_rhosun' + str(rhosun) + '_nside' + str(self.base_nside) + \
                   '.npy'

        # If file exists load, if not create
        if os.path.exists(self.maps_dir + ein_str):
            base_ein_map = np.load(self.maps_dir + ein_str)
        else:
            print "Map does not exist, creating ..."

            self.galactic_coords()
            base_ein_map = np.zeros(self.base_npix)
            for p in tqdm(range(self.base_npix)):
                base_ein_map[p] = dm_Einasto(self.bval[p], self.lval[p], rsun, rs,
                                             alpha, rhosun, decay,
                                             self.xmax, self.xsteps)

            # Multiply by area of pixel to add sr unit
            base_ein_map *= self.pixarea

            np.save(self.maps_dir + ein_str, base_ein_map)
            print "...done!"

        return hp.ud_grade(np.array(base_ein_map), nside, power=-2)


    def nfw_j(self, rsun=8.5, rs=20., gamma=1., rhosun=0.4,
              band_mask=False, band_mask_range=30,
              l_mask=False, l_deg_min=-30, l_deg_max=30,
              b_mask=False, b_deg_min=-30, b_deg_max=30,
              mask_ring=False, inner=0, outer=30,
              ring_b=0, ring_l=0,
              custom_mask=None):
        """ Calculate the NFW J-factor
        """

        ROI = mmt(nside=self.jd_nside, 
                  band_mask=band_mask, band_mask_range=band_mask_range,
                  l_mask=l_mask, l_deg_min=l_deg_min, l_deg_max=l_deg_max,
                  b_mask=b_mask, b_deg_min=b_deg_min, b_deg_max=b_deg_max,
                  mask_ring=mask_ring, inner=inner, outer=outer,
                  ring_b=ring_b, ring_l=ring_l, custom_mask=custom_mask)
        
        ROIsr = float(len(ROI)) * hp.nside2pixarea(self.jd_nside)

        dm_map = self.nfw(nside=self.jd_nside, rsun=rsun, rs=rs, gamma=gamma, 
                          rhosun=rhosun, decay=0)

        print "J-Factor for NFW profile in this ROI:"
        print np.sum(dm_map[ROI]), "GeV^2/cm^5.sr"
        print np.sum(dm_map[ROI])/ROIsr, "GeV^2/cm^5"


    def nfw_d(self, rsun=8.5, rs=20., gamma=1., rhosun=0.4,
              band_mask=False, band_mask_range=30,
              l_mask=False, l_deg_min=-30, l_deg_max=30,
              b_mask=False, b_deg_min=-30, b_deg_max=30,
              mask_ring=False, inner=0, outer=30,
              ring_b=0, ring_l=0,
              custom_mask=None):
        """ Calculate the NFW D-factor
        """

        ROI = mmt(nside=self.jd_nside,
                  band_mask=band_mask, band_mask_range=band_mask_range,
                  l_mask=l_mask, l_deg_min=l_deg_min, l_deg_max=l_deg_max,
                  b_mask=b_mask, b_deg_min=b_deg_min, b_deg_max=b_deg_max,
                  mask_ring=mask_ring, inner=inner, outer=outer,
                  ring_b=ring_b, ring_l=ring_l, custom_mask=custom_mask)

        ROIsr = float(len(ROI)) * hp.nside2pixarea(self.jd_nside)

        dm_map = self.nfw(nside=self.jd_nside, rsun=rsun, rs=rs, gamma=gamma,
                          rhosun=rhosun, decay=1)

        print "D-Factor for NFW profile in this ROI:"
        print np.sum(dm_map[ROI]), "GeV/cm^2.sr"
        print np.sum(dm_map[ROI])/ROIsr, "GeV/cm^2"


    def burkert_j(self, rsun=8.5, rs=14., rhosun=0.4,
                  band_mask=False, band_mask_range=30,
                  l_mask=False, l_deg_min=-30, l_deg_max=30,
                  b_mask=False, b_deg_min=-30, b_deg_max=30,
                  mask_ring=False, inner=0, outer=30,
                  ring_b=0, ring_l=0,
                  custom_mask=None):
        """ Calculate the Burkert J-factor
        """

        ROI = mmt(nside=self.jd_nside,
                  band_mask=band_mask, band_mask_range=band_mask_range,
                  l_mask=l_mask, l_deg_min=l_deg_min, l_deg_max=l_deg_max,
                  b_mask=b_mask, b_deg_min=b_deg_min, b_deg_max=b_deg_max,
                  mask_ring=mask_ring, inner=inner, outer=outer,
                  ring_b=ring_b, ring_l=ring_l, custom_mask=custom_mask)

        ROIsr = float(len(ROI)) * hp.nside2pixarea(self.jd_nside)

        dm_map = self.burkert(nside=self.jd_nside, rsun=rsun, rs=rs,
                              rhosun=rhosun, decay=0)


        print "J-Factor for Burkert profile in this ROI:"
        print np.sum(dm_map[ROI]), "GeV^2/cm^5.sr"
        print np.sum(dm_map[ROI])/ROIsr, "GeV^2/cm^5"


    def burkert_d(self, rsun=8.5, rs=14., rhosun=0.4,
                  band_mask=False, band_mask_range=30,
                  l_mask=False, l_deg_min=-30, l_deg_max=30,
                  b_mask=False, b_deg_min=-30, b_deg_max=30,
                  mask_ring=False, inner=0, outer=30,
                  ring_b=0, ring_l=0,
                  custom_mask=None):
        """ Calculate the Burkert D-factor
        """

        ROI = mmt(nside=self.jd_nside,
                  band_mask=band_mask, band_mask_range=band_mask_range,
                  l_mask=l_mask, l_deg_min=l_deg_min, l_deg_max=l_deg_max,
                  b_mask=b_mask, b_deg_min=b_deg_min, b_deg_max=b_deg_max,
                  mask_ring=mask_ring, inner=inner, outer=outer,
                  ring_b=ring_b, ring_l=ring_l, custom_mask=custom_mask)

        ROIsr = float(len(ROI)) * hp.nside2pixarea(self.jd_nside)

        dm_map = self.burkert(nside=self.jd_nside, rsun=rsun, rs=rs,
                              rhosun=rhosun, decay=1)


        print "D-Factor for Burkert profile in this ROI:"
        print np.sum(dm_map[ROI]), "GeV/cm^2.sr"
        print np.sum(dm_map[ROI])/ROIsr, "GeV/cm^2"


    def einasto_j(self, rsun=8.5, rs=20., alpha=0.17, rhosun=0.4,
                  band_mask=False, band_mask_range=30,
                  l_mask=False, l_deg_min=-30, l_deg_max=30,
                  b_mask=False, b_deg_min=-30, b_deg_max=30,
                  mask_ring=False, inner=0, outer=30,
                  ring_b=0, ring_l=0,
                  custom_mask=None):
        """ Calculate the Einasto J-factor
        """

        ROI = mmt(nside=self.jd_nside,
                  band_mask=band_mask, band_mask_range=band_mask_range,
                  l_mask=l_mask, l_deg_min=l_deg_min, l_deg_max=l_deg_max,
                  b_mask=b_mask, b_deg_min=b_deg_min, b_deg_max=b_deg_max,
                  mask_ring=mask_ring, inner=inner, outer=outer,
                  ring_b=ring_b, ring_l=ring_l, custom_mask=custom_mask)

        ROIsr = float(len(ROI)) * hp.nside2pixarea(self.jd_nside)

        dm_map = self.einasto(nside=self.jd_nside, rsun=rsun, rs=rs, 
                              alpha=alpha, rhosun=rhosun, decay=0)

        
        print "J-Factor for Einasto profile in this ROI:"
        print np.sum(dm_map[ROI]), "GeV^2/cm^5.sr"
        print np.sum(dm_map[ROI])/ROIsr, "GeV^2/cm^5"


    def einasto_d(self, rsun=8.5, rs=20., alpha=0.17, rhosun=0.4,
                  band_mask=False, band_mask_range=30,
                  l_mask=False, l_deg_min=-30, l_deg_max=30,
                  b_mask=False, b_deg_min=-30, b_deg_max=30,
                  mask_ring=False, inner=0, outer=30,
                  ring_b=0, ring_l=0,
                  custom_mask=None):
        """ Calculate the Einasto D-factor
        """

        ROI = mmt(nside=self.jd_nside,
                  band_mask=band_mask, band_mask_range=band_mask_range,
                  l_mask=l_mask, l_deg_min=l_deg_min, l_deg_max=l_deg_max,
                  b_mask=b_mask, b_deg_min=b_deg_min, b_deg_max=b_deg_max,
                  mask_ring=mask_ring, inner=inner, outer=outer,
                  ring_b=ring_b, ring_l=ring_l, custom_mask=custom_mask)

        ROIsr = float(len(ROI)) * hp.nside2pixarea(self.jd_nside)

        dm_map = self.einasto(nside=self.jd_nside, rsun=rsun, rs=rs,
                              alpha=alpha, rhosun=rhosun, decay=1)


        print "D-Factor for Einasto profile in this ROI:"
        print np.sum(dm_map[ROI]), "GeV/cm^2.sr"
        print np.sum(dm_map[ROI])/ROIsr, "GeV/cm^2"


    def galactic_coords(self):
        """ Set galactic coordinates for base nside
        """

        if not self.coords_setup:
        
            print "Setting up coordinates ..."
            theta, phi = hp.pix2ang(self.base_nside, range(self.base_npix))

            self.bval = np.pi/2. - theta
            self.lval = mod(phi+np.pi, 2.*np.pi)-np.pi
            self.pixarea = hp.nside2pixarea(self.base_nside)

            self.coords_setup = True
            print "... done!"


    def set_map_dir(self):
        """ Define and set directory to store the maps
        """

        if self.maps_dir is None:
            self.maps_dir = os.getcwd() + '/DM_Maps/'

        if not os.path.exists(self.maps_dir):
            try:
                os.mkdir(self.maps_dir)
            except OSERROR as e:
                if e.errno != 17:
                    raise

        print "Maps will be stored in:",self.maps_dir
