#!/usr/bin/env python

#################################################################################
# This script is used to read in a ascii table and generated another ascii table#
# with new derived data appended at the end of each row                         #
#################################################################################

from __future__ import division

import numpy as np
import sys
import os
import cmath
import numpy.linalg as LA

#################
# Control Macro #
#################
GLOBAL_STRESS = np.matrix('0 0 0; 0 1 0; 0 0 0')
MATERIAL_CONFIG_TEXTURE = 195


####################
# Class Definition #
####################
class Point:
    '''
    a simple wrapper for points in real space
    '''
    def __init__(self, coords):
        self.m_x = float(coords[0])
        self.m_y = float(coords[1])
        self.m_z = float(coords[2])
        self.m_coords = [float(item) for item in coords]


class Tensor33:
    '''
    a standard 3x3 Hermitian matrix representing physical properties of poly-
    crystalline materials.
    '''
    def __init__(self, vector):
        '''
        [v1, v2, v3, v4, v5, v6]
        ->
        [[v1, v4, v5],
         [v4, v2, v6],
         [v5, v6, v3]]
        Notice:
            This reshaping is due to the output format, not the standard 3x3
            to 6x1 reshaping
        '''
        self.m_tensor33 = np.zeros((3,3))
        self.m_vector = [float(item)/1e9 for item in vector]
        self.m_tensor33[0, 0] = float(vector[0])/1e9
        self.m_tensor33[1, 1] = float(vector[1])/1e9
        self.m_tensor33[2, 2] = float(vector[2])/1e9
        self.m_tensor33[1, 0] = float(vector[3])/1e9
        self.m_tensor33[0, 1] = float(vector[3])/1e9
        self.m_tensor33[1, 2] = float(vector[4])/1e9
        self.m_tensor33[2, 1] = float(vector[4])/1e9
        self.m_tensor33[0, 2] = float(vector[5])/1e9
        self.m_tensor33[2, 0] = float(vector[5])/1e9
        self.m_pstress = {}

    def get_frob_norm(self):
        '''
        return Frobenius norm
        '''
        norm = self.get_frob_norm_scalar()
        return self.m_tensor33/norm

    def get_frob_norm_scalar(self):
        '''
        return the absolute value of Frobenius norm of this tensor
        '''
        return LA.norm(self.m_tensor33, 'fro')

    def get_stress(self):
        '''
        return the stress tensor
        '''
        return self.m_tensor33

    def get_vector(self):
        '''
        return the tensor as a vector
        '''
        return self.m_vector
    
    def get_principle_stress(self):
        '''
        return a dictionary of the principle stress based on the current stress
        tenser.
        Notice:
          The standard python dict does not store the value in order, so when
          calling this method to get the principle stress, it will return an
          unordered dict that have following structure
            pstress = {'magnitude': 'direction'}
          To find the max/min/inter principle stress, use the following:
            for item in sorted(pstress):
                ...
        '''
        mag, direc = LA.eig(self.m_tensor33)
        direc = direc.T
        x = np.array([1, 0, 0])
        y = np.array([0, 1, 0])
        for index in range(3):
            self.m_pstress[mag[index]] = direc[index]
        return self.m_pstress


class EulerAngle:
    '''
    Euler angle convection for describing crystal orientation in a poly-
    crystalline materials
    '''
    def __init__(self, angle_list):
        '''
        initialize with 3 angles
        '''
        self.m_eulerangles = [float(item) for item in angle_list]

    def get_rotation_matrix(self):
        '''
        return the rotation matrix according to given Euler Angles
        '''
        radian2degree = np.pi/180
        phi1 = self.m_eulerangles[0] * radian2degree
        phi = self.m_eulerangles[1] * radian2degree
        phi2 = self.m_eulerangles[2] * radian2degree
        # rotation axis z->x->z
        rotation_matrix_phi1 = np.matrix([[np.cos(phi1), np.sin(phi1), 0],
                                          [-np.sin(phi1), np.cos(phi1), 0],
                                          [0, 0, 1]])
        rotation_matrix_phi = np.matrix([[1, 0, 0],
                                         [0, np.cos(phi), np.sin(phi)],
                                         [0, -np.sin(phi), np.cos(phi)]])
        rotation_matrix_phi2 = np.matrix([[np.cos(phi2), np.sin(phi2), 0],
                                          [-np.sin(phi2), np.cos(phi2), 0],
                                          [0, 0, 1]])
        # return the single rotation matrix
        return rotation_matrix_phi2*rotation_matrix_phi*rotation_matrix_phi1

    def get_phi1(self):
        '''
        return the first Euler Angle
        '''
        return self.m_eulerangles[0]

    def get_phi(self):
        '''
        return the second Euler Angle
        '''
        return self.m_eulerangles[1]

    def get_phi2(self):
        '''
        return the third Euler Angle
        '''
        return self.m_eulerangles[2]

    def get_eulerangles(self):
        '''
        return the full Euler Angle as a list
        '''
        return self.m_eulerangles


class SlipSystem:
    '''
    define slip system, currently only suppor HCP    
    '''
    def __init__(self, crystal_structure='hcp', c_over_a=1.58):
        '''
        initialize slip system handle
        '''
        self.m_crystal_structure = crystal_structure
        self.m_c_over_a = c_over_a
        if self.m_crystal_structure == 'hcp':
            # Miller Indices for display
            self.m_slip_system_disp = {}
            # Cartisan Coords for calculation
            self.m_slip_system_calc = {}
            # setting up slip system for Ti (HCP)
            # Basal Slip {0001}<1120>
            self.m_slip_system_disp[0] = np.matrix('0 0 0 1; 2 -1 -1 0')
            self.m_slip_system_disp[1] = np.matrix('0 0 0 1; -1 2 -1 0')
            self.m_slip_system_disp[2] = np.matrix('0 0 0 1; -1 -1 2 0')
            # Prism Slip {1010}<1120>
            self.m_slip_system_disp[3] = np.matrix('0 1 -1 0; 2 -1 -1 0')
            self.m_slip_system_disp[4] = np.matrix('-1 0 1 0; -1 2 -1 0')
            self.m_slip_system_disp[5] = np.matrix('1 -1 0 0; -1 -1 2 0')
            # Pyramidal a Slip {1011}<1120>
            self.m_slip_system_disp[6] = np.matrix('0 1 -1 1 ; 2 -1 -1 0')
            self.m_slip_system_disp[7] = np.matrix('-1 1 0 1 ; 1 1 -2 0')
            self.m_slip_system_disp[8] = np.matrix('-1 0 1 1 ;-1 2 -1 0')
            self.m_slip_system_disp[9] = np.matrix('0 -1 1 1 ;-2 1 1 0')
            self.m_slip_system_disp[10] = np.matrix('1 -1 0 1 ; -1 -1 2 0')
            self.m_slip_system_disp[11] = np.matrix('1 0 -1 1 ; 1 -2 1 0')
            # Pyramidal c+a Slip {1011}<2113>
            self.m_slip_system_disp[12] = np.matrix('0 1 -1 1 ; -1 2 -1 -3')
            self.m_slip_system_disp[13] = np.matrix('0 1 -1 1 ; 1 1 -2 -3')
            self.m_slip_system_disp[14] = np.matrix('-1 1 0 1 ; -2 1 1 -3')
            self.m_slip_system_disp[15] = np.matrix('-1 1 0 1 ; -1 2 -1 -3')
            self.m_slip_system_disp[16] = np.matrix('-1 0 1 1 ; -1 -1 2 -3')
            self.m_slip_system_disp[17] = np.matrix('-1 0 1 1 ; -2 1 1 -3')
            self.m_slip_system_disp[18] = np.matrix('0 -1 1 1 ; 1 -2 1 -3')
            self.m_slip_system_disp[19] = np.matrix('0 -1 1 1 ;-1 -1 2 -3')
            self.m_slip_system_disp[20] = np.matrix('1 -1 0 1 ; 2 -1 -1 -3')
            self.m_slip_system_disp[21] = np.matrix('1 -1 0 1 ; 1 -2 1 -3')
            self.m_slip_system_disp[22] = np.matrix('1 0 -1 1 ; 1 1 -2 -3')
            self.m_slip_system_disp[23] = np.matrix('1 0 -1 1 ; 2 -1 -1 -3')
            # Convert slip system to cartesian for calculation
            self.m_slip_system_calc = self.bravis_miller_2_cartesian()
        elif self.m_crystal_structure == 'bcc':
            print "Support for BCC will come soon"
            sys.exit(-1)
        elif self.m_crystal_structure == 'fcc':
            print "Support for FCC will come soon"
            sys.exit(-1)
        else:
            print "error, uknown crystal structure: {}".format(
                    self.m_crystal_structure)
            sys.exit(-1)
    
    def bravis_miller_2_cartesian(self):
        '''
        convert Bravis-Miller indices to standard Cartesian Coordinates
        '''
        # a 3x4 matrix that will convert both slip direction and plane normal
        # from Bravis-Miller to Standard Miller indices
        # note:
        # the cartesian version is normalized
        miller2cartesian = np.matrix([[1, 0, 0, 0],
                                      [1/np.sqrt(3), 2/np.sqrt(3), 0, 0],
                                      [0, 0, 0, 1/self.m_c_over_a]])
        cartesian = {}
        for item in self.m_slip_system_disp:
            plane = miller2cartesian * self.m_slip_system_disp[item][0].T
            direc = miller2cartesian * self.m_slip_system_disp[item][1].T
            # normalization
            norm_plane = np.sqrt(plane.T * plane)[0, 0]
            norm_direc = np.sqrt(direc.T * direc)[0, 0]
            cartesian[item] = np.array([[plane.T/norm_plane],
                                        [direc.T/norm_direc]]) 
        return cartesian

    def get_schmid_matrix(self):
        '''
        return schmid matrix (a metrix related to slip system), which is the 
        outter product of slip plane and slip direction
        '''
        schmid_matrix = {}
        for item in self.m_slip_system_calc:
            plane = np.matrix(self.m_slip_system_calc[item][0])
            direc = np.matrix(self.m_slip_system_calc[item][1])
            schmid_matrix[item] = 0.5*(plane.T * direc + direc.T * plane)
        return schmid_matrix


class IntegrationPoint:
    '''
    a container used to store info from selected integration point, currently
    include info:
        1. crystal structure
        2. Euler angle
        3. local stress tensor
        4. global stress tensor
        5. accumulated shear from each slip system
    note:
        most of the new parameters should be calculated and tested in this 
        class, future interesting parameters including evolution of slip
        resistance, fracture initiation parameter and so on.
        Right now global schmid factor and local schmid factor are calculated
        separately, might consider combining these two method in one as they
        are prett much the same thing
    '''
    def __init__(self,
                 coord,
                 texture,
                 orientation,
                 stress_local,
                 accumulated_shear,
                 shearrate,
                 crystal_structure='hcp',
                 c_over_a=1.58):
        '''
        initialize with standard input
        note:
            the global stress is set for an uniaxial tensile test in the y
            direction, change it according to your experiment
        '''
        self.m_slip_system = SlipSystem(crystal_structure, c_over_a)
        self.m_texture = texture
        self.m_coord = Point(coord)
        self.m_stress_local = Tensor33(stress_local)
        self.m_orientation = EulerAngle(orientation)
        self.m_accumulated_shear = accumulated_shear
        self.m_crystal_structure = crystal_structure
        self.m_shear_rate = shearrate

    def get_euerlangles(self):
        '''
        a simple wrapper for getting the Euler Angle list
        '''
        return self.m_orientation.get_eulerangles()
    
    def get_schmid_factor_global(self):
        '''
        calculate the global schmid factor and return a vector containning 
        all the schmid factor for each slip system
        '''
        global_schmid_factor = []
        # normalize the sterss
        stress = GLOBAL_STRESS/LA.norm(GLOBAL_STRESS, 'fro')
        rotation_matrix = self.m_orientation.get_rotation_matrix()
        # transfer the stres to the local configuration
        stress_rotated = rotation_matrix.T * stress * rotation_matrix
        schmid_matrix = self.m_slip_system.get_schmid_matrix()
        for item in schmid_matrix:
            temp_schmid_factor = (stress_rotated*schmid_matrix[item]).trace()
            temp_schmid_factor = np.absolute(temp_schmid_factor)
            global_schmid_factor.append(temp_schmid_factor)
        return global_schmid_factor

    def get_schmid_factor_local(self):
        '''
        calculate the local schmid factor and return a vector contrainning
        all the schmid factor for each slip system
        '''
        local_schmid_factor = []
        # normalize the stress
        stress = self.m_stress_local.get_frob_norm()
        rotation_matrix = self.m_orientation.get_rotation_matrix()
        # transfer the stress to the local configuration
        stress_rotated = rotation_matrix.T * stress * rotation_matrix
        schmid_matrix = self.m_slip_system.get_schmid_matrix()
        for item in schmid_matrix:
            temp_schmid_factor = (stress_rotated*schmid_matrix[item]).trace()
            temp_schmid_factor = np.absolute(temp_schmid_factor)
            local_schmid_factor.append(temp_schmid_factor)
        return local_schmid_factor

    def get_principle_stress(self):
        '''
        return the principle stress of the local stress, just a simple wrapper
        for the get_principle_stress() method in Tensor33 class
        '''
        return self.m_stress_local.get_principle_stress()

    def get_misorientation(self, dict_init_eulerangles):
        '''
        based on crystal structure as well as the initial eulerangle list,
        calculate the mis-orientation between current grain orientation and 
        the initial grain orientation
        '''
        initial_orientation = EulerAngle(dict_init_eulerangles[self.m_texture])
        rotation_init = initial_orientation.get_rotation_matrix()
        rotation_now = self.m_orientation.get_rotation_matrix()
        sym_matrix_1 = self.get_symmetry_operator()
        sym_matrix_2 = self.get_symmetry_operator()
        misorientation = 359
        # considering all the possible equivalent combinations
        for item_1 in sym_matrix_1:
            sym_m_1 = item_1.get_rotation_matrix()
            temp_rotation_init = sym_m_1 * rotation_init
            for item_2 in sym_matrix_2:
                sym_m_2 = item_2.get_rotation_matrix()
                temp_rotation_now = sym_m_2 * rotation_now
                total_ro = temp_rotation_init * temp_rotation_now.T
                values, vectors = LA.eig(total_ro)
                temp_misorientation = "zero"
                # find the eigen value for angle
                for value in values:
                    if value == 1:
                        continue
                    else:
                        temp_misorientation = cmath.phase(value)*180/np.pi
                        if temp_misorientation > 1e-6:
                            break
                # test if we have a near zero rotation situation
                if temp_misorientation == "zero":
                    temp_misorientation = 0
                else:
                    temp_misorientation = np.absolute(temp_misorientation)
                # always store the smallest mis-orientation
                if temp_misorientation < misorientation:
                    misorientation = temp_misorientation
                if misorientation < 1e-6:
                    break
            if misorientation < 1e-6:
                break
        return misorientation

    def get_symmetry_operator(self):
        '''
        return a list of symmetry operators that are in a 3x3 matrix form
        Var:
            list_operator: a list of instances of EulerAngles
        '''
        list_symoperator = []
        if self.m_crystal_structure == 'hcp':
            # only considering the Ti case here where a 32 symmetry group is
            # assumed
            # total number of symmetry group: 3x2=6
            phi1 = np.array([0, 120, 240])
            phi = np.array([0, 180])
            phi2 = 0.0
            for index_1 in range(0,3):
                for index_2 in range(0,2):
                    temp_eulerangles = [phi1[index_1], phi[index_2], phi2]
                    list_symoperator.append(EulerAngle(temp_eulerangles))
        elif self.m_crystal_structure == 'cubic':
            # Cubic has a total number of 24 symmetry group
            # 3 steps rotation: 4x4=16
            phi1 = np.array([0, 90, 180, 270])
            phi = 90.0
            phi2 = np.array([0, 90, 180, 270])
            for index_1 in range(0, 4):
                for index_2 in range(0, 4):
                    temp_eulerangles = [phi1[index_1], phi, phi2[index_2]]
                    list_symoperator.append(EulerAngle(temp_eulerangles))
            # 1 step rotation: 4x2=8
            phi1 = np.array([0, 90, 180, 270])
            phi = np.array([0, 180])
            phi2 = 0.0
            for index_1 in range(0, 4):
                for index_2 in range(0, 2):
                    temp_eulerangles = [phi[index_1], phi[index_2], phi2]
                    list_symoperator.append(EulerAngle(temp_eulerangles))
        else:
            print "{} is not supported".format(self.m_crystal_structure)
            sys.exit(-1)
        return list_symoperator


#################
# Main Function #
#################
def main():
  infilename = os.getcwd() + "/" + sys.argv[1]
  outfilename = infilename.strip(".txt") + "_ascii.txt"
  infile = open(infilename, "r")
  raw_data = infile.readlines()
  infile.close()
  ##
  # generate dict for inital orientation
  #
  config_filename = os.getcwd() + "/material.config"
  config_file = open(config_filename)
  initial_orientation = {}
  orientation_data = config_file.readlines()[MATERIAL_CONFIG_TEXTURE:]
  for index in range(3, len(orientation_data), 3):
    temp_string = orientation_data[index].split()
    temp_eulerang = [temp_string[2], temp_string[4], temp_string[6]]
    initial_orientation[index/3] = [float(item) for item in temp_eulerang]
  ##
  # parse haeder
  #
  header = raw_data.pop(0)
  header += raw_data.pop(0)
  header += raw_data.pop(0).strip("\n")
  header += "\tGSF_basal\tGSF_prism\tGSF_pyra\tGSF_pyrca"
  header += "\tLSF_basal\tLSG_prism\tLSF+pyra\tLSF_pyrca"
  header += "\tshearrate_basal\tshearrate_prism\t_shearrate_pyra\tshearrate_pyrca"
  header += "\tacshear_basal\tacshear_prism\tacshear_pyra\tacshear_pyrca\tmoe\n"
  outfile = open(outfilename, "w")
  outfile.write(header)
  ##
  # parse line by line
  #
  for line in raw_data:
    data = line.split()
    coords = [float(item) for item in data[2:5]]
    stress_local = [float(item) for item in data[6:12]]
    texture = float(data[12])
    orientation = [float(item) for item in data[13:16]]
    slip_resistance = [float(item) for item in data[16:40]]
    shear_rate = [float(item) for item in data[40:64]]
    resolved_stress = [float(item) for item in data[64:88]]
    accumulated_shear = [float(item) for item in data[88:]]
    temp_ip = IntegrationPoint(coords, texture, orientation, 
                               stress_local, accumulated_shear, shear_rate,
                               'hcp', 1.58)
    outstring = line.strip("\n")
    # get global schmid factor
    GSF = [float(item) for item in temp_ip.get_schmid_factor_global()]
    outstring += "\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}".format(max(GSF[0:3]),
                                                           max(GSF[3:6]),
                                                           max(GSF[6:12]),
                                                           max(GSF[12:24]))
    # get local schmid factor
    LSF = [float(item) for item in temp_ip.get_schmid_factor_local()]
    outstring += "\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}".format(max(LSF[0:3]),
                                                           max(LSF[3:6]),
                                                           max(LSF[6:12]),
                                                           max(LSF[12:24]))
    # get shear rate
    SR = temp_ip.m_shear_rate
    outstring += "\t{}\t{}\t{}\t{}".format(max(SR[0:3]),
                                           max(SR[3:6]),
                                           max(SR[6:12]),
                                           max(SR[12:24]))
    # get accumulative shear
    ACS = temp_ip.m_accumulated_shear
    outstring += "\t{}\t{}\t{}\t{}".format(sum(ACS[0:3]),
                                           sum(ACS[3:6]),
                                           sum(ACS[6:12]),
                                           sum(ACS[12:24]))
    # get misorientatino evolution
    MOE = temp_ip.get_misorientation(initial_orientation)
    outstring += "\t{:.2f}\n".format(MOE)
    outfile = open(outfilename, "a+")
    outfile.write(outstring)

  print "ALL DONE, GOOD-BYE"

if __name__ == "__main__":
  main()

