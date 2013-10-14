#!/usr/bin/env python

'''
    @2013-01-28
    @Chen
    @Developer Notes:
        1.usage
        This python script is a prototype for a crystallography analysis
        package, which will be rewritten in Qt and c++. The main purpose 
        of this script is to provide a quick way to generate Schmid Factor
        map and some other derived properties from CPFE simulation. The input
        file should be an ANSII file generated from the FEM software 
        MSC.Marc/Mentat with the help of DAMASK. For detailed usage, use 
            engineSF.py -h
        for instruction.
        2.
        Currently this module only works for HCP material, but all necessary 
        API are left as dummies for further implementation if deemed necessary.
        3.
        Currently most of the bugs have been fixed to address the issue of 
        large Schmid Factor (>0.707). However, under certain circumstances
        large Schmid Factor still shows up, which is quite confusing yet I do
        not have a clear way to fix it for the moment. Will consider further
        address this issue sometime in the next version
'''


#-----------------------------------------------------------------------------#
# heads and official lib
#-----------------------------------------------------------------------------#
from __future__ import division

# The following code is use to enable this script to run in the background on 
# the server, remove the doc string mark when needed
import matplotlib
matplotlib.use('agg')

import numpy as np
import matplotlib.pyplot as plt
import sys 
import argparse
import cmath
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial import KDTree
from numpy import linalg as LA


#-----------------------------------------------------------------------------#
# Contorl Macro
#-----------------------------------------------------------------------------#
GLOBAL_STRESS = np.matrix('0 0 0; 0 1 0; 0 0 0')
MODEL_SIZE_X = 135
MODEL_SIZE_Y = 145
MY_NEIGHBORS = 3
MATERIAL_CONFIG_TEXTURE = 195


#-----------------------------------------------------------------------------#
# class definition
#-----------------------------------------------------------------------------# 
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
        '''
        # reorient principal stress vector to make it fall in the right position
        for i in range(3):
            test_vec = mag[i] * direc[i]
            x_mag = np.dot(test_vec, x)
            y_mag = np.dot(test_vec, y)
            if np.absolute(x_mag) >= np.absolute(y_mag):
                if x_mag < 0:
                    mag[i] = -mag[i]
                    direc[i] = -direc[i]
            else:
              if y_mag < 0:
                mag[i] = -mag[i]
                direc[i] = -direc[i]
        '''
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


class Canvas2D:
    '''
    a plot calss use matplotlib as back engine to generate 2D contour plot
    '''
    def __init__(self,
                 plot_data,
                 row_max=MODEL_SIZE_X, col_max=MODEL_SIZE_Y,
                 method="inv_dist_wgt"):
        '''
        initialize a mesh with 100x100 canvas, the default data interpolation
        method is set to nearest
        the data used for plotting should be a nx3 array and have following
        data structure:
            [coord_x, coord_y, texture_id, value]
        other interpolation method:
            1. inverse distance weighte: inv_dist_wgt
        default sampling neighbours has been set to 3, as higher number of 
        sampling points will result in blurring the image. Currently, use 3
        is the best option
        '''
        self.m_row_max = row_max
        self.m_col_max = col_max
        self.m_plot_data = plot_data    # raw data for plotting
        self.m_interpolated_data = np.zeros((self.m_row_max, self.m_col_max))
        self.m_caption = "figure"
        self.m_show_plot = False
        self.m_value_range = np.linspace(0, 0.7071, 100)
        self.m_ticks = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        self.m_sample_num = MY_NEIGHBORS    # number of points used for
                                            # interpolation
        self.m_method = method
        self.set_interpolation()
        self.m_color_map = plt.get_cmap("jet")

    def set_sample_number(self, sample_number):
        '''
        setter for m_sample_num
        '''
        self.m_sample_num = sample_number
        self.set_interpolation()

    def set_color_map(self, color_map):
        '''
        setter for m_color_map
        '''
        self.m_color_map = color_map

    def set_value_range(self, value_range):
        '''
        setter for color/c-axis range
        '''
        self.m_value_range = value_range

    def set_caption(self, caption):
        '''
        setter for caption of the figure
        '''
        self.m_caption = caption

    def set_show_plot(self, key_bool):
        '''
        set if show plot after plotting
        '''
        self.m_show_plot = key_bool

    def set_ticks(self, ticks):
        '''
        setter for ticks of the legend
        '''
        self.m_ticks = ticks

    def set_interpolation(self):
        '''
        set interpolation method used in doing contour plot. Current there are
        several interpolaiton methods available at this point:
            1 "nearest": this method choose the nearest point and use its
                         data for approximation
            2 "inv_dist_wgt": this method use inverse distance as weighed to 
                              approximate the data point of unknown. 
                              Graphically, each known data point can be 
                              visualized as a cone and the data point of 
                              unknown is the final interference of the cones 
                              included (depended on how many data points we 
                              want to sample.
        note:
            It is known that higher order of interpolation can lead to lose of
            details, but it is also can be used to remove noises in the data.
            A combination might be a better approach
        '''
        # assure use at least 2 neighbours when interpolating
        if self.m_sample_num < 2:
            self.m_sample_num = 2
            print "WARNNING: use at least 2 neighbours to interpolate"
        # data interpolate
        if self.m_method == "nearest":
            self.interpolate_nearest()
        elif self.m_method == "inv_dist_wgt":
            self.interpolate_inv_dist()
        else:
            print "unkown interpolation method: {}".format(self.m_method)
            sys.exit(-1)

    def interpolate_nearest(self):
        '''
        Find the nearest neighbour and use its value for approximation
        '''
        # set up KDTree
        search_tree = []
        for temp_data_point in self.m_plot_data:
            temp_x = temp_data_point[0]
            temp_y = temp_data_point[1]
            search_tree.append([temp_x, temp_y])
        # build the tree
        tree = KDTree(search_tree)
        # query
        for i in range(0, self.m_row_max):
            for j in range(0, self.m_col_max):
                temp_coord = np.array([i, j])
                query_res = tree.query(temp_coord)
                search_key = query_res[1]
                temp_value = self.m_plot_data[search_key][3]
                self.m_interpolated_data[i, j] = temp_value
        print "Data Interpolated Finshed with {}".format(self.m_method)

    def interpolate_inv_dist(self):
        '''
        Find the nearest m_sample_num neighbours and use inverse distance for
        data interpolation
        '''
        # set up KDTree
        search_tree = []
        for temp_data_point in self.m_plot_data:
            temp_x = temp_data_point[0]
            temp_y = temp_data_point[1]
            search_tree.append([temp_x, temp_y])
        # build the tree
        tree = KDTree(search_tree)
        for i in range(0, self.m_row_max):
            for j in range(0, self.m_col_max):
                temp_coord = np.array([i, j])
                query_res = tree.query(temp_coord, self.m_sample_num)
                temp_norm = 0
                temp_sum = 0
                # find the closet point texture ID and assign it to the 
                # mesh location
                key_texture = query_res[1][0]
                sample_texture_id = self.m_plot_data[key_texture][2]
                for neighbour_id in range(0, self.m_sample_num):
                    temp_key = query_res[1][neighbour_id]
                    temp_dist = query_res[0][neighbour_id]
                    temp_texture_id = self.m_plot_data[temp_key][2]
                    # if the texture are the same, average the schmid factor
                    # otherwise just skip this point
                    if temp_texture_id == sample_texture_id:
                        temp_sum += self.m_plot_data[temp_key][3]/temp_dist
                        temp_norm += 1/temp_dist
                self.m_interpolated_data[i, j] = temp_sum/temp_norm

    def give_me_plot(self):
        '''
        generate a contour plot for the interpolated data
        '''
        plt.figure()
        image = plt.contourf(self.m_interpolated_data,
                             self.m_value_range,
                             cmap = self.m_color_map,
                             origin = 'upper')
        # set ticks bar
        dummy_holder = plt.colorbar(image, ticks = self.m_ticks)
        # set caption
        plt.title(self.m_caption)
        frame1 = plt.gca()
        frame1.axes.get_xaxis().set_visible(False)
        frame1.axes.get_yaxis().set_visible(False)
        if self.m_show_plot:
            plt.show()
        file_name = self.m_caption + ".png"
        plt.savefig(file_name, dpi=150)
        # close figure and clear the memory
        plt.clf()
        print "image {} saved to ./".format(file_name)


class ColorSpace:
    '''
    a user defined color space for ID different slip system
    '''
    def __init__(self):
        '''
        Linearized color scale for different slip system
        '''
        self.m_prism_red = {'red': ((0.0, 1.0, 1.0),
                                    (1.0, 0.6, 0.6)),
                            'green': ((0.0, 1.0, 1.0),
                                      (1.0, 0.1, 0.1)),
                             'blue': ((0.0, 1.0, 1.0),
                                      (1.0, 0.2, 0.2))}
        self.m_basal_blue = {'red': ((0.0, 1.0, 1.0),
                                      (1.0, 0.1, 0.1)),
                             'green': ((0.0, 1.0, 1.0),
                                       (1.0, 0.2, 0.2)),
                             'blue':  ((0.0, 1.0, 1.0),
                                       (1.0, 0.6, 0.6))}
        self.m_pyramidal_a = {'red': ((0.0, 1.0, 1.0),
                                      (1.0, 0.0, 0.0)),
                              'green': ((0.0, 1.0, 1.0),
                                        (1.0, 0.5, 0.5)),
                              'blue':  ((0.0, 1.0, 1.0),
                                        (1.0, 0.0, 0.0))}
        self.m_pyramidal_ca = {'red': ((0.0, 1.0, 1.0),
                                       (1.0, 0.3, 0.3)),
                               'green': ((0.0, 1.0, 1.0),
                                         (1.0, 0.3, 0.3)),
                               'blue': ((0.0, 1.0, 1.0),
                                        (1.0, 0.0, 0.0))}
        self.m_redyellow = {'red': ((0.0, 1.0, 1.0),
                                    (0.4, 0.5, 0.5),
                                    (1.0, 0.3, 0.3)),
                            'green': ((0.0, 1.0, 1.0),
                                      (0.4, 0.1, 0.1),
                                      (1.0, 0.3, 0.3)),
                            'blue': ((0.0, 1.0, 1.0),
                                     (0.4, 0.1, 0.1),
                                     (1.0, 0.3, 0.3))}
        self.m_wrb = {'red': ((0.0, 1.0, 1.0),
                              (0.5, 0.2, 0.2),
                              (1.0, 1.0, 1.0)),
                      'green': ((0.0, 1.0, 1.0),
                                (0.5, 0.2, 0.2),
                                (1.0, 0.2, 0.2)),
                      'blue': ((0.0, 1.0, 1.0),
                               (0.5, 1.0, 1.0),
                               (1.0, 0.2, 0.2))}

    def get_color(self, color_id):
        '''
        return color dict corresponding to the color ID
        '''
        if "basal" in color_id:
            color_map = LinearSegmentedColormap(
                    "basal_blue", self.m_basal_blue)
        elif "prism" in color_id:
            color_map = LinearSegmentedColormap(
                    "prism_red", self.m_prism_red)
        elif "pyra" in color_id:
            color_map = LinearSegmentedColormap(
                    "pyr_a_green", self.m_pyramidal_a)
        elif "pyrca" in color_id:
            color_map = LinearSegmentedColormap(
                    "Pyr_ca_yel", self.m_pyramidal_ca)
        elif "redyellow" in color_id:
            color_map = LinearSegmentedColormap(
                    "redyellow", self.m_redyellow)
        elif "wrb" in color_id:
            color_map = LinearSegmentedColormap(
                    "wrb", self.m_wrb)
        else:
            print "ERROR: unknown color ID {}".format(color_id)
            sys.exit(-1)
        return color_map


#-----------------------------------------------------------------------------#
# Main Function
#-----------------------------------------------------------------------------#
def main():
    '''
    The main function here is used to read in the post processed data from 
    DAMASK and generate plot/figure acoording to needs
    '''
    parser = argparse.ArgumentParser(prog="CRYSTAL",
                                     description='''
                                     Generating 2D Contour plot based on info
                                     from *.t16 file''',
                                     epilog='''
                                     only support HCP for the moment''')
    parser.add_argument("input_file",
                        help='''
                        location and file name of the post processed ASCII 
                        file from Marc/Mentat''',
                        default="test.txt")
    parser.add_argument("-v", "--version",
                        action="version",
                        version="%(prog)s 0.2")
    parser.add_argument("-d", "--depth",
                        help='''
                        depth of sampled region from sample surface''',
                        default=-2,
                        type=float)
    parser.add_argument("-n", "--name",
                        help='''
                        name of the patch being processed''',
                        default="Victoria")
    parser.add_argument("-O", "--output",
                        help='''
                        specify desired output, currently available options:
                            SF: Schmid Factor Map
                            OE: Orientation evolution map
                            MOE: Mis-orientation evolution map
                            pStress: Vector filed plot for 3 principal stress
                            acShear: Accumulated Shear
                            pStress_separated: separated principal stress
                            shearrate: shear rate for each slip system''',
                        default="SF acShear pStress")
    # parsing arguments from command line
    args = parser.parse_args()
    input_file_name = args.input_file
    cut_depth = args.depth
    output_type_list = args.output.split()
    model_name = args.name
    print "File {} found, start data loading...".format(input_file_name)
    # load data from ASCII file
    try:
        print "Welcome, {}".format(model_name)
        file_stream = open(input_file_name)
        raw_data = file_stream.readlines()[2:]   # remove meta info
        header = raw_data.pop(0) # store header separately
        file_stream.close()
    except:
        print "cannot open file: {}".format(input_file_name)
        sys.exit(-1)
    # Start of Analysis
    print "Data Loading complete, start analyzing..."
    # get the increment num
    current_inc = raw_data[1].split()[0]
    if int(current_inc) < 10:
        current_inc = "00" + current_inc
    elif int(current_inc) < 100:
        current_inc = "0" + current_inc
    print "current processsing increment: {}".format(current_inc)
    # data slicing: select the sample surface
    counter = 0
    # Note:
    #  Currently, the output file is automatically sorted w.r.t z-axis, so 
    #  here is a quick way to select the data that is close to the surface
    for line in raw_data:
        temp_z = float(line.split()[7])
        if temp_z < cut_depth:
            counter += 1
    # container for raw data from ASCII file
    raw_data = raw_data[counter:]
    # contain header from the ASCII table
    header = header.split()
    # load selected data for analysis
    crystallites = []
    for line in raw_data:
        data = line.split()
        coords = [float(item) for item in data[5:8]]
        stress_local = [float(item) for item in data[9:15]]
        texture = float(data[15])
        orientation = [float(item) for item in data[16:19]]
        slip_resistance = [float(item) for item in data[19:43]]
        shear_rate = [float(item) for item in data[43:67]]
        resolved_stress = [float(item) for item in data[67:91]]
        accumulated_shear = [float(item) for item in data[91:]]
        crystallites.append(IntegrationPoint(coords,
                                             texture,
                                             orientation,
                                             stress_local,
                                             accumulated_shear,
                                             shear_rate,
                                             'hcp',
                                             1.58))
    for item in crystallites:
        item.get_principle_stress()
    # data construction complete, start analysis
    if "SF" in output_type_list:
        print "Start plotting Schmid Factor maps"
        schmid_factor_map(crystallites, current_inc)
    if "acShear" in output_type_list:
        print "Start plotting Accumulated Shear maps"
        accumulated_shear_map(crystallites, current_inc)
    if "pStress" in output_type_list:
        print "Start plotting principal stress map"
        principal_stress_map(crystallites, current_inc)
    if "pStress_separated" in output_type_list:
        print "Start plotting separated principal stress field map"
        principal_stress_field(crystallites, current_inc)
    if "MOE" in output_type_list:
        print "Start plotting mis-orientation evolution map"
        misorientation_map(crystallites, current_inc)
    if "shearrate" in output_type_list:
        print "Start plotting shear rate map"
        shear_rate_map(crystallites, current_inc)
    # Good by message
    print "All analysis has completed, Good-bye."
       

def schmid_factor_map(data, increment_num):
    '''
    When 'SF' is toggled in the output list, this method is called to generate
    Schmid Factor map based on the input ASCII table
    vars:
        data: input data list
        current_inc: integer indicating the increment of the working copy
    '''
    color_space = ColorSpace()
    ticks = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    schmid_factor_range = np.linspace(0, 0.75, 100)
    # testing block
    schmid_factor_max = 0.707
    for item in data:
        if schmid_factor_max < max(item.get_schmid_factor_global()[:]):
            schmid_factor_max =  max(item.get_schmid_factor_global()[:])
    print "Highest Local Schmid Factor: {}".format(schmid_factor_max)
    schmid_factor_max = 0.707
    for item in data:
        if schmid_factor_max < max(item.get_schmid_factor_local()[:]):
            schmid_factor_max = max(item.get_schmid_factor_local()[:])
    print "Highest Global Schmid Factor: {}".format(schmid_factor_max)
    # sys.exit(0)
    # generate global schmid factor map for basal slip
    plot_data = []
    for item in data:
        plot_data.append([item.m_coord.m_x,
                          item.m_coord.m_y,
                          item.m_texture,
                          max(item.get_schmid_factor_global()[0:3])])
    schmid_factor_canvas = Canvas2D(plot_data)
    caption = "GlobalSchmidFactor_Basal_INC" + increment_num
    schmid_factor_canvas.set_caption(caption)
    schmid_factor_canvas.set_color_map(color_space.get_color("basal"))
    schmid_factor_canvas.set_ticks(ticks)
    schmid_factor_canvas.set_value_range(schmid_factor_range)
    schmid_factor_canvas.give_me_plot()
    # generate local schmid factor map for basal slip
    plot_data = []
    for item in data:
        plot_data.append([item.m_coord.m_x,
                          item.m_coord.m_y,
                          item.m_texture,
                          max(item.get_schmid_factor_local()[0:3])])
    schmid_factor_canvas = Canvas2D(plot_data)
    caption = "LocalSchmidFactor_Basal_INC" + increment_num
    schmid_factor_canvas.set_caption(caption)
    schmid_factor_canvas.set_color_map(color_space.get_color("basal"))
    schmid_factor_canvas.set_ticks(ticks)
    schmid_factor_canvas.set_value_range(schmid_factor_range)
    schmid_factor_canvas.give_me_plot()
    # generate global schmid factor map for prism slip
    plot_data = []
    for item in data:
        plot_data.append([item.m_coord.m_x,
                          item.m_coord.m_y,
                          item.m_texture,
                          max(item.get_schmid_factor_global()[3:6])])
    schmid_factor_canvas = Canvas2D(plot_data)
    caption = "GlobalSchmidFactor_Prism_INC" + increment_num
    schmid_factor_canvas.set_caption(caption)
    schmid_factor_canvas.set_color_map(color_space.get_color("prism"))
    schmid_factor_canvas.set_ticks(ticks)
    schmid_factor_canvas.set_value_range(schmid_factor_range)
    schmid_factor_canvas.give_me_plot()
    # generate local schmid factor map for prism slip
    plot_data = []
    for item in data:
        plot_data.append([item.m_coord.m_x,
                          item.m_coord.m_y,
                          item.m_texture,
                          max(item.get_schmid_factor_local()[3:6])])
    schmid_factor_canvas = Canvas2D(plot_data)
    caption = "LocalSchmidFactor_Prism_INC" + increment_num
    schmid_factor_canvas.set_caption(caption)
    schmid_factor_canvas.set_color_map(color_space.get_color("prism"))
    schmid_factor_canvas.set_ticks(ticks)
    schmid_factor_canvas.set_value_range(schmid_factor_range)
    schmid_factor_canvas.give_me_plot()
    # generate global schmid factor map for pyramidal a slip
    plot_data = []
    for item in data:
        plot_data.append([item.m_coord.m_x,
                          item.m_coord.m_y,
                          item.m_texture,
                          max(item.get_schmid_factor_global()[6:12])])
    schmid_factor_canvas = Canvas2D(plot_data)
    caption = "GlobalSchmidFactor_Pyramidal_a_INC" + increment_num
    schmid_factor_canvas.set_caption(caption)
    schmid_factor_canvas.set_color_map(color_space.get_color("pyra"))
    schmid_factor_canvas.set_ticks(ticks)
    schmid_factor_canvas.set_value_range(schmid_factor_range)
    schmid_factor_canvas.give_me_plot()
    # generate local schmid factor map for pyramidal a slip
    plot_data = []
    for item in data:
        plot_data.append([item.m_coord.m_x,
                          item.m_coord.m_y,
                          item.m_texture,
                          max(item.get_schmid_factor_local()[6:12])])
    schmid_factor_canvas = Canvas2D(plot_data)
    caption = "LocalSchmidFactor_Pyramidal_a_INC" + increment_num
    schmid_factor_canvas.set_caption(caption)
    schmid_factor_canvas.set_color_map(color_space.get_color("pyra"))
    schmid_factor_canvas.set_ticks(ticks)
    schmid_factor_canvas.set_value_range(schmid_factor_range)
    schmid_factor_canvas.give_me_plot()
    # generate globla schmid factor map for pyramidal c+a slip
    plot_data = []
    for item in data:
        plot_data.append([item.m_coord.m_x,
                          item.m_coord.m_y,
                          item.m_texture,
                          max(item.get_schmid_factor_global()[12:])])
    schmid_factor_canvas = Canvas2D(plot_data)
    caption = "GlobalSchmidFactor_Pyramidal_c+a_INC" + increment_num
    schmid_factor_canvas.set_caption(caption)
    schmid_factor_canvas.set_color_map(color_space.get_color("pyrca"))
    schmid_factor_canvas.set_ticks(ticks)
    schmid_factor_canvas.set_value_range(schmid_factor_range)
    schmid_factor_canvas.give_me_plot()
    # generate local schmid factor map for pyramidal c+a slip
    plot_data = []
    for item in data:
        plot_data.append([item.m_coord.m_x,
                          item.m_coord.m_y,
                          item.m_texture,
                          max(item.get_schmid_factor_local()[12:])])
    schmid_factor_canvas = Canvas2D(plot_data)
    caption = "LocalSchmidFactor_Pyramidal_c+a_INC" + increment_num
    schmid_factor_canvas.set_caption(caption)
    schmid_factor_canvas.set_color_map(color_space.get_color("pyrca"))
    schmid_factor_canvas.set_ticks(ticks)
    schmid_factor_canvas.set_value_range(schmid_factor_range)
    schmid_factor_canvas.give_me_plot()

def shear_rate_map(data, increment_num):
  """generate shear rate map"""
  color_space = ColorSpace()
  ticks = np.log10([1e-4, 1e-1])
  shear_rate_range =  np.linspace(-4, -1, 100)
  # shear rate map for basal slip
  plot_data = []
  for item in data:
    shear_rate = np.log10(map(np.absolute, 
                          [temp+1e-20 for temp in item.m_shear_rate[0:3]]))
    plot_data.append([item.m_coord.m_x,
                      item.m_coord.m_y,
                      item.m_texture,
                      max(shear_rate),
                      ])
  shear_rate_canvas1 = Canvas2D(plot_data)
  caption = "Shear_Rate_Basal_Inc_" + increment_num
  shear_rate_canvas1.set_caption(caption)
  shear_rate_canvas1.set_color_map(color_space.get_color("basal"))
  shear_rate_canvas1.set_value_range(shear_rate_range)
  shear_rate_canvas1.set_ticks(ticks)
  shear_rate_canvas1.give_me_plot()
  # shear rate map for prism
  plot_data = []
  for item1 in data:
    shear_rate = np.log10(map(np.absolute, 
                          [temp+1e-20 for temp in item1.m_shear_rate[3:6]]))
    plot_data.append([item1.m_coord.m_x,
                      item1.m_coord.m_y,
                      item1.m_texture,
                      max(shear_rate),
                      ])
  shear_rate_canvas2 = Canvas2D(plot_data)
  caption = "Shear_Rate_Prism_Inc_" + increment_num
  shear_rate_canvas2.set_caption(caption)
  shear_rate_canvas2.set_color_map(color_space.get_color("prism"))
  shear_rate_canvas2.set_value_range(shear_rate_range)
  shear_rate_canvas2.set_ticks(ticks)
  shear_rate_canvas2.give_me_plot()
  # shear rate map for pyramidal a
  plot_data = []
  for item2 in data:
    shear_rate = np.log10(map(np.absolute, 
                          [temp+1e-20 for temp in item2.m_shear_rate[6:12]]))
    plot_data.append([item2.m_coord.m_x,
                      item2.m_coord.m_y,
                      item2.m_texture,
                      max(shear_rate),
                      ])
  shear_rate_canvas3 = Canvas2D(plot_data)
  caption = "Shear_Rate_Pyramidal_a_Inc_" + increment_num
  shear_rate_canvas3.set_caption(caption)
  shear_rate_canvas3.set_color_map(color_space.get_color("pyra"))
  shear_rate_canvas3.set_value_range(shear_rate_range)
  shear_rate_canvas3.set_ticks(ticks)
  shear_rate_canvas3.give_me_plot()
  # shear rate map for pyramidal c+a
  plot_data = []
  for item3 in data:
    shear_rate = np.log10(map(np.absolute, 
                         [temp+1e-20 for temp in item3.m_shear_rate[12:24]]))
    plot_data.append([item3.m_coord.m_x,
                      item3.m_coord.m_y,
                      item3.m_texture,
                      max(shear_rate),
                      ])
  shear_rate_canvas4 = Canvas2D(plot_data)
  caption = "Shear_Rate_Pyramidal_c+a_Inc_" + increment_num
  shear_rate_canvas4.set_caption(caption)
  shear_rate_canvas4.set_color_map(color_space.get_color("pyrca"))
  shear_rate_canvas4.set_value_range(shear_rate_range)
  shear_rate_canvas4.set_ticks(ticks)
  shear_rate_canvas4.give_me_plot()

def accumulated_shear_map(data, increment_num):
    '''
    generate the accumulated shear map with respect to slip system
    '''
    color_space = ColorSpace()
    ticks = [0.0, 0.1, 0.2, 0.3, 0.4]
    shear_range = np.linspace(0, 0.4, 100)
    # generate accumulated shear map for basal
    plot_data = []
    for item in data:
        plot_data.append([item.m_coord.m_x,
                          item.m_coord.m_y,
                          item.m_texture,
                          sum(item.m_accumulated_shear[0:3])])
    accumulated_shear_canvas = Canvas2D(plot_data)
    caption = "AccumulatedShear_Basal_INC_" + increment_num
    accumulated_shear_canvas.set_caption(caption)
    accumulated_shear_canvas.set_color_map(color_space.get_color("basal"))
    accumulated_shear_canvas.set_ticks(ticks)
    accumulated_shear_canvas.set_value_range(shear_range)
    accumulated_shear_canvas.give_me_plot()
    # generate accumulated shear map for prism
    plot_data = []
    for item in data:
        plot_data.append([item.m_coord.m_x,
                          item.m_coord.m_y,
                          item.m_texture,
                          sum(item.m_accumulated_shear[3:6])])
    accumulated_shear_canvas = Canvas2D(plot_data)
    caption = "AccumulatedShear_Prism_INC_" + increment_num
    accumulated_shear_canvas.set_caption(caption)
    accumulated_shear_canvas.set_color_map(color_space.get_color("prism"))
    accumulated_shear_canvas.set_ticks(ticks)
    accumulated_shear_canvas.set_value_range(shear_range)
    accumulated_shear_canvas.give_me_plot()
    # generate accumulated shear map for pyramidal a
    plot_data = []
    for item in data:
        plot_data.append([item.m_coord.m_x,
                          item.m_coord.m_y,
                          item.m_texture,
                          sum(item.m_accumulated_shear[6:12])])
    accumulated_shear_canvas = Canvas2D(plot_data)
    caption = "AccumulatedShear_Pyramidal_a_INC_" + increment_num
    accumulated_shear_canvas.set_caption(caption)
    accumulated_shear_canvas.set_color_map(color_space.get_color("pyra"))
    accumulated_shear_canvas.set_ticks(ticks)
    accumulated_shear_canvas.set_value_range(shear_range)
    accumulated_shear_canvas.give_me_plot()
    # generate accumulated shear map for pyramidal c+a
    plot_data = []
    for item in data:
        plot_data.append([item.m_coord.m_x,
                          item.m_coord.m_y,
                          item.m_texture,
                          sum(item.m_accumulated_shear[12:])])
    accumulated_shear_canvas = Canvas2D(plot_data)
    caption = "AccumulatedShear_Pyramidal_c+a_INC_" + increment_num
    accumulated_shear_canvas.set_caption(caption)
    accumulated_shear_canvas.set_color_map(color_space.get_color("pyrca"))
    accumulated_shear_canvas.set_ticks(ticks)
    accumulated_shear_canvas.set_value_range(shear_range)
    accumulated_shear_canvas.give_me_plot()


def principal_stress_map(data, inc_num):
    '''
    generate a vector field plot for all three principal stress
    Note:
        Use nearest interpolation to get regular data
    '''
    # set up data container
    colorspace = ColorSpace()
    pstress_list = []
    search_tree = []
    plot_xt = []
    plot_yt = []
    plot_ut = []
    plot_vt = []
    plot_nt = []
    plot_xc = []
    plot_yc = []
    plot_uc = []
    plot_vc = []
    plot_nc = []
    for item in data:
        search_tree.append([item.m_coord.m_y, -item.m_coord.m_x])
        pstress_list.append(item.get_principle_stress())
    # build the tree
    tree = KDTree(search_tree)
    # query
    for i in range(0, 145, 14):
        for j in range (0, 135, 13):
            temp_coord = np.array([i, -j])
            query_res = tree.query(temp_coord)
            search_key = query_res[1]
            temp_pstress = pstress_list[search_key]
            for key in temp_pstress:
                temp_mag = key
                temp_stress = np.array(temp_pstress[key])
                temp_stress = temp_stress[0:2]/LA.norm(temp_stress[0:2])
                # store plot data
                if temp_mag >= 0:
                  plot_xt.append(i)
                  plot_yt.append(-j)
                  plot_nt.append(temp_mag)
                  plot_ut.append(temp_stress[1])
                  plot_vt.append(-temp_stress[0])
                  # the other half principal stress
                  plot_xt.append(i)
                  plot_yt.append(-j)
                  plot_nt.append(temp_mag)
                  plot_ut.append(-temp_stress[1])
                  plot_vt.append(temp_stress[0])
                else:
                  plot_xc.append(i)
                  plot_yc.append(-j)
                  plot_nc.append(-temp_mag)
                  plot_uc.append(temp_stress[1])
                  plot_vc.append(-temp_stress[0])
                  # the other half principal stress
                  plot_xc.append(i)
                  plot_yc.append(-j)
                  plot_nc.append(-temp_mag)
                  plot_uc.append(-temp_stress[1])
                  plot_vc.append(temp_stress[0])
    # read for plot
    plt.figure()
    pstress_map_c = plt.quiver(plot_xc, plot_yc, 
                               plot_uc, plot_vc, plot_nc,
                               scale=25, pivot="tip", 
                               cmap=plt.get_cmap("Blues"))
    plt.ylim((-140, 5))
    plt.xlim((-5, 150))
    plt.clim((0.0, 0.8))
    plt.colorbar(pstress_map_c)
    pstress_map_t = plt.quiver(plot_xt, plot_yt, 
                               plot_ut, plot_vt, plot_nt,
                               scale=25, pivot="tail", 
                               cmap=plt.get_cmap("Reds"))
    plt.clim((0.0, 0.8))
    plt.colorbar(pstress_map_t)
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    frame.axes.set_aspect('equal')
    title = "PrincipalStress_" + str(inc_num)
    plt.title(title)
    file_name = title + ".png"
    plt.savefig(file_name, dpi=200)
    plt.clf()
    print "save {} to ./".format(file_name)


def principal_stress_field(data, increment_num):
    '''
    generating vector field plot for the principal stress
    '''
    #
    # testing vector field plot here
    #
    colorspace = ColorSpace()
    x = []
    y = []
    u1 = []
    u2 = []
    u3 = []
    v1 = []
    v2 = []
    v3 = []
    n1 = []
    n2 = []
    n3 = []
    for item in data:
        x.append(-item.m_coord.m_x)
        y.append(item.m_coord.m_y)
        pstress = item.get_principle_stress()
        max_stress = max(np.absolute(pstress.keys()))
        min_stress = min(np.absolute(pstress.keys()))
        for key in pstress:
            if key < 0:
                temp_mag = np.absolute(key)
                temp_stress = -np.array(pstress[key])
            else:
                temp_mag = key
                temp_stress = np.array(pstress[key])
            # get u v n for each principle stress component
            if temp_mag == max_stress:
                n1.append(temp_mag)
                u1.append(temp_stress[0])
                v1.append(temp_stress[1])
            elif temp_mag == min_stress:
                n3.append(temp_mag)
                u3.append(temp_stress[0])
                v3.append(temp_stress[1])
            else:
                n2.append(temp_mag)
                u2.append(temp_stress[0])
                v2.append(temp_stress[1])
    # Plot Sigma_1
    plt.figure()
    sigma_1 = plt.quiver(y[::2], x[::2], v1[::2], u1[::2], n1[::2],
               scale=40, cmap=colorspace.get_color("prism"), pivot="tail")
    plt.colorbar(sigma_1)
    plt.ylim((-140, 5))
    plt.xlim((-5,150))
    plt.clim((0.0,3.0))
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    frame.axes.set_aspect('equal')
    title = "PrincipalStress1_" + str(increment_num)
    plt.title(title)
    file_name = title + ".png"
    plt.savefig(file_name)
    plt.clf()
    print "save {} to ./".format(file_name)
    # Plot Sigma_2
    plt.figure()
    sigma_2 = plt.quiver(y[::2], x[::2], v2[::2], u2[::2], n2[::2],
               scale=40, cmap=colorspace.get_color("pyra"), pivot="tail")
    plt.colorbar(sigma_2)
    plt.ylim((-140, 5))
    plt.xlim((-5,150))
    plt.clim((0.0,2.5))
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    frame.axes.set_aspect('equal')
    title = "PrincipalStress2_" + str(increment_num)
    plt.title(title)
    file_name = title + ".png"
    plt.savefig(file_name)
    plt.clf()
    print "save {} to ./".format(file_name)
    # Plot Sigma_3
    plt.figure()
    sigma_3 = plt.quiver(y[::2], x[::2], v3[::2], u3[::2], n3[::2],
               scale=40, cmap=colorspace.get_color("basal"), pivot="tail")
    plt.colorbar(sigma_3)
    plt.ylim((-140, 5))
    plt.xlim((-5,150))
    plt.clim((0.0,2.5))
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    frame.axes.set_aspect('equal')
    title = "PrincipalStress3_" + str(increment_num)
    plt.title(title)
    file_name = title + ".png"
    plt.savefig(file_name)
    plt.clf()
    print "save {} to ./".format(file_name)


def misorientation_map(data, inc_num):
    '''
    plot the mis-orientation evolution map based on the initial orientation
    from "material.config" file, which need to provide in the current folder.
    '''
    try:
        istream = open("material.config")
    except:
        print "cannot open material.config file, make sure file exists."
        sys.exit(-1)
    initial_orientation = {}
    # use a Macro number defined at the beginning to find the location of 
    # texture section, this is a really bad way to processing a configuration
    # file, but for now we will just use the simple method
    raw_data = istream.readlines()[MATERIAL_CONFIG_TEXTURE:]
    for index in range(3, len(raw_data), 3):
        string = raw_data[index].split()
        temp_eulerang = [string[2], string[4], string[6]]
        initial_orientation[index/3] = [float(item) for item in temp_eulerang]
    # prepare for plotting
    color_space = ColorSpace()
    crange = np.linspace(0,15,100)
    ticks = [0, 5, 10, 15]
    plot_data = []
    for item in data:
        plot_data.append([item.m_coord.m_x,
                          item.m_coord.m_y,
                          item.m_texture,
                          item.get_misorientation(initial_orientation)])
    misorientation_canvas = Canvas2D(plot_data)
    caption = "Mis-orientation_Evolution_" + inc_num
    misorientation_canvas.set_caption(caption)
    misorientation_canvas.set_color_map(plt.get_cmap('binary'))
    misorientation_canvas.set_value_range(crange)
    misorientation_canvas.set_ticks(ticks)
    misorientation_canvas.give_me_plot()
#-----------------------------------------------------------------------------#
# End of Main entrance
#-----------------------------------------------------------------------------#
if __name__ == "__main__":
    main()
