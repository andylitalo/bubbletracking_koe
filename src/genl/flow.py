# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 11:47:30 2019

These functions define mathematical functions that compute properties of
flowing streams such as flow rate, pressure, flow speed, and radius (for sheath
flows).

@author: Andy
"""

import numpy as np



########################## FUNCTION DEFINTIONS #################################


def calc_p(p_in, p_atm, v, t, L):
    """
    Calculates the pressure down the observation capillary
    assuming a linear pressure drop based on the estimated
    inlet (p_in) and outlet (p_atm) pressures.

    Parameters:
        p_in : float
            inlet pressure, estimated using the flow.sheath_eqns [Pa]
        p_atm : float
            outlet pressure, estimated to be atmospheric pressure [Pa]
        v : float
            velocity of inner stream, estimated using flow.sheath_eqns [m/s]
        t : float
            time since entering the observation capillary [s]
        L : float
            length of observation capillary [m]

    Returns:
        p : float
            pressure at current point along observation capillary [Pa]
    """
    return p_in + v*t/L*(p_atm-p_in)


def calc_t_s(p_in, p_atm, p_s, v, L):
    """
    Calculates time at which pressure reaches saturation pressure.
    """
    return (p_in - p_s)/(p_in - p_atm)*L/v


def sheath_eqns(eta_i, eta_o, L, p_atm, p_in, Q_i, Q_o, R_i, R_o, v):
    """
    Defines equations derived from Navier-Stokes equations in
    Stokes flow for sheath flow down a cylindrical pipe. The
    derivation is given in YlitaloA_candidacy_report.pdf.

    Parameters
    ----------
    p_in : float
        inlet pressure [Pa]
    v : float
        velocity of center stream [m/s]
    R_i : float
        radius of inner stream [m]
    p_atm : float
        outlet pressure, assumed to be atmospheric [Pa]
    L : float
        length of observation capillary [m]
    eta_i : float
        viscosity of inner stream of polyol-CO2 [Pa.s]
    eta_o : float
        viscosity of outer stream of pure polyol [Pa.s]
    Q_i : float
        inner stream flow rate, supplied by ISCO 100 DM [m^3/s]
    Q_o : float
        outer stream flow rate, supplied by ISCO 260 D [m^3/s]
    R_o : float
        radius of outer stream (half of inner diameter of observation capillary) [m]

    Returns
    -------
    res : 3-tuple of floats
        Residuals of the three flow equations (Poiseuille flow from Stokes
        equation + 2 BCs: Q_i through the inner stream, and Q_o through the
        outer stream BC)
    """
    # boundary condition that the outer stream has flow rate Q_o
    res1 = Q_o - np.pi*(p_in-p_atm)*(R_o**2 - R_i**2)**2/(8*eta_o*L)
    # boundary condition that the inner stream has flow rate Q_i
    res2 = (p_in - p_atm)/L - (8*eta_i*Q_i)/ \
                ( np.pi*R_i**2*(2*(R_o**2 - R_i**2)*eta_i/eta_o + R_i**2) )
    # residual from Stokes flow (v = w_i(r = 0) for w_i z-velocity from Stokes)
    res3 = v - (p_in - p_atm)/L*( (R_o**2 - R_i**2)/(4*eta_o) + \
                R_i**2/(4*eta_i) )
    res = (res1, res2, res3)

    return res


def sheath_eqns_input(variables, args):
    """
    Formats the input to flow_eqns for use in scipy.optimize.root, which
    requires functions to have the format foo(vars, args).
    Formatting is performed by merging variables and arguments according to
    the ordering given as the last argument so that the arguments passed to
    flow_eqns are ordered properly (alphabetically).

    Parameters
    ----------
    variables : 3-tuple of floats
        Quantities to solve for with scipy.optimize.root
    args : 8-tuple of 7 floats followed by a list of 10 ints
        The first 7 floats are the remaining quantities required for sheath_eqns,
        which are provided by the user and held constant.
        The list of 10 ints at the end gives the ordering of each variable in
        alphabetical order so that when variables and args are merged, they are
        in the proper order required for sheath_eqns.

        Example:
            variables = (p_in, v, R_i)
            args = (p_atm, L, eta_i, eta_o, Q_i, Q_o, R_cap, ordering)
            ordering = [5, 6, 4, 3, 0, 7, 8, 2, 9, 1]

    Returns
    -------
    res : 3-tuple of floats
        Residuals of the Stokes flow equation and boundary conditions
        calculated in sheath_eqns.
    """
    # extracts the ordering of the variables/args
    ordering = np.array(args[-1])
    # merges variables and args (except for the ordering)
    unordered_args = list(variables) + list(args[:-1])
    # orders the merged variables and args in alphabetical order, which
    # requires numpy arrays
    ordered_args = list(np.array(unordered_args)[ordering])

    return sheath_eqns(*ordered_args)


def get_dp_R_i_v_max(eta_i, eta_o, L, Q_i, Q_o, R_o, SI=False):
    """
    Computes the pressure drop down the observation capillary, the radius of the inner
    stream, and the maximum velocity (found at the center of the inner stream).
    The equations were obtained by solving the Navier-Stokes equations in Stokes flow
    in a cylindrical geometery with sheath flow, assuming no-slip along the walls and
    no-stress along the interface between inner and outer streams. See
    20200323_cap_dim_calc.nb Mathematica notebook for details.

    Parameters
    ----------
    eta_i : float
        viscosity of inner stream [Pa.s]
    eta_o : float
        viscosity of outer stream [Pa.s]
    L : float
        length of observation capillary [cm]
    Q_i : float
        flow rate of inner stream [uL/min]
    Q_o : float
        flow rate of outer stream [uL/min]
    R_o : float
        radius of outer stream (half of inner diameter of capillary) [um]
    SI : bool (opt)
        If True, treats all values as their SI units instead of the units given here

    Returns
    -------
    dp : float
        pressure drop from entrance of observation capillary to exit [bar]
    R_i : float
        radius of inner stream [um]
    v_max : float
        maximum velocity (center of inner stream) [mm/s]

    """
    # converts parameters to SI if not already done so
    if not SI:
        L /= 1E2 # converts from cm to m
        Q_i /= 60E9 # converts from uL/min to m^3/s
        Q_o /= 60E9 # converts from uL/min to m^3/s
        R_o /= 1E6 # converts from um to m

    # calculates pressure drop along observation capillary (<0) [Pa]
    dp = -(8*L)/(np.pi*R_o**6*eta_o) * (Q_i*R_o**2*eta_i*eta_o \
         + 2*(eta_o-eta_i)*np.sqrt( Q_o*R_o**4*eta_i*(Q_o*eta_i + Q_i*eta_o) )\
         + Q_o*R_o**2*(2*eta_i**2 - 2*eta_i*eta_o + eta_o**2))
    # calculates the inner stream radius [m]
    R_i = np.sqrt( (Q_i*R_o**2*eta_i + Q_o*R_o**2*eta_i - \
                    np.sqrt( Q_o*R_o**4*eta_i*(Q_o*eta_i + Q_i*eta_o) ))/\
                  (Q_i*eta_i + 2*Q_o*eta_i - Q_o*eta_o) )
    # calculates the maximum velocity along the inner stream [m/s]
    v_max = 2*( Q_o*R_o**2*eta_i**2 + Q_i*R_o**2*eta_i*eta_o + (eta_o-eta_i)* \
               np.sqrt( Q_o*R_o**4*eta_i * (Q_o*eta_i + Q_i*eta_o) ) )/ \
            (np.pi*R_o**4*eta_i*eta_o)

    # converts parameters to other units if SI not desired
    if not SI:
        dp /= 1E5 # converts from Pa to bar
        R_i *= 1E6 # converts from m to um
        v_max *= 1E3 # converts from m/s to mm/s

    return dp, R_i, v_max


def get_flow_dir(pts):
    """
    Returns the flow direction (row, col) given 4 points defining the channel
    boundary. The flow direction is assumed to be parallel to the more
    horizontal of the boundaries of the mask.

    Parameters
    ----------
    pts : list
        List of (x,y) values of the vertices that define the boundaries of the
        channel.

    Returns
    -------
    flow_dir : 2-tuple of floats
        Normalized vector (row, col) pointing in the direction of the flow
        (assumed to be along the average of the horizontal boundaries)
    """
    # categorizes points


def get_flow_rates_diff_mu(R_i, R_o, eta_i, eta_o, dp, L):
    """
    Computes the flow rates of inner and outer streams given properties
    of the flow. Allows for different viscosities for inner and outer
    streams. Assumes SI units.

    Inputs:
        R_i : inner stream radius [m]
        R_o : outer stream radius [m]
        eta_i : inner stream viscosity [Pa.s]
        eta_o : outer stream viscosity [Pa.s]
        dp : pressure drop [Pa]
        L : length of observation capillary [m]
    Returns:
        Q_i : flow rate of inner stream [m^3/s]
        Q_o : flow rate of outer stream [m^3/s]
    """
    # calculates pressure gradient [Pa/m]
    G = dp / L
    # inner stream flow rate [m^3/s]
    Q_i = 2*np.pi*( (G*(R_o**2-R_i**2)*R_i**2)/(8*eta_o) + (G*R_i**4)/(16*eta_i) )
    # outer stream flow rate [m^3/s]
    Q_o = np.pi*G/(8*eta_o)*(R_o**2-R_i**2)**2

    return Q_i, Q_o

def get_flow_rates_fixed_speed(d_inner, v_center=1.0, ID=500):
    """
    Computes the flow rates of inner and outer streams given the width of the
    inner stream (d_inner) in um, the velocity at the center of the stream
    (v_center) in m/s, and the inner diameter (ID)in um of the channel.

    Assumes Newtonian fluid and same viscosities for inner and outer streams.

    returns:
        Q_i = flow rate of inner stream in mL/min
        Q_o = flow rate of outer stream in mL/min
    """
    # convert to SI
    R_i_m = d_inner/2/1E6
    R_m = ID/2/1E6

    # compute flow rates
    Q_i_m3s = np.pi*v_center*R_i_m**2*(1-0.5*(R_i_m/R_m)**2)
    Q_o_m3s = 0.5*np.pi*v_center*R_m**2 - Q_i_m3s

    # convert units to mL/min
    Q_i = Q_i_m3s*60E6
    Q_o = Q_o_m3s*60E6

    return Q_i, Q_o

def get_flow_rates_ri_dp(eta, r_i, dp, r_obs_cap=250, l_obs_cap=10):
    """
    Computes the flow rates of inner and outer streams given the radius of the
    inner stream (r_i) in um, the pressure drop down the observation capillary
    (dp) in bar, the radius of the observation capillary (r_obs_cap) in um,
    and the length of the observation capillary (l_obs_cap) in cm. Note that the
    flow rates do not depend on the additional tubing since we are specifically
    controlling the pressure drop down the observation capillary.

    Assumes Newtonian fluid and same viscosities for inner and outer streams.

    returns:
        Q_i = flow rate of inner stream in mL/min
        Q_o = flow rate of outer stream in mL/min
    """
    # convert to SI
    r_i /= 1E6 # um -> m
    dp *= 1E5 # bar -> Pa
    r_obs_cap /= 1E6 # um -> m
    l_obs_cap /= 100 # cm -> m

    # compute flow rates [m^3/s]
    Q_i = np.pi*dp*r_i**2/(4*eta*l_obs_cap)*(r_obs_cap**2 - 0.5*r_i**2)
    Q_o = np.pi*r_obs_cap**4/(8*eta*l_obs_cap)*dp - Q_i

    # convert units to uL/min
    Q_i *= 60E9
    Q_o *= 60E9

    return Q_i, Q_o

def get_flow_rates(eta, p_i=None, Q_i=None, p_o=None, Q_o=None, l_obs_cap=10,
    r_obs_cap=250, l_inner_cap=2.3, r_inner_cap=280, l_tube_i=20,
    r_tube_i=481.25, l_tube_o=20, r_tube_o=481.25):
    """
    Gives flow rates for inner and outer streams given system parameters.
    Assumes uniform viscosity, Newtonian fluids, and outlet to atmospheric
    pressure.

    Equations were solved using Mathematica, the results of which can be found
    in the file "flow_p_q_eqns" in the same folder as this file ("Calculations").

    inputs:
        eta         :   viscosity of fluid [Pa.s]
        p_i         :   pressure at inner stream source [bar]
        Q_i         :   flow rate of inner stream [uL/min]
        p_o         :   pressure at outer stream source [bar]
        Q_o         :   flow rate of outer stream [uL/min]
        l_obs_cap   :   length of observation capillary [cm]
        r_obs_cap   :   inner radius of observation capillary [um]
        l_inner_cap :   length of inner capillary [cm]
        r_inner_cap :   inner radius of inner capillary [um]
        l_tube_i    :   length of tubing for inner stream (source to inner
                            capillary) [cm]
        r_tube_i    :   inner radius of tubing for inner stream
        l_tube_o    :   length of tubing for outer stream (source to
                            microfluidic device/acrylic block) [cm]
        r_tube_o    :   inner radius of tubing for outer stream [um]

    returns:
        Q_i         :   flow rate of inner stream [uL/min]
        Q_o         :   flow rate of outer stream [uL/min]
    """
    # ensure that only one of pressure or flow rate is given
    assert (p_i is None) != (Q_i is None), "Provide only one: p_i or Q_i."
    assert (p_o is None) != (Q_o is None), "Provide only one: p_o or Q_o."

    # CONVERT TO SI
    l_obs_cap /= 100 # cm -> m
    r_obs_cap /= 1E6 # um -> m
    l_inner_cap /= 100 # cm -> m
    r_inner_cap /= 1E6 # um -> m
    l_tube_i /= 100 # cm -> m
    r_tube_i /= 1E6 # um -> m
    l_tube_o /= 100 # cm -> m
    r_tube_o /= 1E6 # um -> m

    # inner and outer pressures given
    if (p_i is not None) and (p_o is not None):
        # CONVERT TO SI
        p_i *= 1E5 # bar -> Pa
        p_o *= 1E5 # bar -> Pa

        # Calculate flow rates
        num_Q_i = np.pi*r_tube_i**4*r_inner_cap**4*(l_obs_cap*(p_i - p_o)* \
                    r_tube_o**4 + l_tube_o*p_i*r_obs_cap**4)
        num_Q_o = np.pi*r_tube_o**4*(l_obs_cap*(p_o - p_i)*r_tube_i**4*\
                    r_inner_cap**4 + p_o*(l_inner_cap*r_tube_i**4 + \
                    l_tube_i*r_inner_cap**4)*r_obs_cap**4)
        denom = (8*eta*(l_inner_cap*r_tube_i**4* \
                (l_obs_cap*r_tube_o**4 + l_tube_o*r_obs_cap**4) + \
                r_inner_cap**4*(l_tube_o*l_obs_cap*r_tube_i**4 + \
                l_tube_i*l_obs_cap*r_tube_o**4 + \
                l_tube_i*l_tube_o*r_obs_cap**4)))
        Q_i = num_Q_i / denom
        Q_o = num_Q_o / denom

    # given inner stream pressure and outer stream flow rate
    elif (p_i is not None) and (Q_o is not None):
        # CONVERT TO SI
        p_i *= 1E5
        Q_o /= 60E9

        # calculate the flow rate of the inner stream [m^3/s]
        Q_i = (r_tube_i**4*r_inner_cap**4*(p_i*np.pi*r_obs_cap**4 - \
                8*l_obs_cap*Q_o*eta)) / \
                (8*eta*(l_obs_cap*r_tube_i**4*r_inner_cap**4 + \
                (l_inner_cap*r_tube_i**4 + l_tube_i*r_inner_cap**4)*r_obs_cap**4))

    # given inner stream flow rate and outer stream pressure
    elif (Q_i is not None) and (p_o is not None):
        # CONVERT TO SI
        Q_i /= 60E9
        p_o *= 1E5

        # calculate the flow rate of the outer stream [m^3/s]
        Q_o = (p_o*np.pi*r_tube_o**4 - 8*eta*l_obs_cap*(r_tube_o/r_obs_cap)**4*Q_i) / \
                (8*eta*(l_obs_cap*(r_tube_o/r_obs_cap)**4 + l_tube_o))

    elif (Q_i is not None) and (Q_o is not None):
        # CONVERT TO SI
        Q_i /= 60E9
        Q_o /= 60E9
    else:
        print("'if' statements failed to elicit a true response.")

    # CONVERT FROM M^3/S -> UL/MIN
    Q_i *= 60E9
    Q_o *= 60E9

    return Q_i, Q_o

def test_get_flow_rates():
    """
    Tests the method "get_flow_rates()".
    """
    # test values for inner stream and outer stream pressures [bar]
    p_i = 12
    p_o = 10
    # test value for viscosity
    eta = 1.412

    # get flow rates
    Q_i, Q_o = get_flow_rates(eta, p_i=p_i, p_o=p_o)
    print("Flow rates are Q_i = {Q_i} uL/min and Q_o = {Q_o} uL/min."\
        .format(Q_i=Q_i, Q_o=Q_o))

    # Test 1: compare pressures
    p_i_1, p_o_1, p_inner_cap, p_obs_cap = get_pressures(eta, Q_i, Q_o)
    # print result of test 1
    print("Test 1")
    print("Test values of pressure:")
    print("p_i = {p_i} bar and p_o = {p_o} bar."\
        .format(p_i=p_i, p_o=p_o))
    print("Resulting values for the pressure:")
    print("p_i = {p_i_1} bar and p_o = {p_o_1} bar."\
        .format(p_i_1=p_i_1, p_o_1=p_o_1))

    # Test 2: compare flow rates given inner pressure and outer flow rate
    Q_i_2, Q_o_2 = get_flow_rates(eta, p_i=p_i, Q_o=Q_o)
    print("Test 2: flow rates given inner pressure and outer flow rate")
    print("Q_i = {Q_i_2} uL/min and Q_o = {Q_o_2} uL/min." \
        .format(Q_i_2=Q_i_2, Q_o_2=Q_o_2))

    # Test 3: compare flow rates given outer pressure and inner flow rate
    Qi3, Qo3 = get_flow_rates(eta, p_o=p_o, Q_i=Q_i)
    print("Test 3: flow rates given outer pressure and inner flow rate")
    print("Q_i = {Qi3} uL/min and Q_o = {Qo3} uL/min." \
            .format(Qi3=Qi3, Qo3=Qo3))

    return

def get_pressures(eta, Q_i, Q_o, l_obs_cap=10,
    r_obs_cap=250, l_inner_cap=2.3, r_inner_cap=280, l_tube_i=20,
    r_tube_i=481.25, l_tube_o=20, r_tube_o=481.25):
    """
    inputs:
        eta         :   viscosity of fluid [Pa.s]
        Q_i         :   flow rate of inner stream [uL/min]
        Q_o         :   flow rate of outer stream [uL/min]
        l_obs_cap   :   length of observation capillary [cm]
        r_obs_cap   :   inner radius of observation capillary [um]
        l_inner_cap :   length of inner capillary [cm]
        r_inner_cap :   inner radius of inner capillary [um]
        l_tube_i    :   length of tubing for inner stream (source to inner
                            capillary) [cm]
        r_tube_i    :   inner radius of tubing for inner stream
        l_tube_o    :   length of tubing for outer stream (source to
                            microfluidic device/acrylic block) [cm]
        r_tube_o    :   inner radius of tubing for outer stream [um]

    returns:
        p_i         :   pressure at source of inner stream [bar]
        p_o         :   pressure at source of outer stream [bar]
        p_inner_cap :   pressure at inlet to inner capillary [bar]
        p_obs_cap   :   pressure at inlet to observation capillary [bar]
                                *assumes no pressure drop down microfluidic device
    """
    # CONVERT TO SI
    Q_i /= 60E9 # uL/min -> m^3/s
    Q_o /= 60E9 # uL/min -> m^3/s
    l_obs_cap /= 100 # cm -> m
    r_obs_cap /= 1E6 # um -> m
    l_inner_cap /= 100 # cm -> m
    r_inner_cap /= 1E6 # um -> m
    l_tube_i /= 100 # cm -> m
    r_tube_i /= 1E6 # um -> m
    l_tube_o /= 100 # cm -> m
    r_tube_o /= 1E6 # um -> m

    # compute pressures using Poiseuille flow pressure drop starting from end
    p_obs_cap = 8*eta*l_obs_cap/(np.pi*r_obs_cap**4)*(Q_i+Q_o)
    p_inner_cap = p_obs_cap + 8*eta*l_inner_cap/(np.pi*r_inner_cap**4)*Q_i
    p_o = p_obs_cap + 8*eta*l_tube_o/(np.pi*r_tube_o**4)*Q_o
    p_i = p_inner_cap + 8*eta*l_tube_i/(np.pi*r_tube_i**4)*Q_i

    # convert from Pa to bar
    p_obs_cap /= 1E5
    p_inner_cap /= 1E5
    p_o /= 1E5
    p_i /= 1E5

    return p_i, p_o, p_inner_cap, p_obs_cap


def get_inner_stream_radius(Q_i, Q_o, r_obs_cap=250):
    """
    Calculates the radius of the inner stream in the observation capillary given
    the flow rates of the inner and outer streams.

    Assumes Newtonian fluids with the same viscosity.

    inputs:
        Q_i         :   flow rate of inner stream [just must be same units as Q_o]
        Q_o         :   flow rate of outer stream [just must be same units as Q_i]
        r_obs_cap   :   inner radius of observation capillary [um]

    returns:
        r_inner_stream  :   radius of the inner stream [um]
    """
    # calculate inner stream using equation A.14 in candidacy report
    r_inner_stream = r_obs_cap*np.sqrt(1 - np.sqrt(Q_o/(Q_i + Q_o)))

    return r_inner_stream

def get_velocity(Q_i, Q_o, r_obs_cap=250):
    """
    Calculates the velocity at the center of the inner stream given the flow
    rates.

    inputs:
        Q_i         :   flow rate of inner stream [uL/min]
        Q_o         :   flow rate of outer stream [uL/min]
        r_obs_cap   :   inner radius of observation capilary [um]

    returns:
        v_center    :   velocity at center of inner stream [m/s]
    """
    # CONVERT TO SI
    Q_i /= 60E9 # uL/min -> m^3/s
    Q_o /= 60E9 # uL/min -> m^3/s
    r_obs_cap /= 1E6 # um -> m

    # maximum velocity in exit capillary [m/s]
    v_center = 2*(Q_o+Q_i)/(np.pi*r_obs_cap**2)
    # convert m/s -> cm/s
    v_center *= 100

    return v_center


def p_pois(eta, L, R, Q):
    """
    Computes the pressure expected down a tube with the given
    parameters based on Poiseuille's law (assumes Newtonian,
    single viscosity). Based on SI units.

    Inputs:
        eta : viscosity [Pa.s]
        L : length of tube [m]
        R : inner radius of tube [m]
        Q : flow rate through tube [m^3/s]
    Returns:
        pressure drop based on Poiseuille flow [Pa]
    """
    return 8*eta*L/(np.pi*R**4)*Q

if __name__=='__main__':
    Q_i, Q_o = get_flow_rates_fixed_speed(20)

    print('Inner flow rate = %.4f mL/min and outer flow rate = %.2f mL/min' % (Q_i,Q_o))


def v_inner(Q_i, Q_o, eta_i, eta_o, R_o, L):
    """
    Computes the velocity at the interface between inner and outer streams.
    Assumes SI units.

    Parameters
    ----------
    Q_i : float
        Inner stream flow rate [m^3/s]
    Q_o : float
        Outer stream flow rate [m^3/s]
    eta_i : float
        Inner stream viscosity [Pa.s]
    eta_o : float
        Outer stream viscosity [Pa.s]
    R_o : float
        Outer stream radius (radius of capillary) [m]
    L : float
        Length of observation capillary [m]

    Returns
    -------
    v_inner : float
        Velocity at the interface of the inner stream [m/s]
    """
    # computes pressure drop and inner stream radius
    dp, R_i, _ = get_dp_R_i_v_max(eta_i, eta_o, L, Q_i, Q_o, R_o, SI=True)
    G = -dp/L # pressure gradient [Pa/m]
    v_inner = G*(R_o**2 - R_i**2) / (4*eta_o) # eqn A.10 p. 35 of YlitaloA_candidacy_report.pdf

    return v_inner
