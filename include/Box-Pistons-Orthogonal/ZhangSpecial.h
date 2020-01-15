/** \file
 * \author Nathan Zimmerberg (nhz2@cornell.edu)
 * \date 13 Jan 2020
 * \brief Class to hold constant pressure/ surface tension/ interface shape ZhangSpecial parameters and helper functions.
   from J. Chem. Phys. 103, 10252 (1995); https://doi.org/10.1063/1.469927.
 */
#pragma once
#include "BoxPistonBase.h"

namespace boxpiston
{
/**
from Zhang et al J. Chem. Phys. 103, 10252 (1995); https://doi.org/10.1063/1.469927.
parameters:
  pressure(double): reference normal pressure.
  surface_tension(double): reference surface tension.
  k(double): Hyy/Hxx.

Q1: Hzz
Q2: Hxx

PE_box(Q): k*Q2*Q2*Q1*pressure-surface_tension*k*Q2*Q2

H(Q):
[Q2 0 0 ]
[0 k*Q2 0 ]
[0 0 Q1 ]

F_pres1:
-k*Q2*Q2*pressure

F_pres2:
-2*k*Q2*Q1*pressure+2*surface_tension*k*Q2

Q1(H):
Hzz

Q2(H):
Hxx

dH(Q1)/d(Q1):
[0 0 0 ]
[0 0 0 ]
[0 0 1 ]

dH(Q2)/d(Q2):
[1 0 0 ]
[0 k 0 ]
[0 0 0 ]

Hdot:
[Q2dot 0 0 ]
[0 k*Q2dot 0 ]
[0 0 Q1dot ]

Q1dot:
Hzzdot

Q2dot:
Hxxdot

H^-1:
[1/Q2 0 0 ]
[0 1/k/Q2 0 ]
[0 0 1/Q1 ]

H^-1*dH(Q1)/d(Q1):
[0 0 0 ]
[0 0 0 ]
[0 0 1/Q1 ]

H^-1*dH(Q2)/d(Q2):
[1/Q2 0 0 ]
[0 1/Q2 0 ]
[0 0 0 ]

F_KE1:
2/Q1*KEzz

F_KE2:
2/Q2*(KExx+KEyy)

H(Q_2)*H^-1(Q_0):
[Q2_2/Q2_0 0 0 ]
[0 Q2_2/Q2_0 0 ]
[0 0 Q1_2/Q1_0 ]

H(Q_2)*H(Q_1)^-2*H(Q_0):
[Q2_2*Q2_0/Q2_1/Q2_1 0 0 ]
[0 Q2_2*Q2_0/Q2_1/Q2_1 0 ]
[0 0 Q1_2*Q1_0/Q1_1/Q1_1 ]

F_virial1:
1/Q1*Virialzz

F_virial2:
1/Q2*(Virialxx + Virialyy)

*/
class ZhangSpecial: public BoxPistonBase{
  public:
    double pressure;/**< Reference normal pressure, dimensions of energy/volume, it is SI compatible with other units.*/
    double surface_tension;/**< reference surface tension, dimensions of energy/area. */
    double zpiston_mass;/**< Piston Mass, dimensions of energy*time^2/length^2. */
    double xpiston_mass;/**< Piston Mass, dimensions of energy*time^2/length^2. */
    double zgamma;/**< Langevin gamma, dimensions of 1/time, 0 is no thermostat, inf is resampling */
    double xgamma;/**< Langevin gamma, dimensions of 1/time, 0 is no thermostat, inf is resampling */
    double k;/**< box.yy/box.xx, unitless.*/
    /** Return the potential energy of the box. */

    inline CUDA_CALLABLE_MEMBER double pe(double3 box) const{
        return box.x*box.y*box.z*pressure - surface_tension*box.x*box.y;
    }
    /** Return the kinetic energy of the box. */
    inline CUDA_CALLABLE_MEMBER double ke(double3 box,double3 box_dot) const{
        return 0.5*xpiston_mass*box_dot.x*box_dot.x+0.5*zpiston_mass*box_dot.z*box_dot.z;
    }
    /** Drift the box, and calculate the center of mass of the scaling terms. */
    inline CUDA_CALLABLE_MEMBER void drift(
            //inputs
            const double3& old_com_ke,/**< kinetic energy of pressure group momentums*/
            const double3& old_box,/**< box lengths*/
            const double3& old_box_dot,/**< box length time dirivatives*/
            //outputs
            double3& new_com_ke,/**< post drift pressure group momentum kinetic energy*/
            double3& new_box,/**< post drift box lengths*/
            double3& new_box_dot,/**< post drift box length time dirivatives*/
            double3& v_scale,/**< final pressure group momentum scale.
                This is the total scale from predrift p group momentum to post drift p group momentum.*/
            double3& v_prescale,/**< initial pressure group momentum scale.
                This is the scaleing of momentum that happens before the pressure group centers of mass are moved.*/
            double3& r_prescale,/**< initial pressure group COM scale.
                This is the scaling p group centers of mass before the are moved by their prescaled momenentums.*/
            //parameters
            double timestep/**< Time step.*/
          ) const{
        double3 box= old_box;
        double3 box_dot= old_box_dot;
        double3 com_ke= old_com_ke;
        double hx0= box.x;
        double hz0= box.z;
        double hx_dot= box_dot.x;
        double hz_dot= box_dot.z;
        // step 1. apply the 1/2 F_KE and F_pres
        double f_kex;
        double f_presx;
        double f_kez;
        double f_presz;
        f_kez= 2.0/hz0*com_ke.z;
        f_kex= 2.0/hx0*(com_ke.x+com_ke.y);
        f_presz= -k*hx0*hx0*pressure;
        f_presx= -2*k*hx0*hz0*pressure+2*surface_tension*k*hx0;
        hz_dot+= (f_kez+f_presz)*0.5*timestep/zpiston_mass;
        hx_dot+= (f_kex+f_presx)*0.5*timestep/xpiston_mass;
        // actual integration of piston
        // step 2. volume on the half step
        double hz1= hz0 + hz_dot*timestep*0.5;
        double hx1= hx0 + hx_dot*timestep*0.5;
        // step 4. volume on the full step
        double hz2= hz0 + hz_dot*timestep;
        double hx2= hx0 + hx_dot*timestep;
        double momentum_scalex= hx0/hx2;
        double momentum_scalez= hz0/hz2;
        com_ke.x*= momentum_scalex*momentum_scalex;
        com_ke.y*= momentum_scalex*momentum_scalex;
        com_ke.z*= momentum_scalez*momentum_scalez;
        // step 5. apply the 1/2 F_KE and F_pres
        f_kez= 2.0/hz2*com_ke.z;
        f_kex= 2.0/hx2*(com_ke.x+com_ke.y);
        f_presz= -k*hx2*hx2*pressure;
        f_presx= -2*k*hx2*hz2*pressure+2*surface_tension*k*hx2;
        hz_dot+= (f_kez+f_presz)*0.5*timestep/zpiston_mass;
        hx_dot+= (f_kex+f_presx)*0.5*timestep/xpiston_mass;

        //write outputs
        new_com_ke= com_ke;
        box.x= hx2;
        box.y= k*hx2;
        box.z= hz2;
        new_box= box;
        new_box_dot.x= hx_dot;
        new_box_dot.y= k*hx_dot;
        new_box_dot.z= hz_dot;
        v_scale.x=momentum_scalex;
        v_scale.y=momentum_scalex;
        v_scale.z=momentum_scalez;
        double momentum_prescalex= hx2*hx0/(hx1*hx1);
        double momentum_prescalez= hz2*hz0/(hz1*hz1);
        v_prescale.x=momentum_prescalex;
        v_prescale.y=momentum_prescalex;
        v_prescale.z=momentum_prescalez;
        double position_scalex= hx2/hx0;
        double position_scalez= hz2/hz0;
        r_prescale.x=position_scalex;
        r_prescale.y=position_scalex;
        r_prescale.z=position_scalez;
    }
    /** Virial kick. */
    inline CUDA_CALLABLE_MEMBER void virialKick(
            //inputs
            const double3& virial,/**< corrected virial*/
            const double3& box,/**< box lengths*/
            const double3& old_box_dot,/**< box length time dirivatives*/
            //outputs
            double3& new_box_dot,/**< save post drift box length time dirivatives*/
            //parameters
            double timestep/**< Time step.*/
          ) const{
        double3 box_dot= old_box_dot;
        double hx= box.x;
        double hz= box.z;
        double hx_dot= box_dot.x;
        double hz_dot= box_dot.z;
        //actual integration of piston
        hx_dot += -1/hx*(virial.x+virial.y)/xpiston_mass*timestep;
        hz_dot += -1/hz*(virial.z)/zpiston_mass*timestep;

        //write outputs
        new_box_dot.x= hx_dot;
        new_box_dot.y= k*hx_dot;
        new_box_dot.z= hz_dot;
    }
    /** Thermostat kick. Uses up to 256 rng counter values, counter_lsw to counter_lsw + 255 inclusive. */
    virtual inline CUDA_CALLABLE_MEMBER void thermostat(
            //inputs
            const double3& box,/**< box lengths*/
            const double3& old_box_dot,/**< box length time dirivatives*/
            //outputs
            double3& new_box_dot,/**< post drift box length time dirivatives*/
            //parameters
            double temperature,/**< kT, dimensions of energy. */
            double timestep,/**< Time step.*/
            uint64_t key,/**< philox random CBRNG 4x32_10 key.*/
            uint64_t counter_msw,/**< philox random CBRNG 4x32_10 counter most significant word.*/
            uint64_t counter_lsw /**< philox random CBRNG 4x32_10 counter least significant word.*/
          ) const{
        double3 box_dot= old_box_dot;
        double hx_dot= box_dot.x;
        double hz_dot= box_dot.z;
        //actual integration of piston
        double2 randn2= randnormal(key,counter_msw,counter_lsw);
        //actual integration of piston
        hx_dot = singleLangevin(hx_dot,temperature,xgamma,xpiston_mass,timestep,randn2.x);
        hz_dot = singleLangevin(hz_dot,temperature,zgamma,zpiston_mass,timestep,randn2.y);
        //write outputs
        new_box_dot.x= hx_dot;
        new_box_dot.y= k*hx_dot;
        new_box_dot.z= hz_dot;
    }
};
}
