/** \file
 * \author Nathan Zimmerberg (nhz2@cornell.edu)
 * \date 13 Jan 2020
 * \brief Class to hold constant pressure AndersenBarostat parameters and helper functions
 */
#pragma once
#include "BoxPistonBase.h"

namespace boxpiston
{
/**Andersen Barostat.
parameters:
  pressure(double): reference pressure
  k(double3): reference box lengths

Q: Volume

PE_box(Q): Q*pressure

H(Q):
[k.x*cbrt(Q) 0 0 ]
[0 k.y*cbrt(Q) 0 ]
[0 0 k.z*cbrt(Q) ]

F_pres:
-pressure

Q(H):
Hxx*Hyy*Hzz/(k.x*k.y*k.z)

dH(Q)/d(Q):
[k.x*rcbrt(Q*Q)/3 0 0 ]
[0 k.y*rcbrt(Q*Q)/3 0 ]
[0 0 k.z*rcbrt(Q*Q)/3 ]

Hdot:
[k.x*rcbrt(Q*Q)/3*Qdot 0 0 ]
[0 k.y*rcbrt(Q*Q)/3*Qdot 0 ]
[0 0 k.z*rcbrt(Q*Q)/3*Qdot ]

Qdot:
Hyy*Hzz/(k.x*k.y*k.z)*Hxxdot +
Hxx*Hzz/(k.x*k.y*k.z)*Hyydot +
Hxx*Hyy/(k.x*k.y*k.z)*Hzzdot

H^-1:
[1/k.x*rcbrt(Q) 0 0 ]
[0 1/k.y*rcbrt(Q) 0 ]
[0 0 1/k.z*rcbrt(Q) ]

H^-1*dH(Q)/d(Q):
[1/3/Q 0 0 ]
[0 1/3/Q 0 ]
[0 0 1/3/Q ]

F_KE:
2/3/Q*trace(KE)

H(Q_2)*H^-1(Q_0):
[cbrt(Q_2/Q_0) 0 0 ]
[0 cbrt(Q_2/Q_0) 0 ]
[0 0 cbrt(Q_2/Q_0) ]

H(Q_2)*H(Q_1)^-2*H(Q_0):
[cbrt(Q_2*Q_0/Q_1/Q_1) 0 0 ]
[0 cbrt(Q_2*Q_0/Q_1/Q_1) 0 ]
[0 0 cbrt(Q_2*Q_0/Q_1/Q_1) ]

F_virial:
1/3/Q*trace(virial)


*/
class AndersenBarostat: public BoxPistonBase{
  public:
    double pressure;/**< Reference pressure, dimensions of energy/volume, it is SI compatible with other units.*/
    double piston_mass;/**< Piston Mass, dimensions of energy*time^2/volume^2.*/
    double gamma;/**< Langevin gamma, dimensions of 1/time, 0 is no thermostat. */
    double3 k;/**< Reference box lengths, at volume one, unitless.*/
    /** Return the potential energy of the box. */

    inline CUDA_CALLABLE_MEMBER double pe(double3 box) const{
        return (box.x*box.y*box.z)/(k.x*k.y*k.z)*pressure;
    }
    /** Return the kinetic energy of the box. */
    inline CUDA_CALLABLE_MEMBER double ke(double3 box,double3 box_dot) const{
        double volume_dot= box_dot.x*box.y*box.z+box.x*box_dot.y*box.z+box.x*box.y*box_dot.z;
        volume_dot= volume_dot/(k.x*k.y*k.z);
        return 0.5*piston_mass*volume_dot*volume_dot;
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
        double ke= com_ke.x+com_ke.y+com_ke.z;
        double volume0= (box.x*box.y*box.z)/(k.x*k.y*k.z);
        double volume_dot= box_dot.x*box.y*box.z+box.x*box_dot.y*box.z+box.x*box.y*box_dot.z;
        volume_dot= volume_dot/(k.x*k.y*k.z);
        // step 1. apply the 1/2 F_KE and F_pres
        double f_ke;
        double f_pres;
        f_ke= 2.0/3.0/volume0*ke;
        f_pres= -pressure;
        volume_dot+= (f_ke+f_pres)*0.5*timestep/piston_mass;
        // actual integration of piston
        // step 2. volume on the half step
        double volume1= volume0 + volume_dot*timestep*0.5;
        // step 4. volume on the full step
        double volume2= volume0 + volume_dot*timestep;
        double momentum_scale= cbrt(volume0/volume2);
        double ke_scale=momentum_scale*momentum_scale;
        ke*= ke_scale;
        // step 5. apply the 1/2 F_KE and F_pres
        f_ke= 2.0/3.0/volume2*ke;
        f_pres= -pressure;
        volume_dot+= (f_ke+f_pres)*0.5*timestep/piston_mass;

        //write outputs
        double position_scale= cbrt(volume2/volume0);
        new_com_ke.x= com_ke.x*ke_scale;
        new_com_ke.y= com_ke.y*ke_scale;
        new_com_ke.z= com_ke.z*ke_scale;
        box.x= k.x*cbrt(volume2);
        box.y= k.y*cbrt(volume2);
        box.z= k.z*cbrt(volume2);
        new_box= box;
        new_box_dot.x= k.x*rcbrt(volume2*volume2)/3*volume_dot;
        new_box_dot.y= k.y*rcbrt(volume2*volume2)/3*volume_dot;
        new_box_dot.z= k.z*rcbrt(volume2*volume2)/3*volume_dot;
        v_scale.x=momentum_scale;
        v_scale.y=momentum_scale;
        v_scale.z=momentum_scale;
        double momentum_prescale= cbrt(volume2*volume0/volume1/volume1);
        v_prescale.x=momentum_prescale;
        v_prescale.y=momentum_prescale;
        v_prescale.z=momentum_prescale;
        r_prescale.x=position_scale;
        r_prescale.y=position_scale;
        r_prescale.z=position_scale;
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
        double volume= (box.x*box.y*box.z)/(k.x*k.y*k.z);
        double volume_dot= box_dot.x*box.y*box.z+box.x*box_dot.y*box.z+box.x*box.y*box_dot.z;
        //actual integration of piston
        volume_dot += (-virial.x-virial.y-virial.z)/volume/3.0/piston_mass*timestep;
        //write outputs
        new_box_dot.x= k.x*rcbrt(volume*volume)/3*volume_dot;
        new_box_dot.y= k.y*rcbrt(volume*volume)/3*volume_dot;
        new_box_dot.z= k.z*rcbrt(volume*volume)/3*volume_dot;
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
        double volume= (box.x*box.y*box.z)/(k.x*k.y*k.z);
        double volume_dot= box_dot.x*box.y*box.z+box.x*box_dot.y*box.z+box.x*box.y*box_dot.z;
        double2 randn2= randnormal(key,counter_msw,counter_lsw);
        //actual integration of piston
        volume_dot = singleLangevin(volume_dot,temperature,gamma,piston_mass,timestep,randn2.x);
        //write outputs
        new_box_dot.x= k.x*rcbrt(volume*volume)/3*volume_dot;
        new_box_dot.y= k.y*rcbrt(volume*volume)/3*volume_dot;
        new_box_dot.z= k.z*rcbrt(volume*volume)/3*volume_dot;
          }
};
}
