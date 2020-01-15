/** \file
 * \author Nathan Zimmerberg (nhz2@cornell.edu)
 * \date 10 Jan 2020
 * \brief Base class to hold box pistons
 */
#pragma once
#include <stdint.h>
#include <random_utils.h>
#include <cuda_utils.h>

namespace boxpiston
{
/**Base class and interface for all box pistons.
All derived classes should have H(Qs) and PE_box(Qs),
where H is the symetric box matrix, and Qs are the box degrees of freedom,
and PE_box is the potential energy of the box.
Each Q also has a piston mass, and gamma.
All units are SI compatible.

Definition
Q:
PE_box(Q):

Intermediate math expansions to help write code.
F_pres:
H(Q):
Q(H):
dH(Q)/d(Q):
H^-1:
H^-1*dH(Q)/d(Q):
F_KE:
*/
class BoxPistonBase{
  public:
    /** Return the potential energy of the box. */
    virtual inline CUDA_CALLABLE_MEMBER double pe(double3 box) const = 0;
    /** Return the kinetic energy of the box. */
    virtual inline CUDA_CALLABLE_MEMBER double ke(double3 box,double3 box_dot) const = 0;
    /** Drift the box, and calculate the pressure group center of mass scaling terms. */
    virtual inline CUDA_CALLABLE_MEMBER void drift(
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
          ) const = 0;
    /** Virial kick. */
    virtual inline CUDA_CALLABLE_MEMBER void virialKick(
            //inputs
            const double3& virial,/**< corrected virial*/
            const double3& box,/**< box lengths*/
            const double3& old_box_dot,/**< box length time dirivatives*/
            //outputs
            double3& new_box_dot,/**< save post drift box length time dirivatives*/
            //parameters
            double timestep/**< Time step.*/
          ) const = 0;
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
          ) const = 0;
};

/** Returns the new theromstated velocity */
inline CUDA_CALLABLE_MEMBER double singleLangevin(
      double velocity, /**< velocity of ? degree of freedom, dimensions of ?/time. */
      double temperature,/**< kT, dimensions of energy. */
      double gamma, /**< gamma, dimensions of 1/time. 0 is no thermostat, inf is resample velocity */
      double mass, /**< dimensions of energy*time^2/?^2. */
      double timestep,/**< Time step.*/
      double randn/**< Normally distributed number, zero mean, one variance. */
  ){
    velocity*= exp(-gamma*timestep);
    velocity+= sqrt(1.0-exp(-2.0*gamma*timestep))*sqrt(temperature/mass)*randn;
    return velocity;
}

}
