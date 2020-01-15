#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include <cuda_runtime.h>
#include <cuda_utils.h>
#include <Box-Pistons-Orthogonal/ZhangSpecial.h>

template<typename T, int N>
struct H_D{
    T* h;
    T* d;
    H_D(){
        h= (T*)malloc(N*sizeof(T));
        cudaMalloc((void**)&d,N*sizeof(T));
    }
    void c2d(){
        cudaCheck(cudaMemcpy(d,h,N*sizeof(T),cudaMemcpyHostToDevice));
    }
    void c2h(){
        cudaCheck(cudaMemcpy(h,d,N*sizeof(T),cudaMemcpyDeviceToHost));
    }
};


TEST_CASE("Constant pressure piston drift"){
    H_D<double3,1> virial;
    H_D<double3,1> comke;
    H_D<double3,1> box;
    H_D<double3,1> box_dot;
    H_D<double3,1> v_prescale;
    H_D<double3,1> v_scale;
    H_D<double3,1> r_prescale;
    //initialize everything to zero
    double3 zero3;
    zero3.x=0;
    zero3.y=0;
    zero3.z=0;
    double3 one3;
    one3.x=1;
    one3.y=1;
    one3.z=1;
    boxpiston::ZhangSpecial piston;
    piston.xgamma= 0;
    piston.zgamma= 0;
    piston.xpiston_mass= 10;
    piston.zpiston_mass= 10;
    piston.pressure=1;
    piston.surface_tension=0;
    piston.k= 1;
    *virial.h= zero3;
    *comke.h= one3;
    *box.h= one3;
    *box_dot.h= zero3;
    double timestep= 0.01;
    SECTION("NPH ideal gas simulation"){
        double initial_total_energy= piston.pe(*box.h)+piston.ke(*box.h,*box_dot.h)+
                            comke.h->x+comke.h->y+comke.h->z;
        for (int i=0;i<1000000;i++){
            piston.virialKick(*virial.h,*box.h,*box_dot.h,*box_dot.h,timestep*0.5);
            piston.drift(*comke.h,*box.h,*box_dot.h,*comke.h,*box.h,*box_dot.h,
              *v_scale.h,*v_prescale.h,*r_prescale.h,timestep*0.5);
            piston.thermostat(*box.h,*box_dot.h,*box_dot.h,85,timestep,0,0,i*256);
            piston.drift(*comke.h,*box.h,*box_dot.h,*comke.h,*box.h,*box_dot.h,
              *v_scale.h,*v_prescale.h,*r_prescale.h,timestep*0.5);
            piston.virialKick(*virial.h,*box.h,*box_dot.h,*box_dot.h,timestep*0.5);
            double final_total_energy= piston.pe(*box.h)+piston.ke(*box.h,*box_dot.h)+
                                comke.h->x+comke.h->y+comke.h->z;
            REQUIRE(final_total_energy==Approx(initial_total_energy).epsilon(1E-4));
        }
        piston.pressure=5;
        piston.surface_tension=1;
        initial_total_energy= piston.pe(*box.h)+piston.ke(*box.h,*box_dot.h)+
                            comke.h->x+comke.h->y+comke.h->z;
        for (int i=0;i<1000000;i++){
          piston.virialKick(*virial.h,*box.h,*box_dot.h,*box_dot.h,timestep*0.5);
          piston.drift(*comke.h,*box.h,*box_dot.h,*comke.h,*box.h,*box_dot.h,
            *v_scale.h,*v_prescale.h,*r_prescale.h,timestep*0.5);
          piston.thermostat(*box.h,*box_dot.h,*box_dot.h,85,timestep,0,1,i*256);
          piston.drift(*comke.h,*box.h,*box_dot.h,*comke.h,*box.h,*box_dot.h,
            *v_scale.h,*v_prescale.h,*r_prescale.h,timestep*0.5);
          piston.virialKick(*virial.h,*box.h,*box_dot.h,*box_dot.h,timestep*0.5);
          double final_total_energy= piston.pe(*box.h)+piston.ke(*box.h,*box_dot.h)+
                              comke.h->x+comke.h->y+comke.h->z;
          REQUIRE(final_total_energy==Approx(initial_total_energy).epsilon(1E-4));
        }
    }
}
