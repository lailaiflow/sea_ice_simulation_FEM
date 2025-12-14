/*
 * InputData.h
 *
 *  Created on: Jan 26, 2017
 *      Author: lailai
 */

#ifndef SOURCE_INPUTDATA_H_
#define SOURCE_INPUTDATA_H_

#include <string>


namespace EquationData
  {
    const double eta = 1;
    const double kappa = 1e-6;
    const double density = 1;
    const double sec_per_day = 24*3600;
    double dt;
    double dt_dim;
    int NT;
    const int    Nsub = 20;
    const double Tsub = 0.2; // between 0 to 1
    double Re;
    double time;


    int          NtExp;
    bool         subcycle_stokes_sig = false;
    bool         pseudo_stokes = true;
    bool         ifstr_stokes; // whether remove the time-derivative term absolutely.

    double       dte = dt/Nsub;
    double       dte_inv = 1./dte;
    double       chi = Tsub*dt;
  }


namespace UserGeometry  // currently only for 2d simulations.
  {
  	const double x_inlet = -5;
  	const double x_outlet = 5;
  	const double y_bottom = -0.5;
  	const double y_top = 0.5;
  }


  namespace CarreauModel
  {
  	const double mu_zero = 1;
  	const double mu_infty = 0;
  	const double lambda = 10;
  	const double n = 0.25;
  }

  namespace IceModel
  {
  const double Estar_dim = 2e-9;
  //const double f = 0.2;
  const double Cw = 5.5e-3;
  const double Ca = 1.2e-3;
//  const double zeta_min = 4e8;
  const double zeta_max = 1e12; //used for granular model
  const double rho_water = 1e3;
  const double rho_air = 1.3;
  const double S = 1.35e4;
  double zeta_max_non; //the nondimensional value of zeta_max scaled by zeta_min

  //parameter controlling frazil ice growth rate
  const double a=1.95E-4;
  const double b=-1.607;
  const double p1=-2.311;
  const double p2=0.131;
  const double lim=0.04; // meter
  //parameter controlling frazil ice growth rate

  const double sinp_gra = 0.5; // sin(30 deg), constant for granular model

      double zeta_min;
      double f;
      double alpha;
      double k;
      double c0;
      double Estar;
      double h0;
      double hinput;//dimensional value of input thickness
      double tol_pseudo;
      bool   ifwdrag;
      bool   ifperi_wd;
      bool   ifthermal;
      bool   ifthermal_cmax;
      bool   ifcorio;
      double beta;
      double hC;//characteristic ice thickness;
      double wC;//characteristic size
      double tC;//characteristic time
      double uC;//characteristic ice velocity
      double hd;//dimensional demarcation thickness
      double hd_non;//dimensionless demarcation thickness
      double Tperi_non; // nondimensional value of the period
      double Tperi;//dimensional value of the period, in days
      double Vf;//dimensonal value of freezing rate, velocity scale, meter per day, typical value of 0.27 m per day, as suggested by
      //'A Two-Dimensional Time-Dependent Model of a Wind-Driven Coastal Polynya: Application to the St. Lawrence Island Polynya '
      double Sigma;//nondimensional value of Vf
      double fco;//dimensional Coriolis parameter, 1/second
      double fco_non;//nondimensional Coriolis parameter

      double uwind; // meter per second

      double theta; //angle (degree) of the wind with respect to the x axis.

      double size_kata; // characteristic length of the katabatic wind; in meter
      double size_kata_non; // nondimensional value of size_kata

      bool   ifsts_wd; // whether input directly the wind stress magnitude or the velocity of wind
      bool   iffix_thick;
      bool   iffix_conc;
      bool   ifbound_thick;
      bool   ifbound_conc;
      bool   ifkata; // if apply the katabatic wind;
      bool   ifgra; // whether use granular rheology model

  }

  namespace SolverConfig
  {
  	  std::string stokes;
  }
#endif /* SOURCE_INPUTDATA_H_ */
