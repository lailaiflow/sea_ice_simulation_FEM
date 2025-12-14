#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/constraint_matrix.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>


#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/solution_transfer.h>

#include <deal.II/base/index_set.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/trilinos_block_vector.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>



#include <iostream>
#include <fstream>
#include <sstream>
#include <limits>
#include <string>
#include <map>
#include <locale>
#include <sys/stat.h>

#include <deal.II/distributed/solution_transfer.h>

#include "InputData.h"

#define PI 3.14159265

namespace Solver_
  {
	using namespace dealii;

    TrilinosWrappers::SolverDirect::AdditionalData data_klu ("Amesos_Klu");
    SolverControl           solver_control_klu (1000, 1e-10);
    TrilinosWrappers::SolverDirect solver_klu(solver_control_klu, data_klu);

  }

namespace IceRheo
{
  using namespace dealii;

  namespace Assembly
  {
  	namespace Scratch
    {
  		template <int dim>
  		struct StokesSystem
		{
  			StokesSystem (const FiniteElement<dim> &stokes_fe,
   				          const Mapping<dim>       &mapping,
  					      const Quadrature<dim>    &stokes_quadrature,
					      const UpdateFlags         stokes_update_flags,
						  const FiniteElement<dim> &thick_fe,
						  const UpdateFlags         thick_update_flags,
						  const FiniteElement<dim> &conc_fe,
						  const UpdateFlags         conc_update_flags,
						  const FiniteElement<dim> &p_fe,
						  const UpdateFlags         p_update_flags);

  			StokesSystem (const StokesSystem &data);

  			FEValues<dim>               stokes_fe_values;
  		    FEValues<dim>               thick_fe_values;
  		    FEValues<dim>               conc_fe_values;
  		    FEValues<dim>               p_fe_values;

  		    std::vector<Vector<double>> rhs_values;
  		    std::vector<Tensor<1,dim> > old_velocity_values;
  		    std::vector<Tensor<1,dim> > old_velocity_values_wdrag;
  		    std::vector<Tensor<2,dim> > velocity_gradients;
  			std::vector<double>         value_thick,
			                            value_conc;
  			//std::vector<double>         old_pressure_values;
  		    std::vector<Tensor<1,dim> > old_pressure_gradients;



  		    std::vector<Tensor<1,dim> >          phi_u;
  		    std::vector<SymmetricTensor<2,dim> > grads_phi_u;
  		    std::vector<double>                  div_phi_u;

	    };

        template <int dim>
        StokesSystem<dim>::
        StokesSystem (const FiniteElement<dim> &stokes_fe,
			          const Mapping<dim>       &mapping,
			          const Quadrature<dim>    &stokes_quadrature,
			          const UpdateFlags         stokes_update_flags,
				      const FiniteElement<dim> &thick_fe,
				      const UpdateFlags         thick_update_flags,
					  const FiniteElement<dim> &conc_fe,
					  const UpdateFlags         conc_update_flags,
				      const FiniteElement<dim> &p_fe,
				      const UpdateFlags         p_update_flags)
          :

          stokes_fe_values (mapping, stokes_fe, stokes_quadrature,
                            stokes_update_flags),

	      thick_fe_values    (mapping, thick_fe, stokes_quadrature,
							  thick_update_flags),

		  conc_fe_values    (mapping, conc_fe, stokes_quadrature,
							 conc_update_flags),

		  p_fe_values    (mapping, p_fe, stokes_quadrature,
						  p_update_flags),

		  rhs_values(stokes_quadrature.size(),Vector<double>(dim)),
          old_velocity_values(stokes_quadrature.size()),
          old_velocity_values_wdrag(stokes_quadrature.size()),
          velocity_gradients(stokes_quadrature.size()),
		  value_thick(stokes_quadrature.size()),
		  value_conc(stokes_quadrature.size()),
		  //old_pressure_values(stokes_quadrature.size()),
	      old_pressure_gradients(stokes_quadrature.size()),

		  phi_u(stokes_fe.dofs_per_cell),
		  grads_phi_u(stokes_fe.dofs_per_cell),
		  div_phi_u(stokes_fe.dofs_per_cell)
        {}



        template <int dim>
        StokesSystem<dim>::
        StokesSystem (const StokesSystem &scratch)
          :
          stokes_fe_values (scratch.stokes_fe_values.get_mapping(),
                            scratch.stokes_fe_values.get_fe(),
                            scratch.stokes_fe_values.get_quadrature(),
                            scratch.stokes_fe_values.get_update_flags()),

	      thick_fe_values  (scratch.thick_fe_values.get_mapping(),
	  		                scratch.thick_fe_values.get_fe(),
	  		                scratch.thick_fe_values.get_quadrature(),
	  		                scratch.thick_fe_values.get_update_flags()),

		  conc_fe_values  (scratch.conc_fe_values.get_mapping(),
						   scratch.conc_fe_values.get_fe(),
						   scratch.conc_fe_values.get_quadrature(),
						   scratch.conc_fe_values.get_update_flags()),

	       p_fe_values    (scratch.p_fe_values.get_mapping(),
			               scratch.p_fe_values.get_fe(),
			               scratch.p_fe_values.get_quadrature(),
			               scratch.p_fe_values.get_update_flags()),

		  rhs_values(scratch.rhs_values),
		  old_velocity_values(scratch.old_velocity_values),
		  old_velocity_values_wdrag(scratch.old_velocity_values_wdrag),
		  velocity_gradients(scratch.velocity_gradients),
		  value_thick(scratch.value_thick),
		  value_conc(scratch.value_conc),
		  //old_pressure_values(scratch.old_pressure_values),
		  old_pressure_gradients(scratch.old_pressure_gradients),

		  phi_u(scratch.phi_u),
		  grads_phi_u(scratch.grads_phi_u),
		  div_phi_u(scratch.div_phi_u)
        {}




        template <int dim>
        struct PSystem
        {
          PSystem (const FiniteElement<dim> &p_fe,
                   const Mapping<dim>       &mapping,
                   const Quadrature<dim>    &p_quadrature,
    			   const UpdateFlags         p_update_flags,
				   const FiniteElement<dim> &stokes_fe,
				   const UpdateFlags         stokes_update_flags,
				   const FiniteElement<dim> &thick_fe,
				   const UpdateFlags         thick_update_flags,
				   const FiniteElement<dim> &conc_fe,
				   const UpdateFlags         conc_update_flags);


          PSystem (const PSystem &data);


          FEValues<dim>               p_fe_values;
          FEValues<dim>               stokes_fe_values;
          FEValues<dim>               thick_fe_values;
          FEValues<dim>               conc_fe_values;

		  std::vector<double>         value_thick,
				                      value_conc;
     	  std::vector<Tensor<2,dim> > old_velocity_gradients;
        };


        template <int dim>
        PSystem<dim>::
        PSystem (const FiniteElement<dim> &p_fe,
                 const Mapping<dim>       &mapping,
                 const Quadrature<dim>    &p_quadrature,
 			     const UpdateFlags         p_update_flags,
				 const FiniteElement<dim> &stokes_fe,
				 const UpdateFlags         stokes_update_flags,
				 const FiniteElement<dim> &thick_fe,
				 const UpdateFlags         thick_update_flags,
				 const FiniteElement<dim> &conc_fe,
				 const UpdateFlags         conc_update_flags)
          :
          p_fe_values (mapping,
        		       p_fe,
                       p_quadrature,
    				   p_update_flags),

		  stokes_fe_values (mapping,
				            stokes_fe,
							p_quadrature,
							stokes_update_flags),

		  thick_fe_values (mapping,
						   thick_fe,
						   p_quadrature,
						   thick_update_flags),

		  conc_fe_values (mapping,
						  conc_fe,
						  p_quadrature,
						  conc_update_flags),

		  value_thick (p_quadrature.size()),
		  value_conc  (p_quadrature.size()),
		  old_velocity_gradients (p_quadrature.size())
        {}

        template <int dim>
        PSystem<dim>::
        PSystem (const PSystem &scratch)
          :
          p_fe_values (scratch.p_fe_values.get_mapping(),
                       scratch.p_fe_values.get_fe(),
                       scratch.p_fe_values.get_quadrature(),
                       scratch.p_fe_values.get_update_flags()),

          stokes_fe_values (scratch.stokes_fe_values.get_mapping(),
                            scratch.stokes_fe_values.get_fe(),
                            scratch.stokes_fe_values.get_quadrature(),
                            scratch.stokes_fe_values.get_update_flags()),

	  thick_fe_values    (scratch.thick_fe_values.get_mapping(),
	  		      scratch.thick_fe_values.get_fe(),
	  		      scratch.thick_fe_values.get_quadrature(),
	  		      scratch.thick_fe_values.get_update_flags()),

	  conc_fe_values     (scratch.conc_fe_values.get_mapping(),
	  		      scratch.conc_fe_values.get_fe(),
	  		      scratch.conc_fe_values.get_quadrature(),
	  		      scratch.conc_fe_values.get_update_flags()),
	  
	  value_thick (scratch.value_thick),
	  value_conc  (scratch.value_conc),
	  old_velocity_gradients (scratch.old_velocity_gradients)

        {}

        template <int dim>
        struct SigSystem
        {
          SigSystem (const FiniteElement<dim> &sig_fe,
                     const Mapping<dim>       &mapping,
                     const Quadrature<dim>    &sig_quadrature,
		     const UpdateFlags         sig_update_flags,
		     const FiniteElement<dim> &stokes_fe,
                     const UpdateFlags         stokes_update_flags,
		     const FiniteElement<dim> &p_fe,
                     const UpdateFlags         p_update_flags);

          SigSystem (const SigSystem &data);


          FEValues<dim>               sig_fe_values;
          FEValues<dim>               stokes_fe_values;
          FEValues<dim>               p_fe_values;


    	  std::vector<Tensor<2,dim> > grads_velocity;
    	  std::vector<double>         value_pressure;
    	  std::vector<double>         old_sig11_value,
    			              old_sig22_value;

    	  std::vector<double>         phi_sig11,
	                              phi_sig22,
	                              phi_sig_diag_sum;

        };

        template <int dim>
        SigSystem<dim>::
        SigSystem (const FiniteElement<dim> &sig_fe,
                   const Mapping<dim>       &mapping,
                   const Quadrature<dim>    &sig_quadrature,
		   const UpdateFlags         sig_update_flags,
		   const FiniteElement<dim> &stokes_fe,
                   const UpdateFlags         stokes_update_flags,
		   const FiniteElement<dim> &p_fe,
                   const UpdateFlags         p_update_flags)
          :
          sig_fe_values (mapping,
                         sig_fe, sig_quadrature,
                         update_values    | update_gradients |
                         update_JxW_values),

          stokes_fe_values (mapping,
        		            stokes_fe, sig_quadrature,
							update_gradients),

	      p_fe_values (mapping,
					   p_fe, sig_quadrature,
					   update_values),

		  grads_velocity  (sig_quadrature.size()),
		  value_pressure  (sig_quadrature.size()),
		  old_sig11_value (sig_quadrature.size()),
		  old_sig22_value (sig_quadrature.size()),

		  phi_sig11       (sig_fe.dofs_per_cell),
		  phi_sig22       (sig_fe.dofs_per_cell),
		  phi_sig_diag_sum(sig_fe.dofs_per_cell)
        {}

        template <int dim>
        SigSystem<dim>::
        SigSystem (const SigSystem &scratch)
          :
          sig_fe_values (scratch.sig_fe_values.get_mapping(),
                         scratch.sig_fe_values.get_fe(),
                         scratch.sig_fe_values.get_quadrature(),
                         scratch.sig_fe_values.get_update_flags()),


	      stokes_fe_values   (scratch.stokes_fe_values.get_mapping(),
	  		      scratch.stokes_fe_values.get_fe(),
	  		      scratch.stokes_fe_values.get_quadrature(),
	  		      scratch.stokes_fe_values.get_update_flags()),


	      p_fe_values   (scratch.p_fe_values.get_mapping(),
	  		 scratch.p_fe_values.get_fe(),
	  	         scratch.p_fe_values.get_quadrature(),
	         	 scratch.p_fe_values.get_update_flags()),
	  
		  grads_velocity (scratch.grads_velocity),
		  value_pressure (scratch.value_pressure),
		  old_sig11_value (scratch.old_sig11_value),
		  old_sig22_value (scratch.old_sig22_value),
		  phi_sig11 (scratch.phi_sig11),
		  phi_sig22 (scratch.phi_sig22),
		  phi_sig_diag_sum (scratch.phi_sig_diag_sum)
        {}


        template <int dim>
        struct SigDiagSystem
        {
          SigDiagSystem (const FiniteElement<dim> &sig_diag_fe,
                     const Mapping<dim>       &mapping,
                     const Quadrature<dim>    &sig_diag_quadrature,
					 const UpdateFlags         sig_diag_update_flags,
					 const FiniteElement<dim> &stokes_fe,
                     const UpdateFlags         stokes_update_flags,
					 const FiniteElement<dim> &p_fe,
                     const UpdateFlags         p_update_flags);

          SigDiagSystem (const SigDiagSystem &data);


          FEValues<dim>               sig_diag_fe_values;
          FEValues<dim>               stokes_fe_values;
          FEValues<dim>               p_fe_values;

    	  std::vector<Tensor<2,dim> > grads_velocity;
    	  std::vector<double>         value_pressure;
    	  std::vector<double>         old_sig_diag_value;
        };

        template <int dim>
        SigDiagSystem<dim>::
        SigDiagSystem (const FiniteElement<dim> &sig_diag_fe,
                       const Mapping<dim>       &mapping,
                       const Quadrature<dim>    &sig_diag_quadrature,
			           const UpdateFlags         sig_diag_update_flags,
				       const FiniteElement<dim> &stokes_fe,
                       const UpdateFlags         stokes_update_flags,
				       const FiniteElement<dim> &p_fe,
                       const UpdateFlags         p_update_flags)
          :
          sig_diag_fe_values (mapping,
                         sig_diag_fe, sig_diag_quadrature,
                         update_values    |
                         update_JxW_values),

          stokes_fe_values (mapping,
        		            stokes_fe, sig_diag_quadrature,
							update_gradients),

	      p_fe_values (mapping,
					   p_fe, sig_diag_quadrature,
					   update_values),

		  grads_velocity  (sig_diag_quadrature.size()),
		  value_pressure  (sig_diag_quadrature.size()),
		  old_sig_diag_value (sig_diag_quadrature.size())
        {}

        template <int dim>
        SigDiagSystem<dim>::
        SigDiagSystem (const SigDiagSystem &scratch)
          :
          sig_diag_fe_values (scratch.sig_diag_fe_values.get_mapping(),
                              scratch.sig_diag_fe_values.get_fe(),
                              scratch.sig_diag_fe_values.get_quadrature(),
                              scratch.sig_diag_fe_values.get_update_flags()),
	  
	  stokes_fe_values   (scratch.stokes_fe_values.get_mapping(),
	  		      scratch.stokes_fe_values.get_fe(),
	  		      scratch.stokes_fe_values.get_quadrature(),
	  		      scratch.stokes_fe_values.get_update_flags()),


	  p_fe_values   (scratch.p_fe_values.get_mapping(),
	  		 scratch.p_fe_values.get_fe(),
	  	         scratch.p_fe_values.get_quadrature(),
	         	 scratch.p_fe_values.get_update_flags()),
	  
	  grads_velocity (scratch.grads_velocity),
	  value_pressure (scratch.value_pressure),
	  old_sig_diag_value (scratch.old_sig_diag_value)
        {}

        template <int dim>
        struct ThickConcSystem
        {
          ThickConcSystem (const FiniteElement<dim> &thick_fe,
                           const Mapping<dim>       &mapping,
                           const Quadrature<dim>    &thick_quadrature,
			               const UpdateFlags         thick_update_flags,
                           const FiniteElement<dim>    &conc_fe,
			               const UpdateFlags         conc_update_flags,
			               const FiniteElement<dim> &stokes_fe,
			               const UpdateFlags         stokes_update_flags);


          ThickConcSystem (const ThickConcSystem &data);


          FEValues<dim>               thick_fe_values;
          FEValues<dim>               conc_fe_values;
          FEValues<dim>               stokes_fe_values;

    	  std::vector<Tensor<2,dim> > grads_velocity;
    	  std::vector<Tensor<1,dim> > velocity;
    	  std::vector<double>         old_thick_values;
    	  std::vector<double>         old_conc_values;
        };

        template <int dim>
        ThickConcSystem<dim>::
		ThickConcSystem (const FiniteElement<dim> &thick_fe,
                         const Mapping<dim>       &mapping,
                         const Quadrature<dim>    &thick_quadrature,
	                     const UpdateFlags         thick_update_flags,
                         const FiniteElement<dim>    &conc_fe,
	                     const UpdateFlags         conc_update_flags,
	                     const FiniteElement<dim> &stokes_fe,
	                     const UpdateFlags         stokes_update_flags)
	    :
		  thick_fe_values (mapping,
							                        thick_fe, thick_quadrature,
							                        update_values    | update_gradients |
							                        update_JxW_values),

								   conc_fe_values  (mapping,
											        conc_fe, thick_quadrature,
												    update_values),

								   stokes_fe_values (mapping,
									                 stokes_fe, thick_quadrature,
													 update_values    | update_gradients),

						           grads_velocity (thick_quadrature.size()),
							       velocity (thick_quadrature.size()),
								   old_thick_values (thick_quadrature.size()),
								   old_conc_values (thick_quadrature.size())
								   {}

        template <int dim>
        ThickConcSystem<dim>::
        ThickConcSystem (const ThickConcSystem &scratch)
          :
          thick_fe_values (scratch.thick_fe_values.get_mapping(),
                           scratch.thick_fe_values.get_fe(),
                           scratch.thick_fe_values.get_quadrature(),
                           scratch.thick_fe_values.get_update_flags()),
	  
          conc_fe_values  (scratch.conc_fe_values.get_mapping(),
                           scratch.conc_fe_values.get_fe(),
                           scratch.conc_fe_values.get_quadrature(),
                           scratch.conc_fe_values.get_update_flags()),

          stokes_fe_values (scratch.stokes_fe_values.get_mapping(),
                            scratch.stokes_fe_values.get_fe(),
                            scratch.stokes_fe_values.get_quadrature(),
                            scratch.stokes_fe_values.get_update_flags()),
	  
		  grads_velocity (scratch.grads_velocity),
		  velocity       (scratch.velocity),
		  old_thick_values (scratch.old_thick_values),
		  old_conc_values (scratch.old_conc_values)
        {}

    }

  	namespace CopyData
	{
  	template <int dim>
  	struct StokesSystem
	{
  		StokesSystem (const FiniteElement<dim> &stokes_fe);
  		StokesSystem (const StokesSystem &data);

  		FullMatrix<double>          local_matrix;
  		Vector<double>              local_rhs;
  		std::vector<types::global_dof_index>   local_dof_indices;
	};

  	template <int dim>
  	StokesSystem<dim>::
	StokesSystem (const FiniteElement<dim> &stokes_fe)
	:
	local_matrix (stokes_fe.dofs_per_cell,
			      stokes_fe.dofs_per_cell),
	local_rhs    (stokes_fe.dofs_per_cell),
	local_dof_indices (stokes_fe.dofs_per_cell)
	{}

  	template <int dim>
  	StokesSystem<dim>::
	StokesSystem (const StokesSystem &data)
	:
	local_matrix (data.local_matrix),
	local_rhs    (data.local_rhs),
	local_dof_indices (data.local_dof_indices)
	{}

  	template <int dim>
  	struct PSystem
	{
  		PSystem (const FiniteElement<dim> &p_fe);
  		PSystem (const PSystem &data);

  		FullMatrix<double>          local_matrix;
  		Vector<double>              local_rhs;
  		std::vector<types::global_dof_index>   local_dof_indices;
	};

  	template <int dim>
  	PSystem<dim>::
	PSystem (const FiniteElement<dim> &p_fe)
	:
	local_matrix (p_fe.dofs_per_cell,
			      p_fe.dofs_per_cell),
	local_rhs    (p_fe.dofs_per_cell),
	local_dof_indices (p_fe.dofs_per_cell)
	{}

  	template <int dim>
  	PSystem<dim>::
	PSystem (const PSystem &data)
	:
	local_matrix (data.local_matrix),
	local_rhs    (data.local_rhs),
	local_dof_indices (data.local_dof_indices)
	{}

  	template <int dim>
  	struct SigSystem
	{
  		SigSystem (const FiniteElement<dim> &sig_fe);
  		SigSystem (const SigSystem &data);

  		FullMatrix<double>          local_matrix;
  		Vector<double>              local_rhs;
  		std::vector<types::global_dof_index>   local_dof_indices;
	};

  	template <int dim>
  	SigSystem<dim>::
	SigSystem (const FiniteElement<dim> &sig_fe)
	:
	local_matrix (sig_fe.dofs_per_cell,
			      sig_fe.dofs_per_cell),
	local_rhs    (sig_fe.dofs_per_cell),
	local_dof_indices (sig_fe.dofs_per_cell)
	{}

  	template <int dim>
  	SigSystem<dim>::
	SigSystem (const SigSystem &data)
	:
	local_matrix (data.local_matrix),
	local_rhs    (data.local_rhs),
	local_dof_indices (data.local_dof_indices)
	{}

  	template <int dim>
  	struct SigDiagSystem
	{
  		SigDiagSystem (const FiniteElement<dim> &sig_diag_fe);
  		SigDiagSystem (const SigDiagSystem &data);

  		FullMatrix<double>          local_matrix;
  		Vector<double>              local_rhs;
  		std::vector<types::global_dof_index>   local_dof_indices;
	};

  	template <int dim>
  	SigDiagSystem<dim>::
	SigDiagSystem (const FiniteElement<dim> &sig_diag_fe)
	:
	local_matrix (sig_diag_fe.dofs_per_cell,
			      sig_diag_fe.dofs_per_cell),
	local_rhs    (sig_diag_fe.dofs_per_cell),
	local_dof_indices (sig_diag_fe.dofs_per_cell)
	{}

  	template <int dim>
  	SigDiagSystem<dim>::
	SigDiagSystem (const SigDiagSystem &data)
	:
	local_matrix (data.local_matrix),
	local_rhs    (data.local_rhs),
	local_dof_indices (data.local_dof_indices)
	{}

  	template <int dim>
  	struct ThickConcSystem
	{
  		ThickConcSystem (const FiniteElement<dim> &thick_fe);
  		ThickConcSystem (const ThickConcSystem &data);

  		FullMatrix<double>          local_matrix;
  		Vector<double>              local_rhs,
		                       conc_local_rhs;
  		std::vector<types::global_dof_index>   local_dof_indices;
	};

  	template <int dim>
  	ThickConcSystem<dim>::
	ThickConcSystem (const FiniteElement<dim> &thick_fe)
	:
	local_matrix (thick_fe.dofs_per_cell,
			      thick_fe.dofs_per_cell),
	local_rhs    (thick_fe.dofs_per_cell),
    conc_local_rhs    (thick_fe.dofs_per_cell),
    local_dof_indices (thick_fe.dofs_per_cell)
	{}

  	template <int dim>
  	ThickConcSystem<dim>::
	ThickConcSystem (const ThickConcSystem &data)
	:
	local_matrix (data.local_matrix),
	local_rhs    (data.local_rhs),
	conc_local_rhs    (data.conc_local_rhs),
	local_dof_indices (data.local_dof_indices)
	{}
	}


  }




  template <int dim>
  class PoiseuilleBoundaryValues : public Function<dim>
  {
  public:
    PoiseuilleBoundaryValues () : Function<dim>(dim) {}

    virtual double value (const Point<dim>   &p,
                          const unsigned int  component = 0) const;

    virtual void vector_value (const Point<dim> &p,
                               Vector<double>   &value) const;
  };


  template <int dim>
  double
  PoiseuilleBoundaryValues<dim>::value (const Point<dim>  &p,
                              const unsigned int component) const
  {
    Assert (component < this->n_components,
            ExcIndexRange (component, 0, this->n_components));

    double umax = 1.5;

    if (component == 0)
      return umax*(1-pow(2*p[1]/(UserGeometry::y_bottom-UserGeometry::y_top),2));
    return 0;
  }


  template <int dim>
  void
  PoiseuilleBoundaryValues<dim>::vector_value (const Point<dim> &p,
                                     Vector<double>   &values) const
  {
    for (unsigned int c=0; c<this->n_components; ++c)
      values(c) = PoiseuilleBoundaryValues<dim>::value (p, c);
  }


  template <int dim>
  class NoSlipBoundaryValues : public Function<dim>
  {
  public:
    NoSlipBoundaryValues () : Function<dim>(dim) {}

    virtual double value (const Point<dim>   &p,
                          const unsigned int  component = 0) const;

    virtual void vector_value (const Point<dim> &p,
                               Vector<double>   &value) const;
  };


  template <int dim>
  double
  NoSlipBoundaryValues<dim>::value (const Point<dim>  &p,
                              const unsigned int component) const
  {
    Assert (component < this->n_components,
            ExcIndexRange (component, 0, this->n_components));

    return 0;
  }

  template <int dim>
  void
  NoSlipBoundaryValues<dim>::vector_value (const Point<dim> &p,
                                     Vector<double>   &values) const
  {
    for (unsigned int c=0; c<this->n_components; ++c)
      values(c) = NoSlipBoundaryValues<dim>::value (p, c);
  }



  template <int dim>
  class InletThickBoundaryValues : public Function<dim>
  {
  public:
    InletThickBoundaryValues () : Function<dim>(1) {}

    virtual double value (const Point<dim>   &p,
                          const unsigned int  component = 0) const;

  };


  template <int dim>
  double
  InletThickBoundaryValues<dim>::value (const Point<dim>  &p,
                              const unsigned int component) const
  {
    Assert (component < this->n_components,
            ExcIndexRange (component, 0, this->n_components));

    return IceModel::h0;
  }

  template <int dim>
  class InletConcBoundaryValues : public Function<dim>
  {
  public:
    InletConcBoundaryValues () : Function<dim>(1) {}

    virtual double value (const Point<dim>   &p,
                          const unsigned int  component = 0) const;

  };


  template <int dim>
  double
  InletConcBoundaryValues<dim>::value (const Point<dim>  &p,
                              const unsigned int component) const
  {
    Assert (component < this->n_components,
            ExcIndexRange (component, 0, this->n_components));

    return IceModel::c0;
  }



  template <int dim>
  class NoVerticalBoundaryValues : public Function<dim>
  {
  public:
    NoVerticalBoundaryValues () : Function<dim>(dim) {}

    virtual double value (const Point<dim>   &p,
                          const unsigned int  component = 0) const;

  };


  template <int dim>
  double
  NoVerticalBoundaryValues<dim>::value (const Point<dim>  &p,
                              const unsigned int component) const
  {
    Assert (component < this->n_components,
            ExcIndexRange (component, 0, this->n_components));

    return 0;
  }


  template <int dim>
   class RightHandSide : public Function<dim>
   {
   public:
     RightHandSide () : Function<dim>(dim) {}
     virtual double value (const Point<dim>   &p,
                           const unsigned int  component = 0) const;
     virtual void vector_value (const Point<dim> &p,
                                Vector<double>   &value) const;
   };

   template <int dim>
   double
   RightHandSide<dim>::value (const Point<dim>  &p,
                              const unsigned int component) const
   {
	 double temp;
	 temp = IceModel::ifperi_wd ? cos(2*M_PI*EquationData::time/IceModel::Tperi_non) : 1.0;

	 if(IceModel::ifkata == true)
		 temp *= std::exp(-2*p[1]/IceModel::size_kata_non);

     if (component == 0)
       return temp*cos(IceModel::theta*M_PI/180);
     else if (component == 1)
       return temp*sin(IceModel::theta*M_PI/180);

     return 0;
   }

   template <int dim>
   void
   RightHandSide<dim>::vector_value (const Point<dim> &p,
                                     Vector<double>   &values) const
   {
     for (unsigned int c=0; c<this->n_components; ++c)
       values(c) = RightHandSide<dim>::value (p, c);
   }


  template <int dim>
  class SeaIceRheologyProblem
  {
  public:
    SeaIceRheologyProblem (const ParameterHandler &prm);
    void run ();
    static void declare_parameters (ParameterHandler &prm);
    void init_icemodel (const ParameterHandler &prm);
  private:
    void setup_dofs ();
    void assemble_stokes_system (const bool &ifcor);
    void assemble_stokes_system_parallel (const bool &ifcor);

    void assemble_p_system ();
    void assemble_p_system_parallel (const bool &ifcor);

    void assemble_sigma_system ();
    void assemble_sigma_system_parallel ();

    void assemble_sigma_diag_system ();
    void assemble_sigma_diag_system_parallel ();

    void assemble_thick_conc_system ();
    void assemble_thick_conc_system_parallel ();

    void solve_stokes();
    void solve_sigma ();
    void solve_thick_conc ();
    void solve_p ();

    void solve_stokes_parallel ();
    void solve_sigma_parallel ();
    void solve_thick_conc_parallel ();
    void solve_p_parallel ();


    void assign_old_stokes_sig_solution ();
    void assign_old_solution ();
    void output_results () const;
    void output_results_parallel  (const std::string &filename_base);
    void output_results_parallel2 (const std::string &filename_base);


    void refine_mesh ();
    double compute_viscosity(const Tensor<2, dim> &u_grad, const double &pice) const;
    double compute_pice (const double &h, const double &c) const;
    double compute_source_height (const double &h, const double &c) const;

    double compute_E (const Tensor<2, dim> &u_grad) const;
    void bound_conc ();
    void bound_thick ();
    void print_mesh_info (const Triangulation<dim> &tria,
                          const std::string        &filename);

    parallel::distributed::Triangulation<dim> triangulation;
    double                              global_Omega_diameter;

    ConditionalOStream                  pcout;
    const MappingQ<dim>                 mapping;


    const unsigned int                  stokes_degree;
    FESystem<dim>                       stokes_fe;
    DoFHandler<dim>                     stokes_dof_handler;
    ConstraintMatrix                    stokes_constraints;

    std::vector<IndexSet>               stokes_partitioning;
    TrilinosWrappers::SparseMatrix      stokes_matrix;
    TrilinosWrappers::SparseMatrix      stokes_preconditioner_matrix;

    TrilinosWrappers::MPI::Vector       stokes_solution;
    TrilinosWrappers::MPI::Vector       old_stokes_solution,
	                                sub_old_stokes_solution,
									    prd_stokes_solution,
										cor_stokes_solution,
										dif_stokes_solution;
										//tp1_stokes_solution,
    TrilinosWrappers::MPI::Vector       stokes_rhs;





    const unsigned int                  p_degree;
    FE_Q<dim>                           p_fe;
    DoFHandler<dim>                     p_dof_handler;
    ConstraintMatrix                    p_constraints;
    TrilinosWrappers::SparseMatrix      p_matrix;
    TrilinosWrappers::MPI::Vector       p_rhs;
    TrilinosWrappers::MPI::Vector       p_solution,
	                                    prd_p_solution,
										dif_p_solution,
	                                    old_p_solution;



    const unsigned int                  sig_degree;
    FESystem<dim>                       sig_fe;
    DoFHandler<dim>                     sig_dof_handler;
    ConstraintMatrix                    sig_constraints;
    TrilinosWrappers::SparseMatrix      sig_matrix;

    TrilinosWrappers::MPI::Vector       sig_rhs;
    TrilinosWrappers::MPI::Vector       sig_solution,
							        old_sig_solution;

    const unsigned int                  sig_diag_degree;
    FE_Q<dim>                           sig_diag_fe;
    DoFHandler<dim>                     sig_diag_dof_handler;
    ConstraintMatrix                    sig_diag_constraints;
    TrilinosWrappers::SparseMatrix      sig_diag_matrix;

    TrilinosWrappers::MPI::Vector       sig_diag_rhs;
    TrilinosWrappers::MPI::Vector       sig_diag_solution,
							        old_sig_diag_solution;

    const unsigned int                  thick_degree;
    FE_Q<dim>                           thick_fe;
    DoFHandler<dim>                     thick_dof_handler;
    ConstraintMatrix                    thick_constraints;
    TrilinosWrappers::SparseMatrix      thick_matrix;
    TrilinosWrappers::MPI::Vector       thick_rhs;


    TrilinosWrappers::MPI::Vector       thick_solution,
	                                old_thick_solution,
									tmp_thick_solution;




    const unsigned int                  conc_degree;
    FE_Q<dim>                           conc_fe;
    DoFHandler<dim>                     conc_dof_handler;
    ConstraintMatrix                    conc_constraints;
    TrilinosWrappers::SparseMatrix      conc_matrix;
    TrilinosWrappers::MPI::Vector       conc_rhs;
    TrilinosWrappers::MPI::Vector       conc_solution,
	                                old_conc_solution,
									tmp_conc_solution;



    double                              time_step;
    double                              old_time_step;
    unsigned int                        timestep_number;


    bool                                rebuild_stokes_matrix;
    bool                                rebuild_stokes_preconditioner;
    bool                                rebuild_p_matrix;
    bool                                rebuild_sig_matrix;
    bool                                rebuild_sig_diag_matrix;
    bool                                rebuild_thick_matrix;
    bool                                if_carreau;
    bool                                subcycle_stokes_sig;
    bool                                pseudo_stokes;
	bool                                update_viscosity_every_subcycle;


    int isig11,isig22;
    unsigned int                        max_pseudo_step;

    TimerOutput                         computing_timer;
    std::string                         output_dir;

    void setup_stokes_matrices (const IndexSet &stokes_partitioning,
                                const IndexSet &stokes_relevant_partitioning);

    void setup_p_matrices (const IndexSet &p_partitioning,
                           const IndexSet &p_relevant_partitioning);

    void setup_sig_matrices (const IndexSet &sig_partitioning,
                             const IndexSet &sig_relevant_partitioning);

    void setup_sig_diag_matrices (const IndexSet &sig_diag_partitioning,
                                  const IndexSet &sig_diag_relevant_partitioning);

    void setup_thick_matrices (const IndexSet &thick_partitioning,
                               const IndexSet &thick_relevant_partitioning);

    void setup_conc_matrices (const IndexSet &conc_partitioning,
                              const IndexSet &conc_relevant_partitioning);

    void local_assemble_stokes_matrix (const typename DoFHandler<dim>::active_cell_iterator &cell,
                                       Assembly::Scratch ::StokesSystem<dim>  &scratch,
                                       Assembly::CopyData::StokesSystem<dim>  &data,
									   const bool &ifcor);


    void local_assemble_p_matrix      (const typename DoFHandler<dim>::active_cell_iterator &cell,
                                       Assembly::Scratch ::PSystem<dim>  &scratch,
                                       Assembly::CopyData::PSystem<dim>  &data,
									   const bool &ifcor);

    void local_assemble_sig_matrix    (const typename DoFHandler<dim>::active_cell_iterator &cell,
                                       Assembly::Scratch ::SigSystem<dim>  &scratch,
                                       Assembly::CopyData::SigSystem<dim>  &data);

    void local_assemble_sig_diag_matrix    (const typename DoFHandler<dim>::active_cell_iterator &cell,
                                            Assembly::Scratch ::SigDiagSystem<dim>  &scratch,
                                            Assembly::CopyData::SigDiagSystem<dim>  &data);
    void local_assemble_thick_conc_matrix    (const typename DoFHandler<dim>::active_cell_iterator &cell,
                                              Assembly::Scratch ::ThickConcSystem<dim>  &scratch,
                                              Assembly::CopyData::ThickConcSystem<dim>  &data);

    void copy_local_to_global_stokes_matrix     (const Assembly::CopyData::StokesSystem<dim>    &data);
    void copy_local_to_global_p_matrix          (const Assembly::CopyData::PSystem<dim>         &data);
    void copy_local_to_global_sig_matrix        (const Assembly::CopyData::SigSystem<dim>       &data);
    void copy_local_to_global_sig_diag_matrix   (const Assembly::CopyData::SigDiagSystem<dim>   &data);
    void copy_local_to_global_thick_conc_matrix (const Assembly::CopyData::ThickConcSystem<dim> &data);

    class Postprocessor;
    std_cxx11::shared_ptr<TrilinosWrappers::PreconditionAMG>    Amg_preconditioner;

  };



  template <int dim>
  SeaIceRheologyProblem<dim>::SeaIceRheologyProblem (const ParameterHandler &prm)
    :
	pcout (std::cout,
        (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)
        == 0)),

	triangulation (MPI_COMM_WORLD,
			       typename Triangulation<dim>::MeshSmoothing
	              (Triangulation<dim>::smoothing_on_refinement |
	              Triangulation<dim>::smoothing_on_coarsening)),

    mapping (2),

    stokes_degree (1),
    stokes_fe (FE_Q<dim>(stokes_degree), dim),
    stokes_dof_handler (triangulation),



	p_degree (1),
	p_fe (p_degree),
	p_dof_handler (triangulation),

    sig_degree (1),
    sig_fe (FE_Q<dim>(sig_degree), 2),
    sig_dof_handler (triangulation),

    sig_diag_degree (1),
    sig_diag_fe (sig_diag_degree),
    sig_diag_dof_handler (triangulation),

    thick_degree (1),
    thick_fe (thick_degree),
    thick_dof_handler (triangulation),

	conc_degree (thick_degree),
    conc_fe (conc_degree),
    conc_dof_handler (triangulation),



    time_step (0),
    old_time_step (0),
    timestep_number (0),
    rebuild_stokes_matrix (true),
    rebuild_stokes_preconditioner (true),
	rebuild_sig_matrix (true),
	rebuild_sig_diag_matrix (true),
    rebuild_thick_matrix (true),
	rebuild_p_matrix (true),
	if_carreau (true),
	subcycle_stokes_sig (EquationData::subcycle_stokes_sig),
	pseudo_stokes (EquationData::pseudo_stokes),
	update_viscosity_every_subcycle (true),


	isig11(0),
	isig22(1),
	max_pseudo_step (0),

	output_dir ("data/"),

    computing_timer (MPI_COMM_WORLD,
                     pcout,
                     TimerOutput::summary,
                     TimerOutput::wall_times)
	{}










  template <int dim>
  void SeaIceRheologyProblem<dim>::setup_dofs ()
  {
	  computing_timer.enter_section("Setup dof systems");

	  stokes_dof_handler.distribute_dofs (stokes_fe);
	  p_dof_handler.distribute_dofs (p_fe);
	  sig_dof_handler.distribute_dofs (sig_fe);
	  sig_diag_dof_handler.distribute_dofs (sig_diag_fe);
	  thick_dof_handler.distribute_dofs (thick_fe);
	  conc_dof_handler.distribute_dofs (conc_fe);


	  const unsigned int n_stk = stokes_dof_handler.n_dofs(),
			             n_sig = sig_dof_handler.n_dofs(),
			             n_sig_diag = sig_diag_dof_handler.n_dofs(),
			             n_h   = thick_dof_handler.n_dofs(),
			             n_c   = conc_dof_handler.n_dofs(),
			             n_p   = p_dof_handler.n_dofs();


	  pcout << "Number of active cells: "
			  << triangulation.n_active_cells()
			  << " (on "
			  << triangulation.n_levels()
			  << " levels)"
			  << std::endl
			  << "Number of degrees of freedom: "
			  << n_stk + n_sig + n_sig_diag + n_h + n_c + n_p
			  << '(' << n_stk << '+' << n_sig << '+' << n_sig_diag << '+' << n_h << '+' << n_c << '+' <<n_p<<')'
			  << std::endl
			  << std::endl;

	  IndexSet stokes_partitioning (n_stk), stokes_relevant_partitioning (n_stk),
			   p_partitioning (n_p), p_relevant_partitioning (n_p),
			   sig_partitioning (n_sig), sig_relevant_partitioning (n_sig),
			   sig_diag_partitioning (n_sig_diag), sig_diag_relevant_partitioning (n_sig_diag),
			   thick_partitioning (n_h), thick_relevant_partitioning (n_h),
			   conc_partitioning (n_c), conc_relevant_partitioning (n_c);


	  stokes_partitioning = stokes_dof_handler.locally_owned_dofs();
	  DoFTools::extract_locally_relevant_dofs (stokes_dof_handler,
			                                   stokes_relevant_partitioning);

	  p_partitioning = p_dof_handler.locally_owned_dofs();
	  DoFTools::extract_locally_relevant_dofs (p_dof_handler,
			                                   p_relevant_partitioning);

	  sig_partitioning = sig_dof_handler.locally_owned_dofs();
	  DoFTools::extract_locally_relevant_dofs (sig_dof_handler,
			                                   sig_relevant_partitioning);

	  sig_diag_partitioning = sig_diag_dof_handler.locally_owned_dofs();
	  DoFTools::extract_locally_relevant_dofs (sig_diag_dof_handler,
			                                   sig_diag_relevant_partitioning);

	  thick_partitioning = thick_dof_handler.locally_owned_dofs();
	  DoFTools::extract_locally_relevant_dofs (thick_dof_handler,
			                                   thick_relevant_partitioning);

	  conc_partitioning = conc_dof_handler.locally_owned_dofs();
	  DoFTools::extract_locally_relevant_dofs (conc_dof_handler,
			                                   conc_relevant_partitioning);


    {

      stokes_constraints.clear ();
      stokes_constraints.reinit (stokes_relevant_partitioning);

      FEValuesExtractors::Vector velocities(0);
      FEValuesExtractors::Scalar velocity_y(1);

      DoFTools::make_hanging_node_constraints (stokes_dof_handler,
                                               stokes_constraints);


/////////////////////Inlet BC/////////////////////////////////////
////      Inlet Poiseuille flow
//      VectorTools::interpolate_boundary_values (stokes_dof_handler,
//                                                1,
//												PoiseuilleBoundaryValues<dim>(),
//                                                stokes_constraints,
//                                                stokes_fe.component_mask(velocities));

//      Inlet Zero vertical velocity BC

//      VectorTools::interpolate_boundary_values (stokes_dof_handler,
//                                                0,
//                                                NoVerticalBoundaryValues<dim>(),
//                                                stokes_constraints,
//                                                stokes_fe.component_mask(velocity_y));
//
//      VectorTools::interpolate_boundary_values (stokes_dof_handler,
//                                                1,
//                                                NoVerticalBoundaryValues<dim>(),
//                                                stokes_constraints,
//                                                stokes_fe.component_mask(velocity_y));

//      	VectorTools::interpolate_boundary_values (stokes_dof_handler,
//                                                  1,
//                                                  NoSlipBoundaryValues<dim>(),
//                                                  stokes_constraints,
//                                                  stokes_fe.component_mask(velocities));

/////////////////////Inlet BC/////////////////////////////////////
      VectorTools::interpolate_boundary_values (stokes_dof_handler,
                                                0,
                                                NoVerticalBoundaryValues<dim>(),
                                                stokes_constraints,
                                                stokes_fe.component_mask(velocity_y));

//      VectorTools::interpolate_boundary_values (stokes_dof_handler,
//                                                1,
//                                                NoVerticalBoundaryValues<dim>(),
//                                                stokes_constraints,
//                                                stokes_fe.component_mask(velocity_y));
//
//      VectorTools::interpolate_boundary_values (stokes_dof_handler,
//                                                2,
//                                                NoVerticalBoundaryValues<dim>(),
//                                                stokes_constraints,
//                                                stokes_fe.component_mask(velocity_y));

      VectorTools::interpolate_boundary_values (stokes_dof_handler,
                                                3,
                                                NoVerticalBoundaryValues<dim>(),
                                                stokes_constraints,
                                                stokes_fe.component_mask(velocity_y));

      VectorTools::interpolate_boundary_values (stokes_dof_handler,
                                                4,
                                                NoSlipBoundaryValues<dim>(),
                                                stokes_constraints,
                                                stokes_fe.component_mask(velocities));
//
//


      stokes_constraints.close ();
    }


    {
      p_constraints.clear ();
      p_constraints.reinit (p_relevant_partitioning);

      DoFTools::make_hanging_node_constraints (p_dof_handler,
                                               p_constraints);
      p_constraints.close ();
    }


    {

      sig_constraints.clear ();
      sig_constraints.reinit (sig_relevant_partitioning);

      DoFTools::make_hanging_node_constraints (sig_dof_handler,
                                               sig_constraints);
      sig_constraints.close ();
    }

    {
      sig_diag_constraints.clear ();
      sig_diag_constraints.reinit (sig_diag_relevant_partitioning);

      DoFTools::make_hanging_node_constraints (sig_diag_dof_handler,
                                               sig_diag_constraints);
      sig_diag_constraints.close ();
    }


    {
      thick_constraints.clear ();
      thick_constraints.reinit (thick_relevant_partitioning);

      DoFTools::make_hanging_node_constraints (thick_dof_handler,
                                               thick_constraints);

////      ///////////Gmesh////////////
//      if(IceModel::ifperi_wd != true)
      VectorTools::interpolate_boundary_values (thick_dof_handler,
                                                0,
                                                InletThickBoundaryValues<dim>(),
                                                thick_constraints);
//      ///////////Gmesh////////////

      ///////////Rectangle dealII Mesh////////////
//      VectorTools::interpolate_boundary_values (thick_dof_handler,
//                                                1,
//                                                InletThickBoundaryValues<dim>(),
//                                                thick_constraints);
      ///////////Rectangle dealII Mesh////////////

      thick_constraints.close ();
    }

    {
      conc_constraints.clear ();
      conc_constraints.reinit (conc_relevant_partitioning);

      DoFTools::make_hanging_node_constraints (conc_dof_handler,
                                               conc_constraints);

//      ///////////Gmesh////////////
      if(IceModel::ifperi_wd != true)
      VectorTools::interpolate_boundary_values (conc_dof_handler,
                                                0,
                                                InletConcBoundaryValues<dim>(),
                                                conc_constraints);
//      ///////////Gmesh////////////

//      ///////////Rectangle dealII Mesh////////////
//      VectorTools::interpolate_boundary_values (conc_dof_handler,
//                                                1,
//                                                InletConcBoundaryValues<dim>(),
//                                                conc_constraints);
//      ///////////Rectangle dealII Mesh////////////

      conc_constraints.close ();
    }


    setup_stokes_matrices (stokes_partitioning, stokes_relevant_partitioning);
    setup_p_matrices (p_partitioning, p_relevant_partitioning);
    setup_sig_matrices (sig_partitioning, sig_relevant_partitioning);
    setup_sig_diag_matrices (sig_diag_partitioning, sig_diag_relevant_partitioning);
    setup_thick_matrices (thick_partitioning, thick_relevant_partitioning);
    setup_conc_matrices (conc_partitioning, conc_relevant_partitioning);


    stokes_rhs.reinit   (stokes_partitioning, stokes_relevant_partitioning, MPI_COMM_WORLD, true);
    p_rhs.reinit        (p_partitioning, p_relevant_partitioning, MPI_COMM_WORLD, true);
    sig_rhs.reinit      (sig_partitioning, sig_relevant_partitioning, MPI_COMM_WORLD, true);
    sig_diag_rhs.reinit (sig_diag_partitioning, sig_diag_relevant_partitioning, MPI_COMM_WORLD, true);
    thick_rhs.reinit    (thick_partitioning, thick_relevant_partitioning, MPI_COMM_WORLD, true);
    conc_rhs.reinit     (conc_partitioning, conc_relevant_partitioning, MPI_COMM_WORLD, true);


    stokes_solution.reinit (stokes_relevant_partitioning, MPI_COMM_WORLD);
    old_stokes_solution.reinit (stokes_solution);
    sub_old_stokes_solution.reinit (stokes_solution);
    prd_stokes_solution.reinit (stokes_solution);
    //tp1_stokes_solution.reinit (stokes_solution);

    cor_stokes_solution.reinit (stokes_rhs);
    dif_stokes_solution.reinit (stokes_rhs);
//    tp2_stokes_solution.reinit (stokes_rhs);

    p_solution.reinit (p_relevant_partitioning, MPI_COMM_WORLD);
    old_p_solution.reinit (p_solution);
    prd_p_solution.reinit (p_solution);
    dif_p_solution.reinit (p_solution);


    sig_solution.reinit (sig_relevant_partitioning, MPI_COMM_WORLD);
    old_sig_solution.reinit (sig_solution);

    sig_diag_solution.reinit (sig_diag_relevant_partitioning, MPI_COMM_WORLD);
    old_sig_diag_solution.reinit (sig_diag_solution);

    thick_solution.reinit (thick_relevant_partitioning, MPI_COMM_WORLD);
    old_thick_solution.reinit (thick_solution);
    tmp_thick_solution.reinit (thick_rhs);

    conc_solution.reinit (conc_relevant_partitioning, MPI_COMM_WORLD);
    old_conc_solution.reinit (conc_solution);

    tmp_conc_solution.reinit (conc_rhs);


    stokes_solution = 0;
    old_stokes_solution = 0;
    sub_old_stokes_solution = 0;

    p_solution = 0;
    old_p_solution = 0;
    prd_p_solution = 0;
    dif_p_solution = 0;


    sig_solution = 0;
    old_sig_solution = sig_solution;

    sig_diag_solution = 0;
    old_sig_diag_solution = sig_diag_solution;

    conc_solution = IceModel::c0;
    old_conc_solution = conc_solution;

    thick_solution = IceModel::h0;
    old_thick_solution = thick_solution;

    computing_timer.exit_section();

  }

  template <int dim>
  void SeaIceRheologyProblem<dim>::
  setup_stokes_matrices (const IndexSet &stokes_partitioner,
                         const IndexSet &stokes_relevant_partitioner)
  {
    stokes_matrix.clear ();

    TrilinosWrappers::SparsityPattern sp(stokes_partitioner,
                                         stokes_partitioner,
                                         stokes_relevant_partitioner,
                                         MPI_COMM_WORLD);


    DoFTools::make_sparsity_pattern (stokes_dof_handler, sp,
                                     stokes_constraints, false,
                                     Utilities::MPI::
                                     this_mpi_process(MPI_COMM_WORLD));
    sp.compress();

    stokes_matrix.reinit (sp);
  }

  template <int dim>
  void SeaIceRheologyProblem<dim>::
  setup_p_matrices (const IndexSet &p_partitioner,
                    const IndexSet &p_relevant_partitioner)
  {
    p_matrix.clear ();

    TrilinosWrappers::SparsityPattern sp(p_partitioner,
                                         p_partitioner,
                                         p_relevant_partitioner,
                                         MPI_COMM_WORLD);


    DoFTools::make_sparsity_pattern (p_dof_handler, sp,
                                     p_constraints, false,
                                     Utilities::MPI::
                                     this_mpi_process(MPI_COMM_WORLD));
    sp.compress();

    p_matrix.reinit (sp);
  }

  template <int dim>
  void SeaIceRheologyProblem<dim>::
  setup_sig_matrices (const IndexSet &sig_partitioner,
                      const IndexSet &sig_relevant_partitioner)
  {
    sig_matrix.clear ();

    TrilinosWrappers::SparsityPattern sp(sig_partitioner,
                                         sig_partitioner,
                                         sig_relevant_partitioner,
                                         MPI_COMM_WORLD);


    DoFTools::make_sparsity_pattern (sig_dof_handler, sp,
                                     sig_constraints, false,
                                     Utilities::MPI::
                                     this_mpi_process(MPI_COMM_WORLD));
    sp.compress();

    sig_matrix.reinit (sp);
  }

  template <int dim>
  void SeaIceRheologyProblem<dim>::
  setup_sig_diag_matrices (const IndexSet &sig_diag_partitioner,
                           const IndexSet &sig_diag_relevant_partitioner)
  {
    sig_diag_matrix.clear ();

    TrilinosWrappers::SparsityPattern sp(sig_diag_partitioner,
                                         sig_diag_partitioner,
                                         sig_diag_relevant_partitioner,
                                         MPI_COMM_WORLD);


    DoFTools::make_sparsity_pattern (sig_diag_dof_handler, sp,
                                     sig_diag_constraints, false,
                                     Utilities::MPI::
                                     this_mpi_process(MPI_COMM_WORLD));
    sp.compress();

    sig_diag_matrix.reinit (sp);
  }

  template <int dim>
  void SeaIceRheologyProblem<dim>::
  setup_thick_matrices (const IndexSet &thick_partitioner,
                        const IndexSet &thick_relevant_partitioner)
  {
    thick_matrix.clear ();

    TrilinosWrappers::SparsityPattern sp(thick_partitioner,
                                         thick_partitioner,
                                         thick_relevant_partitioner,
                                         MPI_COMM_WORLD);


    DoFTools::make_sparsity_pattern (thick_dof_handler, sp,
                                     thick_constraints, false,
                                     Utilities::MPI::
                                     this_mpi_process(MPI_COMM_WORLD));
    sp.compress();

    thick_matrix.reinit (sp);
  }

  template <int dim>
  void SeaIceRheologyProblem<dim>::
  setup_conc_matrices (const IndexSet &conc_partitioner,
                       const IndexSet &conc_relevant_partitioner)
  {
    conc_matrix.clear ();

    TrilinosWrappers::SparsityPattern sp(conc_partitioner,
                                         conc_partitioner,
                                         conc_relevant_partitioner,
                                         MPI_COMM_WORLD);


    DoFTools::make_sparsity_pattern (conc_dof_handler, sp,
                                     conc_constraints, false,
                                     Utilities::MPI::
                                     this_mpi_process(MPI_COMM_WORLD));
    sp.compress();

    conc_matrix.reinit (sp);
  }


  template <int dim>
  void SeaIceRheologyProblem<dim>::local_assemble_stokes_matrix(const typename DoFHandler<dim>::active_cell_iterator &cell,
                                                                Assembly::Scratch::StokesSystem<dim> &scratch,
                                                                Assembly::CopyData::StokesSystem<dim> &data,
																const bool &ifcor)
  {
	  const unsigned int dofs_per_cell = scratch.stokes_fe_values.get_fe().dofs_per_cell;
	  const unsigned int n_q_points    = scratch.stokes_fe_values.n_quadrature_points;

	  const FEValuesExtractors::Vector velocities (0);
	  const FEValuesExtractors::Scalar pressure (dim);

	  const RightHandSide<dim> right_hand_side;

	  std::vector<double> coef (n_q_points);

	  scratch.stokes_fe_values.reinit (cell);

	  typename DoFHandler<dim>::active_cell_iterator
	  p_cell (&triangulation,
			  cell->level(),
			  cell->index(),
			  &p_dof_handler);
	  scratch.p_fe_values.reinit (p_cell);


	  typename DoFHandler<dim>::active_cell_iterator
	  thick_cell (&triangulation,
			      cell->level(),
			      cell->index(),
			      &thick_dof_handler);
	  scratch.thick_fe_values.reinit (thick_cell);


	  typename DoFHandler<dim>::active_cell_iterator
	  conc_cell (&triangulation,
			     cell->level(),
			     cell->index(),
			     &conc_dof_handler);
	  scratch.conc_fe_values.reinit (conc_cell);

	  if (rebuild_stokes_matrix)
		  data.local_matrix = 0;

	  data.local_rhs = 0;


	  right_hand_side.vector_value_list(scratch.stokes_fe_values.get_quadrature_points(),
	                                    scratch.rhs_values);

//	  scratch.stokes_fe_values[velocities].get_function_values (old_stokes_solution,
//	                                                            scratch.old_velocity_values);

	  scratch.stokes_fe_values[velocities].get_function_values (((EquationData::ifstr_stokes==true && EquationData::pseudo_stokes==true) ? stokes_solution : old_stokes_solution),
	                                                            scratch.old_velocity_values);

	  scratch.stokes_fe_values[velocities].get_function_values (ifcor==true ? cor_stokes_solution : old_stokes_solution,
		                                                        scratch.old_velocity_values_wdrag);

	  scratch.stokes_fe_values[velocities].get_function_gradients (ifcor==true ? stokes_solution : old_stokes_solution,
	                                                               scratch.velocity_gradients); //updating viscosity based on the previous pseudo-step velocity


	  scratch.p_fe_values.get_function_gradients (p_solution,
	                                              scratch.old_pressure_gradients);

	  scratch.thick_fe_values.get_function_values (thick_solution,
			                                       scratch.value_thick);

	  scratch.conc_fe_values.get_function_values (conc_solution,
			                                      scratch.value_conc);


	  double eta,pice;

      for (unsigned int q=0; q<n_q_points; ++q)
      {
    	  coef [q] =   EquationData::Re
    			  / EquationData::dt
				  * scratch.value_thick[q];

    	  for (unsigned int i=0; i<dim; ++i)
    	  {
    		  scratch.rhs_values[q](i) += scratch.old_velocity_values[q][i] * coef[q]
										- scratch.old_pressure_gradients[q][i];
    	  }

    	  if(IceModel::ifcorio==true)
    	  {
    		  scratch.rhs_values[q](0) +=  coef[q]*EquationData::dt*IceModel::fco_non*scratch.old_velocity_values[q][1];
    		  scratch.rhs_values[q](1) += -coef[q]*EquationData::dt*IceModel::fco_non*scratch.old_velocity_values[q][0];
    	  }
      }

      for (unsigned int q=0; q<n_q_points; ++q)
        {
		  pice = scratch.value_thick[q]*std::exp(-IceModel::k*(1-scratch.value_conc[q]));
		  eta  = compute_viscosity(scratch.velocity_gradients[q], pice);

          for (unsigned int k=0; k<dofs_per_cell; ++k)
            {
              scratch.phi_u[k] = scratch.stokes_fe_values[velocities].value (k,q);
              if (rebuild_stokes_matrix)
                {
                  scratch.grads_phi_u[k] = scratch.stokes_fe_values[velocities].symmetric_gradient(k,q);
                  scratch.div_phi_u[k]   = scratch.stokes_fe_values[velocities].divergence (k, q);
                }
            }


          if (rebuild_stokes_matrix)
            for (unsigned int i=0; i<dofs_per_cell; ++i)
              for (unsigned int j=0; j<dofs_per_cell; ++j)
                data.local_matrix(i,j) += ((coef[q] +
                		(IceModel::ifwdrag ? IceModel::beta*std::pow(IceModel::alpha,4) *
                		 scratch.old_velocity_values_wdrag[q].norm(): 0) )
										  *scratch.phi_u[i]*scratch.phi_u[j]
								          +  eta * 2. *
						                     (scratch.grads_phi_u[i] * scratch.grads_phi_u[j]))
                                            * scratch.stokes_fe_values.JxW(q);
          else
          {
        	  for (unsigned int i=0; i<dofs_per_cell; ++i)
        	  {
        		  if (stokes_constraints.is_inhomogeneously_constrained(data.local_dof_indices[i]))
        			  for (unsigned int j=0; j<dofs_per_cell; ++j)
        			  {
        				  data.local_matrix(j,i) += (coef[q]*scratch.phi_u[i]*scratch.phi_u[j]
												  +  eta * 2. *
												     (scratch.grads_phi_u[i] * scratch.grads_phi_u[j]))
												  * scratch.stokes_fe_values.JxW(q);
        			  }
        	  }
          }


          for (unsigned int i=0; i<dofs_per_cell; ++i)
          {
          	const unsigned int component_i = stokes_fe.system_to_component_index(i).first;
          	data.local_rhs(i) += scratch.stokes_fe_values.shape_value(i,q) *
          			             scratch.rhs_values[q](component_i) *
						         scratch.stokes_fe_values.JxW(q);
          }
        }
      cell->get_dof_indices (data.local_dof_indices);
  }





  template <int dim>
  void SeaIceRheologyProblem<dim>::local_assemble_p_matrix(const typename DoFHandler<dim>::active_cell_iterator &cell,
                                                           Assembly::Scratch::PSystem<dim> &scratch,
                                                           Assembly::CopyData::PSystem<dim> &data,
														   const bool &ifcor)
   {
	  double coef = (IceModel::ifgra == true ? -1 : (std::pow(IceModel::alpha,2) - 1));

	  const unsigned int dofs_per_cell = scratch.p_fe_values.get_fe().dofs_per_cell;
	  const unsigned int n_q_points    = scratch.p_fe_values.n_quadrature_points;

 	  const FEValuesExtractors::Vector velocities (0);

	  scratch.p_fe_values.reinit (cell);
	  cell->get_dof_indices (data.local_dof_indices);


	  typename DoFHandler<dim>::active_cell_iterator
	  stokes_cell (&triangulation,
			       cell->level(),
			       cell->index(),
			       &stokes_dof_handler);
	  scratch.stokes_fe_values.reinit (stokes_cell);

	  typename DoFHandler<dim>::active_cell_iterator
	  thick_cell (&triangulation,
			      cell->level(),
			      cell->index(),
			      &thick_dof_handler);
	  scratch.thick_fe_values.reinit (thick_cell);

	  typename DoFHandler<dim>::active_cell_iterator
	  conc_cell (&triangulation,
			     cell->level(),
			     cell->index(),
			     &conc_dof_handler);
	  scratch.conc_fe_values.reinit (conc_cell);

	  if (rebuild_p_matrix == true)
		  data.local_matrix=0;

	  data.local_rhs=0;

	  scratch.thick_fe_values.get_function_values (thick_solution,
			  scratch.value_thick);
	  scratch.conc_fe_values.get_function_values  (conc_solution,
			  scratch.value_conc);

//	  scratch.stokes_fe_values[velocities].get_function_gradients  (old_stokes_solution,
//			  scratch.old_velocity_gradients);

	  scratch.stokes_fe_values[velocities].get_function_gradients  (ifcor==true ? stokes_solution : old_stokes_solution, // momentum equations are solved before pressure, so one can use the current velocity fields?
			  scratch.old_velocity_gradients);

	  for (unsigned int q=0; q<n_q_points; ++q)
	  {
		  if (rebuild_p_matrix == true)
			  for (unsigned int i=0; i<dofs_per_cell; ++i)
				  for (unsigned int j=0; j<dofs_per_cell; ++j)
				  {
					  data.local_matrix(i,j) +=  scratch.p_fe_values.shape_value (i, q) *
							  scratch.p_fe_values.shape_value (j, q) *
							  scratch.p_fe_values.JxW (q);
				  }
		  for (unsigned int i=0; i<dofs_per_cell; ++i)
		  {
			  double temp = scratch.p_fe_values.shape_value(i,q)
		  				   *scratch.p_fe_values.JxW(q);
			  double pice = scratch.value_thick[q]*std::exp(-IceModel::k*(1-scratch.value_conc[q]));
			  double eta  = compute_viscosity(scratch.old_velocity_gradients[q], pice);
			  data.local_rhs(i) += (IceModel::alpha*pice - eta*coef*(scratch.old_velocity_gradients[q][0][0]
								   +scratch.old_velocity_gradients[q][1][1]))
								   *temp;
		  }
	  }

      cell->get_dof_indices (data.local_dof_indices);

   }


  template <int dim>
  void SeaIceRheologyProblem<dim>::local_assemble_sig_matrix(const typename DoFHandler<dim>::active_cell_iterator &cell,
                                                             Assembly::Scratch::SigSystem<dim> &scratch,
                                                             Assembly::CopyData::SigSystem<dim> &data)
  {
	  const unsigned int dofs_per_cell = scratch.sig_fe_values.get_fe().dofs_per_cell;
	  const unsigned int n_q_points    = scratch.sig_fe_values.n_quadrature_points;

 	  const FEValuesExtractors::Vector velocities (0);
	  const FEValuesExtractors::Scalar sig11 (isig11),
			                           sig22 (isig22);

	  scratch.sig_fe_values.reinit (cell);
	  cell->get_dof_indices (data.local_dof_indices);


	  typename DoFHandler<dim>::active_cell_iterator
	  stokes_cell (&triangulation,
			       cell->level(),
			       cell->index(),
			       &stokes_dof_handler);
	  scratch.stokes_fe_values.reinit (stokes_cell);

	  typename DoFHandler<dim>::active_cell_iterator
	  p_cell (&triangulation,
	          cell->level(),
		      cell->index(),
			  &p_dof_handler);
	  scratch.p_fe_values.reinit (p_cell);

	  if (rebuild_sig_matrix == true)
		  data.local_matrix=0;

	  data.local_rhs=0;

	  double al2   =  std::pow(IceModel::alpha,2);
	  double coef  =  EquationData::chi*(subcycle_stokes_sig == true ? EquationData::dte_inv : 1/EquationData::dt);
	  double coef2 =  (1-al2)/4;
	  double coef3 =  al2 / 2;
	  double coef4 = -IceModel::alpha/2;

	  scratch.stokes_fe_values[velocities].get_function_gradients
	  			((update_viscosity_every_subcycle == true) ? stokes_solution : sub_old_stokes_solution,
	  			scratch.grads_velocity);


	  scratch.p_fe_values.get_function_values      (p_solution,
			                                        scratch.value_pressure);

	  scratch.sig_fe_values[sig11].get_function_values (old_sig_solution,
			                                            scratch.old_sig11_value);

	  scratch.sig_fe_values[sig22].get_function_values (old_sig_solution,
			                                            scratch.old_sig22_value);

      for (unsigned int q=0; q<n_q_points; ++q)
      {
  		double eta  = compute_viscosity (scratch.grads_velocity[q], scratch.value_pressure[q]);

  		for (unsigned int i=0; i<dofs_per_cell; ++i)
  		{
  			scratch.phi_sig11[i] = scratch.sig_fe_values[sig11].value (i,q);
  			scratch.phi_sig22[i] = scratch.sig_fe_values[sig22].value (i,q);
  			scratch.phi_sig_diag_sum[i] = scratch.phi_sig11 [i] + scratch.phi_sig22 [i];
  		}

		if (rebuild_sig_matrix == true)
    		for (unsigned int i=0; i<dofs_per_cell; ++i)
    		  for (unsigned int j=0; j<dofs_per_cell; ++j)
    		    data.local_matrix(i,j) +=  (
    		    		              (scratch.phi_sig11 [i] * scratch.phi_sig11 [j]
    		                         + scratch.phi_sig22 [i] * scratch.phi_sig22 [j]
									   )*(coef + coef3)
									 + (scratch.phi_sig_diag_sum [i])
									 * (scratch.phi_sig_diag_sum [j])*coef2
									   )
									  *
									  scratch.sig_fe_values.JxW (q);

		for (unsigned int i=0; i<dofs_per_cell; ++i)
		{

			data.local_rhs(i) += (coef4 * scratch.value_pressure[q] * scratch.phi_sig_diag_sum [i]
							     +al2 * eta *(scratch.phi_sig11 [i] * scratch.grads_velocity[q][0][0] +
											  scratch.phi_sig22 [i] * scratch.grads_velocity[q][1][1])
								 +coef *(scratch.phi_sig11 [i] * scratch.old_sig11_value[q] +
										 scratch.phi_sig22 [i] * scratch.old_sig22_value[q])
			                     )*scratch.sig_fe_values.JxW (q);
		}
      }

      cell->get_dof_indices (data.local_dof_indices);
  }

  template <int dim>
  void SeaIceRheologyProblem<dim>::local_assemble_sig_diag_matrix(const typename DoFHandler<dim>::active_cell_iterator &cell,
                                                                  Assembly::Scratch::SigDiagSystem<dim> &scratch,
                                                                  Assembly::CopyData::SigDiagSystem<dim> &data)
  {
	  const unsigned int dofs_per_cell = scratch.sig_diag_fe_values.get_fe().dofs_per_cell;
	  const unsigned int n_q_points    = scratch.sig_diag_fe_values.n_quadrature_points;

 	  const FEValuesExtractors::Vector velocities (0);

	  scratch.sig_diag_fe_values.reinit (cell);
	  cell->get_dof_indices (data.local_dof_indices);

	  typename DoFHandler<dim>::active_cell_iterator
	  stokes_cell (&triangulation,
			       cell->level(),
			       cell->index(),
			       &stokes_dof_handler);
	  scratch.stokes_fe_values.reinit (stokes_cell);

	  typename DoFHandler<dim>::active_cell_iterator
	  p_cell (&triangulation,
	          cell->level(),
		      cell->index(),
			  &p_dof_handler);
	  scratch.p_fe_values.reinit (p_cell);

	  if (rebuild_sig_diag_matrix == true)
		  data.local_matrix=0;

	  data.local_rhs=0;

	  double al2   =  std::pow(IceModel::alpha,2);
	  double coef  =  EquationData::chi*(subcycle_stokes_sig == true ? EquationData::dte_inv : 1/EquationData::dt);
	  double coef2 =  (1-al2)/4;
	  double coef3 =  al2 / 2;
	  double coef4 = -IceModel::alpha/2;

	  scratch.stokes_fe_values[velocities].get_function_gradients
	                                       ((update_viscosity_every_subcycle == true) ? stokes_solution : sub_old_stokes_solution,
			                                scratch.grads_velocity);
	  scratch.p_fe_values.get_function_values      (p_solution,
			                                        scratch.value_pressure);

	  scratch.sig_diag_fe_values.get_function_values (old_sig_diag_solution,
			                                          scratch.old_sig_diag_value);

	  for (unsigned int q=0; q<n_q_points; ++q)
	  {
		  double eta  = compute_viscosity (scratch.grads_velocity[q], scratch.value_pressure[q]);

		  if (rebuild_sig_diag_matrix == true)
			  for (unsigned int i=0; i<dofs_per_cell; ++i)
				  for (unsigned int j=0; j<dofs_per_cell; ++j)
					  data.local_matrix(i,j) +=
							                   scratch.sig_diag_fe_values.shape_value (i,q)
											 * scratch.sig_diag_fe_values.shape_value (j,q)
							                 * (coef + coef3)
							                 * scratch.sig_diag_fe_values.JxW (q);

		  for (unsigned int i=0; i<dofs_per_cell; ++i)
		  {

			  data.local_rhs(i) += (
					  al2 * eta *
					  (scratch.sig_diag_fe_values.shape_value (i,q) * (scratch.grads_velocity[q][0][1]
					                                                 + scratch.grads_velocity[q][1][0])/2)
					 + coef *
					  (scratch.sig_diag_fe_values.shape_value (i,q) * scratch.old_sig_diag_value[q])
			          )*scratch.sig_diag_fe_values.JxW (q);
		  }
	  }

	  cell->get_dof_indices (data.local_dof_indices);
  }

  template <int dim>
    void SeaIceRheologyProblem<dim>::local_assemble_thick_conc_matrix(const typename DoFHandler<dim>::active_cell_iterator &cell,
                                                                      Assembly::Scratch::ThickConcSystem<dim> &scratch,
                                                                      Assembly::CopyData::ThickConcSystem<dim> &data)
    {
	  const unsigned int dofs_per_cell = scratch.thick_fe_values.get_fe().dofs_per_cell;
	  const unsigned int n_q_points    = scratch.thick_fe_values.n_quadrature_points;

 	  const FEValuesExtractors::Vector velocities (0);

	  scratch.thick_fe_values.reinit (cell);
	  cell->get_dof_indices (data.local_dof_indices);

	  typename DoFHandler<dim>::active_cell_iterator
	  stokes_cell (&triangulation,
			       cell->level(),
			       cell->index(),
			       &stokes_dof_handler);
	  scratch.stokes_fe_values.reinit (stokes_cell);

	  typename DoFHandler<dim>::active_cell_iterator
	  conc_cell (&triangulation,
			     cell->level(),
			     cell->index(),
			     &conc_dof_handler);
	  scratch.conc_fe_values.reinit (conc_cell);

	  if (rebuild_thick_matrix == true)
		  data.local_matrix=0;

	  data.local_rhs=0;
	  data.conc_local_rhs=0;

      scratch.stokes_fe_values[velocities].get_function_gradients (stokes_solution,
    		                               scratch.grads_velocity);

      scratch.stokes_fe_values[velocities].get_function_values    (stokes_solution,
    		                               scratch.velocity);

      scratch.thick_fe_values.get_function_values (old_thick_solution,
    		                  scratch.old_thick_values);

      scratch.conc_fe_values.get_function_values  (old_conc_solution,
    		                 scratch.old_conc_values);


      for (unsigned int q=0; q<n_q_points; ++q)
        {

      	if (rebuild_thick_matrix == true)
      		for (unsigned int i=0; i<dofs_per_cell; ++i)
      		  for (unsigned int j=0; j<dofs_per_cell; ++j)
      		  {
      			  data.local_matrix(i,j) +=  scratch.thick_fe_values.shape_value(i,q) *
      					                    (scratch.thick_fe_values.shape_grad(j,q)*scratch.velocity[q] +
      					            		 scratch.thick_fe_values.shape_value(j,q)*(1/EquationData::dt +
      					            	     scratch.grads_velocity[q][0][0] + scratch.grads_velocity[q][1][1])) *
											 scratch.thick_fe_values.JxW(q);
      		  }
      	for (unsigned int i=0; i<dofs_per_cell; ++i)
      	{
      		double temp = scratch.thick_fe_values.shape_value(i,q)*scratch.thick_fe_values.JxW(q);

      		data.local_rhs(i) += temp * (scratch.old_thick_values[q]/EquationData::dt
      				+ (((IceModel::ifthermal_cmax ? IceModel::c0 : 1)-scratch.old_conc_values[q])
      			    + scratch.old_conc_values[q]*compute_source_height(scratch.old_thick_values[q],scratch.old_conc_values[q]))
					* (IceModel::ifthermal==true ? IceModel::Sigma : 0));
      		data.conc_local_rhs(i) += temp * (scratch.old_conc_values[q]/ EquationData::dt
      				+ ((IceModel::ifthermal_cmax ? IceModel::c0 : 1)-scratch.old_conc_values[q])*(IceModel::ifthermal==true ? IceModel::Sigma/IceModel::hd_non : 0));
      	}

        }
	  cell->get_dof_indices (data.local_dof_indices);
    }

  template <int dim>
  void SeaIceRheologyProblem<dim>::
  copy_local_to_global_stokes_matrix (const Assembly::CopyData::StokesSystem<dim> &data)
  {
    if (rebuild_stokes_matrix == true)
      stokes_constraints.distribute_local_to_global (data.local_matrix,
                                                     data.local_rhs,
                                                     data.local_dof_indices,
                                                     stokes_matrix,
                                                     stokes_rhs);
    else
    {
    ///SOME THING COMPLICATED TO IMPLEMENT FOR HANGING NODES AND DIRICH BC
    	stokes_constraints.distribute_local_to_global (data.local_rhs,
                                                       data.local_dof_indices,
                                                       stokes_rhs,
													   data.local_matrix);
    }
    //rebuild_stokes_matrix == false;
  }

  template <int dim>
  void SeaIceRheologyProblem<dim>::
  copy_local_to_global_p_matrix (const Assembly::CopyData::PSystem<dim> &data)
  {
      if (rebuild_p_matrix == true)
      	p_constraints.distribute_local_to_global (data.local_matrix,
                                                  data.local_rhs,
                                                  data.local_dof_indices,
                                                  p_matrix,
                                                  p_rhs);
      else
      	p_constraints.distribute_local_to_global (data.local_rhs,
                                                  data.local_dof_indices,
                                                  p_rhs);
  }

  template <int dim>
  void SeaIceRheologyProblem<dim>::
  copy_local_to_global_sig_matrix (const Assembly::CopyData::SigSystem<dim> &data)
  {
      if (rebuild_sig_matrix == true)
      	sig_constraints.distribute_local_to_global (data.local_matrix,
                                                    data.local_rhs,
                                                    data.local_dof_indices,
                                                    sig_matrix,
                                                    sig_rhs);
      else
      	sig_constraints.distribute_local_to_global (data.local_rhs,
                                                    data.local_dof_indices,
                                                    sig_rhs);
  }

  template <int dim>
  void SeaIceRheologyProblem<dim>::
  copy_local_to_global_sig_diag_matrix (const Assembly::CopyData::SigDiagSystem<dim> &data)
  {
      if (rebuild_sig_diag_matrix == true)
      	sig_diag_constraints.distribute_local_to_global (data.local_matrix,
                                                         data.local_rhs,
                                                         data.local_dof_indices,
                                                         sig_diag_matrix,
                                                         sig_diag_rhs);
      else
      	sig_diag_constraints.distribute_local_to_global (data.local_rhs,
                                                         data.local_dof_indices,
                                                         sig_diag_rhs);
  }

  template <int dim>
  void SeaIceRheologyProblem<dim>::
  copy_local_to_global_thick_conc_matrix (const Assembly::CopyData::ThickConcSystem<dim> &data)
  {
      if (rebuild_thick_matrix == true)
      {
    	  thick_constraints.distribute_local_to_global (data.local_matrix,
                                                        data.local_rhs,
                                                        data.local_dof_indices,
                                                        thick_matrix,
                                                        thick_rhs);

    	  conc_constraints.distribute_local_to_global  (data.local_matrix,
    	                                                data.conc_local_rhs,
    	                                                data.local_dof_indices,
    	                                                conc_matrix,
    	                                                conc_rhs);
      }
      else
      {
      	thick_constraints.distribute_local_to_global (data.local_rhs,
                                                      data.local_dof_indices,
                                                      thick_rhs);

      	conc_constraints.distribute_local_to_global  (data.conc_local_rhs,
                                                      data.local_dof_indices,
                                                      conc_rhs);
      }
  }

  template <int dim>
    class SeaIceRheologyProblem<dim>::Postprocessor : public DataPostprocessor<dim>
    {
    public:
      Postprocessor (const unsigned int partition,
                     const double       minimal_pressure);

      virtual
      void
      compute_derived_quantities_vector (const std::vector<Vector<double> >              &uh,
                                         const std::vector<std::vector<Tensor<1,dim> > > &duh,
                                         const std::vector<std::vector<Tensor<2,dim> > > &dduh,
                                         const std::vector<Point<dim> >                  &normals,
                                         const std::vector<Point<dim> >                  &evaluation_points,
                                         std::vector<Vector<double> >                    &computed_quantities) const;

      virtual std::vector<std::string> get_names () const;

      virtual
      std::vector<DataComponentInterpretation::DataComponentInterpretation>
      get_data_component_interpretation () const;

      virtual UpdateFlags get_needed_update_flags () const;

    private:
      const unsigned int partition;
      const double       minimal_pressure;
    };


  template <int dim>
  SeaIceRheologyProblem<dim>::Postprocessor::
  Postprocessor (const unsigned int partition,
                 const double       minimal_pressure)
    :
    partition (partition),
    minimal_pressure (minimal_pressure)
  {}

  template <int dim>
  std::vector<std::string>
  SeaIceRheologyProblem<dim>::Postprocessor::get_names() const
  {
    std::vector<std::string> solution_names (dim, "velocity");
    solution_names.push_back ("p");
    solution_names.push_back ("h");
    solution_names.push_back ("c");

    return solution_names;
  }

  template <int dim>
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
  SeaIceRheologyProblem<dim>::Postprocessor::
  get_data_component_interpretation () const
  {
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
    interpretation (dim,
                    DataComponentInterpretation::component_is_part_of_vector);

    interpretation.push_back (DataComponentInterpretation::component_is_scalar);
    interpretation.push_back (DataComponentInterpretation::component_is_scalar);
    interpretation.push_back (DataComponentInterpretation::component_is_scalar);

    return interpretation;
  }

  template <int dim>
  UpdateFlags
  SeaIceRheologyProblem<dim>::Postprocessor::get_needed_update_flags() const
  {
    return update_values | update_gradients | update_q_points;
  }


  template <int dim>
   void
   SeaIceRheologyProblem<dim>::Postprocessor::
   compute_derived_quantities_vector (const std::vector<Vector<double> >              &uh,
                                      const std::vector<std::vector<Tensor<1,dim> > > &duh,
                                      const std::vector<std::vector<Tensor<2,dim> > > &/*dduh*/,
                                      const std::vector<Point<dim> >                  &/*normals*/,
                                      const std::vector<Point<dim> >                  &/*evaluation_points*/,
                                      std::vector<Vector<double> >                    &computed_quantities) const
   {
	  return;
   }

  /////Assume that building stokes matrix every time.
  template <int dim>
  void SeaIceRheologyProblem<dim>::assemble_stokes_system_parallel (const bool &ifcor)
  {
    pcout << "   Assembling Stokes Matrix in Parallel..." << std::flush;
    computing_timer.enter_section ("   Assemble Stokes matrices");

    if (rebuild_stokes_matrix == true)
    	stokes_matrix = 0;
    stokes_rhs = 0;

    const QGauss<dim> quadrature_formula(stokes_degree+2);

    typedef
    FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>
    CellFilter;

    WorkStream::
        run (CellFilter (IteratorFilters::LocallyOwnedCell(),
                         stokes_dof_handler.begin_active()),
             CellFilter (IteratorFilters::LocallyOwnedCell(),
                         stokes_dof_handler.end()),
             std_cxx11::bind (&SeaIceRheologyProblem<dim>::
                              local_assemble_stokes_matrix,
                              this,
                              std_cxx11::_1,
                              std_cxx11::_2,
                              std_cxx11::_3,
							  ifcor),
             std_cxx11::bind (&SeaIceRheologyProblem<dim>::
                              copy_local_to_global_stokes_matrix,
                              this,
                              std_cxx11::_1),
             Assembly::Scratch::
             StokesSystem<dim>  (stokes_fe, mapping, quadrature_formula,
            		            (update_values    |
            		             update_quadrature_points  |
            		             update_JxW_values |
            		             update_gradients
            		             ),
								 thick_fe,
								 update_values,
								 conc_fe,
								 update_values,
								 p_fe,
								 update_values |
								 update_gradients),
             Assembly::CopyData::
             StokesSystem<dim> (stokes_fe));

    if (rebuild_stokes_matrix == true)
      stokes_matrix.compress(VectorOperation::add);
    stokes_rhs.compress(VectorOperation::add);

    pcout<<stokes_matrix.l1_norm()<<" norm matrix rhs  "<<stokes_rhs.l2_norm()<<std::endl;

    rebuild_stokes_matrix = true; //IMPORTANT, IT HAS TO BE TRUE, SINCE THE VISCOSITY ETA IS VARYING.
    computing_timer.exit_section();
  }

  template <int dim>
  void SeaIceRheologyProblem<dim>::assemble_p_system_parallel (const bool &ifcor)
  {
    pcout << "   Assembling Pressure Matrix in Parallel..." << std::flush;
    computing_timer.enter_section ("   Assemble Pressure matrices");

    if (rebuild_p_matrix == true)
    	p_matrix = 0;
    p_rhs = 0;

    const QGauss<dim> quadrature_formula(p_degree+2);

    typedef
    FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>
    CellFilter;

    WorkStream::
        run (CellFilter (IteratorFilters::LocallyOwnedCell(),
                         p_dof_handler.begin_active()),
             CellFilter (IteratorFilters::LocallyOwnedCell(),
                         p_dof_handler.end()),
             std_cxx11::bind (&SeaIceRheologyProblem<dim>::
                              local_assemble_p_matrix,
                              this,
                              std_cxx11::_1,
                              std_cxx11::_2,
                              std_cxx11::_3,
							  ifcor),
             std_cxx11::bind (&SeaIceRheologyProblem<dim>::
                              copy_local_to_global_p_matrix,
                              this,
                              std_cxx11::_1),
             Assembly::Scratch::
             PSystem<dim>      (p_fe, mapping, quadrature_formula,
                                update_values    |
                                update_JxW_values,
                                stokes_fe,
                                update_gradients,
    							thick_fe,
    							update_values,
    							conc_fe,
    							update_values),
             Assembly::CopyData::
             PSystem<dim> (p_fe));

    if (rebuild_p_matrix == true)
    	p_matrix.compress(VectorOperation::add);

    p_rhs.compress(VectorOperation::add);

    rebuild_p_matrix = false;

    pcout << std::endl;
    computing_timer.exit_section();

  }

  template <int dim>
    void SeaIceRheologyProblem<dim>::assemble_sigma_system_parallel ()
    {
      pcout << "   Assembling Sigma Matrix in Parallel..." << std::flush;
      computing_timer.enter_section ("   Assemble Sigma matrices");

      if (rebuild_sig_matrix == true)
      	sig_matrix = 0;
      sig_rhs = 0;

      const QGauss<dim> quadrature_formula(sig_degree+2);

      typedef
      FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>
      CellFilter;


      WorkStream::
          run (CellFilter (IteratorFilters::LocallyOwnedCell(),
                           sig_dof_handler.begin_active()),
               CellFilter (IteratorFilters::LocallyOwnedCell(),
                           sig_dof_handler.end()),
               std_cxx11::bind (&SeaIceRheologyProblem<dim>::
                                local_assemble_sig_matrix,
                                this,
                                std_cxx11::_1,
                                std_cxx11::_2,
                                std_cxx11::_3),
               std_cxx11::bind (&SeaIceRheologyProblem<dim>::
                                copy_local_to_global_sig_matrix,
                                this,
                                std_cxx11::_1),
               Assembly::Scratch::
               SigSystem<dim>    (sig_fe, mapping, quadrature_formula,
                                  update_values    |
                                  update_JxW_values,
                                  stokes_fe,
                                  update_gradients,
      							  p_fe,
      							  update_values),
               Assembly::CopyData::
               SigSystem<dim> (sig_fe));

      if (rebuild_sig_matrix == true)
      	sig_matrix.compress(VectorOperation::add);

      sig_rhs.compress(VectorOperation::add);

      rebuild_sig_matrix = false;

      pcout << std::endl;
      computing_timer.exit_section();

    }

  template <int dim>
  void SeaIceRheologyProblem<dim>::assemble_sigma_diag_system_parallel ()
  {
    pcout << "   Assembling Sigma Diagnal Matrix in Parallel..." << std::flush;
    computing_timer.enter_section ("   Assemble Sigma Diagnal matrices");

    if (rebuild_sig_diag_matrix == true)
    	sig_diag_matrix = 0;
    sig_diag_rhs = 0;

    const QGauss<dim> quadrature_formula(sig_diag_degree+2);

    typedef
    FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>
    CellFilter;

    WorkStream::
        run (CellFilter (IteratorFilters::LocallyOwnedCell(),
                         sig_diag_dof_handler.begin_active()),
             CellFilter (IteratorFilters::LocallyOwnedCell(),
                         sig_diag_dof_handler.end()),
             std_cxx11::bind (&SeaIceRheologyProblem<dim>::
                              local_assemble_sig_diag_matrix,
                              this,
                              std_cxx11::_1,
                              std_cxx11::_2,
                              std_cxx11::_3),
             std_cxx11::bind (&SeaIceRheologyProblem<dim>::
                              copy_local_to_global_sig_diag_matrix,
                              this,
                              std_cxx11::_1),
             Assembly::Scratch::
             SigDiagSystem<dim>(sig_diag_fe, mapping, quadrature_formula,
                                update_values    |
                                update_JxW_values,
                                stokes_fe,
                                update_gradients,
    							p_fe,
    							update_values),
             Assembly::CopyData::
             SigDiagSystem<dim> (sig_diag_fe));

    if (rebuild_sig_diag_matrix == true)
    	sig_diag_matrix.compress(VectorOperation::add);

    sig_diag_rhs.compress(VectorOperation::add);

    rebuild_sig_diag_matrix = false;

    pcout << std::endl;
    computing_timer.exit_section();

  }

  template <int dim>
  void SeaIceRheologyProblem<dim>::assemble_thick_conc_system_parallel ()
  {
    pcout << "   Assembling Thick and Conc Matrix in Parallel..." << std::flush;
    computing_timer.enter_section ("   Assemble Thick and Conc matrices");

    if (rebuild_thick_matrix == true)
    {
    	thick_matrix=0;
    	conc_matrix =0;
    }

    thick_rhs=0;
    conc_rhs =0;

    const QGauss<dim> quadrature_formula(thick_degree+2);

    typedef
    FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>
    CellFilter;

    WorkStream::
        run (CellFilter (IteratorFilters::LocallyOwnedCell(),
                         thick_dof_handler.begin_active()),
             CellFilter (IteratorFilters::LocallyOwnedCell(),
                         thick_dof_handler.end()),
             std_cxx11::bind (&SeaIceRheologyProblem<dim>::
                              local_assemble_thick_conc_matrix,
                              this,
                              std_cxx11::_1,
                              std_cxx11::_2,
                              std_cxx11::_3),
             std_cxx11::bind (&SeaIceRheologyProblem<dim>::
                              copy_local_to_global_thick_conc_matrix,
                              this,
                              std_cxx11::_1),
             Assembly::Scratch::
             ThickConcSystem<dim> (thick_fe, mapping, quadrature_formula,
                                   update_values    |
							   	   update_gradients |
                                   update_JxW_values,
                                   conc_fe,
                                   update_values,
    							   stokes_fe,
								   update_values |
    							   update_gradients),
             Assembly::CopyData::
             ThickConcSystem<dim> (thick_fe));

    if (rebuild_thick_matrix == true)
    	{
    		thick_matrix.compress(VectorOperation::add);
    		conc_matrix.compress(VectorOperation::add);
    	}

    thick_rhs.compress(VectorOperation::add);
    conc_rhs.compress(VectorOperation::add);


    rebuild_thick_matrix = true;

    pcout << std::endl;
    computing_timer.exit_section();
  }


  template <int dim>
  void SeaIceRheologyProblem<dim>::assemble_stokes_system (const bool &ifcor)
  {
    std::cout << "   Assembling..." << std::flush;

    if (rebuild_stokes_matrix == true)
    {
    	stokes_matrix = 0;
    }

    stokes_rhs = 0;

    const QGauss<dim> quadrature_formula (stokes_degree+2);
    FEValues<dim>     stokes_fe_values (stokes_fe, quadrature_formula,
                                        update_values    |
                                        update_gradients    |
                                        update_quadrature_points  |
                                        update_JxW_values);

    FEValues<dim>     thick_fe_values (thick_fe, quadrature_formula,
                                       update_values);

    FEValues<dim>     p_fe_values (p_fe, quadrature_formula,
    		                       update_values |
                                   update_gradients);

    const unsigned int   dofs_per_cell   = stokes_fe.dofs_per_cell;
    const unsigned int   n_q_points      = quadrature_formula.size();

    FullMatrix<double>   local_matrix (dofs_per_cell, dofs_per_cell);
    Vector<double>       local_rhs    (dofs_per_cell);


    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

    const RightHandSide<dim> right_hand_side;

    std::vector<Vector<double>> rhs_values (n_q_points,
                                                  Vector<double>(dim));
    std::vector<Tensor<1,dim> > old_velocity_values (n_q_points);
    std::vector<Tensor<2,dim> > velocity_gradients (n_q_points);
	std::vector<double>         value_thick(n_q_points);
	std::vector<double>         old_pressure_values(n_q_points);
    std::vector<Tensor<1,dim> > old_pressure_gradients (n_q_points);



    std::vector<Tensor<1,dim> >          phi_u       (dofs_per_cell);
    std::vector<SymmetricTensor<2,dim> > grads_phi_u (dofs_per_cell);
    std::vector<double>                  div_phi_u   (dofs_per_cell);


    const FEValuesExtractors::Vector velocities (0);
    const FEValuesExtractors::Scalar pressure (dim);


    typename DoFHandler<dim>::active_cell_iterator
    cell = stokes_dof_handler.begin_active(),
    endc = stokes_dof_handler.end();



    typename DoFHandler<dim>::active_cell_iterator
    thick_cell = thick_dof_handler.begin_active();

    typename DoFHandler<dim>::active_cell_iterator
    p_cell = p_dof_handler.begin_active();
    std::vector<double> coef (n_q_points);

    for (; cell!=endc; ++cell, ++thick_cell, ++p_cell)
      {
        stokes_fe_values.reinit (cell);
        thick_fe_values.reinit (thick_cell);
        p_fe_values.reinit (p_cell);


        local_matrix = 0;
        local_rhs = 0;

        right_hand_side.vector_value_list(stokes_fe_values.get_quadrature_points(),
                                          rhs_values);
        stokes_fe_values[velocities].get_function_values (old_stokes_solution,
                                                          old_velocity_values);

        stokes_fe_values[velocities].get_function_gradients (ifcor==true ? cor_stokes_solution : old_stokes_solution,
                                                             velocity_gradients);

        p_fe_values.get_function_gradients (p_solution,
                                            old_pressure_gradients);

        p_fe_values.get_function_values (p_solution,
                                         old_pressure_values);

        thick_fe_values.get_function_values (thick_solution,
                                             value_thick);


        for (unsigned int q=0; q<n_q_points; ++q)
        {
        	coef [q] =   EquationData::Re
	                   / EquationData::dt
	                   * value_thick[q];

        	for (unsigned int i=0; i<dim; ++i)
        	{
        		rhs_values[q](i) +=  old_velocity_values[q][i] * coef[q]
							       - old_pressure_gradients[q][i];
        	}
        }

        for (unsigned int q=0; q<n_q_points; ++q)
          {
    		double eta  = compute_viscosity (velocity_gradients[q], old_pressure_values[q]);

            for (unsigned int k=0; k<dofs_per_cell; ++k)
              {
                phi_u[k] = stokes_fe_values[velocities].value (k,q);
                if (rebuild_stokes_matrix)
                  {
                    grads_phi_u[k] = stokes_fe_values[velocities].symmetric_gradient(k,q);
                    div_phi_u[k]   = stokes_fe_values[velocities].divergence (k, q);
                  }
              }


            if (rebuild_stokes_matrix)
              for (unsigned int i=0; i<dofs_per_cell; ++i)
                for (unsigned int j=0; j<dofs_per_cell; ++j)
                  local_matrix(i,j) += (coef[q]*phi_u[i]*phi_u[j]
								       +eta * 2. *
						               (grads_phi_u[i] * grads_phi_u[j]))
                                       * stokes_fe_values.JxW(q);



            for (unsigned int i=0; i<dofs_per_cell; ++i)
            {
            	const unsigned int component_i =
            			stokes_fe.system_to_component_index(i).first;
            	local_rhs(i) += stokes_fe_values.shape_value(i,q) *
            			rhs_values[q](component_i) *
						stokes_fe_values.JxW(q);
            }
          }

        cell->get_dof_indices (local_dof_indices);

        if (rebuild_stokes_matrix == true)
          stokes_constraints.distribute_local_to_global (local_matrix,
                                                         local_rhs,
                                                         local_dof_indices,
                                                         stokes_matrix,
                                                         stokes_rhs);
        else
        {
        	for (unsigned int q=0; q<n_q_points; ++q)
        	{
        		for (unsigned int k=0; k<dofs_per_cell; ++k)
        		{
        			phi_u[k] = stokes_fe_values[velocities].value (k,q);
        		}
        		for (unsigned int i=0; i<dofs_per_cell; ++i)
        		{
        			if (stokes_constraints.is_inhomogeneously_constrained(local_dof_indices[i]))
        				for (unsigned int j=0; j<dofs_per_cell; ++j)
        				{
        					local_matrix(j,i) += (coef[q]*phi_u[i]*phi_u[j])
											     * stokes_fe_values.JxW(q);
        				}
        		}
        	}
        	stokes_constraints.distribute_local_to_global (local_rhs,
                                                           local_dof_indices,
                                                           stokes_rhs,
														   local_matrix);
        }
      }

    rebuild_stokes_matrix = true;

    std::cout << std::endl;
  }


  template <int dim>
    void SeaIceRheologyProblem<dim>::assemble_p_system ()
    {
	  if (rebuild_p_matrix == true)
		  p_matrix=0;

	  	  p_rhs=0;
		  const QGauss<dim> quadrature_formula(p_degree+2);
		  FEValues<dim>     p_fe_values    (p_fe, quadrature_formula,
		  			                        update_values |
		  								    update_JxW_values);
		  FEValues<dim>     thick_fe_values (thick_fe, quadrature_formula,
				                             update_values);
		  FEValues<dim>     conc_fe_values  (conc_fe, quadrature_formula,
				                             update_values);
		  FEValues<dim>     stokes_fe_values (stokes_fe, quadrature_formula,
		                                      update_gradients);

		  const unsigned int   dofs_per_cell   = p_fe.dofs_per_cell;
		  const unsigned int   n_q_points      = quadrature_formula.size();

		  FullMatrix<double>   local_matrix (dofs_per_cell, dofs_per_cell);
		  Vector<double>       local_rhs    (dofs_per_cell);

		  std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

		  std::vector<double>         value_thick(n_q_points),
				                      value_conc (n_q_points);
     	  std::vector<Tensor<2,dim> > old_velocity_gradients (n_q_points);

     	  const FEValuesExtractors::Vector velocities (0);


		  typename DoFHandler<dim>::active_cell_iterator
		  cell = p_dof_handler.begin_active(),
		  endc = p_dof_handler.end();
		  typename DoFHandler<dim>::active_cell_iterator
		  thick_cell = thick_dof_handler.begin_active();
		  typename DoFHandler<dim>::active_cell_iterator
		  conc_cell = conc_dof_handler.begin_active();
		  typename DoFHandler<dim>::active_cell_iterator
		  stokes_cell = stokes_dof_handler.begin_active();

		  double coef = std::pow(IceModel::alpha,2) - 1;

		  for (; cell!=endc; ++cell, ++thick_cell, ++conc_cell, ++stokes_cell)
		  	  {
		        p_fe_values.reinit (cell);
		        thick_fe_values.reinit (thick_cell);
		        conc_fe_values.reinit (conc_cell);
		        stokes_fe_values.reinit (stokes_cell);



		        local_matrix = 0;
		        local_rhs = 0;

		        thick_fe_values.get_function_values (thick_solution,
		                                             value_thick);
		        conc_fe_values.get_function_values  (conc_solution,
		                                             value_conc);
		        stokes_fe_values[velocities].get_function_gradients  (old_stokes_solution,
		                                                              old_velocity_gradients);

		        for (unsigned int q=0; q<n_q_points; ++q)
		          {

		        	if (rebuild_p_matrix == true)
		        		for (unsigned int i=0; i<dofs_per_cell; ++i)
		        		  for (unsigned int j=0; j<dofs_per_cell; ++j)
		        			  {
		        			  	  local_matrix(i,j) +=  p_fe_values.shape_value (i, q) *
							                            p_fe_values.shape_value (j, q) *
		        		                                p_fe_values.JxW (q);
		        			  }

		        	for (unsigned int i=0; i<dofs_per_cell; ++i)
		        	{
		        		double temp = p_fe_values.shape_value(i,q)
						             *p_fe_values.JxW(q);
		        		double pice = value_thick[q]*std::exp(-IceModel::k*(1-value_conc[q]));
		        		double eta  = compute_viscosity(old_velocity_gradients[q], pice);
		        		local_rhs(i) += (IceModel::alpha*pice - eta*coef*(old_velocity_gradients[q][0][0]
																	     +old_velocity_gradients[q][1][1]))
									    *temp;
		        	}
		          }

		        cell->get_dof_indices (local_dof_indices);

		        if (rebuild_p_matrix == true)
		        	p_constraints.distribute_local_to_global (local_matrix,
		                                                      local_rhs,
		                                                      local_dof_indices,
		                                                      p_matrix,
		                                                      p_rhs);
		        else
		        	p_constraints.distribute_local_to_global (local_rhs,
		                                                      local_dof_indices,
		                                                      p_rhs);

		  	  }

		    rebuild_p_matrix = false;
		    std::cout << std::endl;

    }








  template <int dim>
  void SeaIceRheologyProblem<dim>::assemble_sigma_system ()
  {
	  if (rebuild_sig_matrix == true)
		  sig_matrix=0;

	  sig_rhs=0;

	  const QGauss<dim> quadrature_formula(sig_degree+2);

	  FEValues<dim>     stokes_fe_values (stokes_fe, quadrature_formula,
										  update_gradients);

	  FEValues<dim>     p_fe_values   (p_fe, quadrature_formula,
	  			                       update_values);

	  FEValues<dim>     sig_fe_values    (sig_fe, quadrature_formula,
	  			                          update_values |
	  								      update_JxW_values);

	  const unsigned int   dofs_per_cell   = sig_fe.dofs_per_cell;
	  const unsigned int   n_q_points      = quadrature_formula.size();

	  FullMatrix<double>   local_matrix (dofs_per_cell, dofs_per_cell);
	  Vector<double>       local_rhs (dofs_per_cell);

	  std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

	  std::vector<Tensor<2,dim> > grads_velocity (n_q_points);

	  std::vector<double>         value_pressure (n_q_points);

	  std::vector<double>    phi_sig11(dofs_per_cell),
			                 phi_sig22(dofs_per_cell),
							 phi_sig_diag_sum (dofs_per_cell);

	  std::vector<double>         old_sig11_value(n_q_points),
			                      old_sig22_value(n_q_points);

	  double al2   =  std::pow(IceModel::alpha,2);
	  double coef  =  EquationData::chi*(subcycle_stokes_sig == true ? EquationData::dte_inv : 1/EquationData::dt);
	  double coef2 =  (1-al2)/4;
	  double coef3 =  al2 / 2;
	  double coef4 = -IceModel::alpha/2;

	  const FEValuesExtractors::Vector velocities (0);

	  const FEValuesExtractors::Scalar sig11 (isig11),
			                           sig22 (isig22);



	  typename DoFHandler<dim>::active_cell_iterator
	  cell = sig_dof_handler.begin_active(),
	  endc = sig_dof_handler.end();
	  typename DoFHandler<dim>::active_cell_iterator
	  stokes_cell = stokes_dof_handler.begin_active();
	  typename DoFHandler<dim>::active_cell_iterator
	  p_cell = p_dof_handler.begin_active();



	  for (; cell!=endc; ++cell, ++stokes_cell, ++p_cell)
	  	  {
	        sig_fe_values.reinit (cell);
	        stokes_fe_values.reinit (stokes_cell);
            p_fe_values.reinit (p_cell);

	        local_matrix = 0;
	        local_rhs = 0;

            stokes_fe_values[velocities].get_function_gradients
			((update_viscosity_every_subcycle == true) ? stokes_solution : sub_old_stokes_solution,
            grads_velocity);


	        p_fe_values.get_function_values      (p_solution,
	                                              value_pressure);

	        sig_fe_values[sig11].get_function_values (old_sig_solution,
	        		                                  old_sig11_value);
	        sig_fe_values[sig22].get_function_values (old_sig_solution,
	        		                                  old_sig22_value);


	        for (unsigned int q=0; q<n_q_points; ++q)
	          {
        		double eta  = compute_viscosity (grads_velocity[q], value_pressure[q]);

        		for (unsigned int i=0; i<dofs_per_cell; ++i)
        		{
                    phi_sig11[i] = sig_fe_values[sig11].value (i,q);
                    phi_sig22[i] = sig_fe_values[sig22].value (i,q);
                    phi_sig_diag_sum[i] = phi_sig11 [i] + phi_sig22 [i];
        		}

        		if (rebuild_sig_matrix == true)
	        		for (unsigned int i=0; i<dofs_per_cell; ++i)
	        		  for (unsigned int j=0; j<dofs_per_cell; ++j)
	        		    local_matrix(i,j) +=  (
	        		    		              (phi_sig11 [i] * phi_sig11 [j]
	        		                         + phi_sig22 [i] * phi_sig22 [j]
											   )*(coef + coef3)
											 + (phi_sig_diag_sum [i])
											 * (phi_sig_diag_sum [j])*coef2
											   )
											  *
	        		                          sig_fe_values.JxW (q);



	        	for (unsigned int i=0; i<dofs_per_cell; ++i)
	        	{

	        		local_rhs(i) += (coef4 * value_pressure[q] * phi_sig_diag_sum [i]
	        				         +al2 * eta *
									 (
									  phi_sig11 [i] * grads_velocity[q][0][0] +
									  phi_sig22 [i] * grads_velocity[q][1][1]
									  //phi_sig12 [i] * (grads_velocity[q][0][1]
									  //				 + grads_velocity[q][1][0])/2
									 )
									 +coef *
									 (
									  phi_sig11 [i] * old_sig11_value[q] +
									  phi_sig22 [i] * old_sig22_value[q]
									  //phi_sig12 [i] * old_sig12_value[q]
									 )
									 )*sig_fe_values.JxW (q);
	        	}
	          }
	        cell->get_dof_indices (local_dof_indices);

	        if (rebuild_sig_matrix == true)
	          sig_constraints.distribute_local_to_global (local_matrix,
	                                                      local_rhs,
	                                                      local_dof_indices,
	                                                      sig_matrix,
	                                                      sig_rhs);
	        else
	          sig_constraints.distribute_local_to_global (local_rhs,
	                                                      local_dof_indices,
	                                                      sig_rhs);

	  	  }

	    rebuild_sig_matrix = false;
	    std::cout << std::endl;
  }

  template <int dim>
  void SeaIceRheologyProblem<dim>::assemble_sigma_diag_system ()
  {
	  if (rebuild_sig_diag_matrix == true)
		  sig_diag_matrix=0;

	  sig_diag_rhs=0;

	  const QGauss<dim> quadrature_formula(sig_diag_degree+2);

	  FEValues<dim>     stokes_fe_values (stokes_fe, quadrature_formula,
										  update_gradients);

	  FEValues<dim>     p_fe_values   (p_fe, quadrature_formula,
	  			                       update_values);

	  FEValues<dim>     sig_diag_fe_values    (sig_diag_fe, quadrature_formula,
	  			                          update_values |
	  								      update_JxW_values);

	  const unsigned int   dofs_per_cell   = sig_diag_fe.dofs_per_cell;
	  const unsigned int   n_q_points      = quadrature_formula.size();

	  FullMatrix<double>   local_matrix (dofs_per_cell, dofs_per_cell);
	  Vector<double>       local_rhs (dofs_per_cell);

	  std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

	  std::vector<Tensor<2,dim> > grads_velocity (n_q_points);

	  std::vector<double>         value_pressure (n_q_points);

	  std::vector<double>         old_sig_diag_value(n_q_points);

	  double al2   =  std::pow(IceModel::alpha,2);
	  double coef  =  EquationData::chi*(subcycle_stokes_sig == true ? EquationData::dte_inv : 1/EquationData::dt);
	  double coef2 =  (1-al2)/4;
	  double coef3 =  al2 / 2;
	  double coef4 = -IceModel::alpha/2;

	  const FEValuesExtractors::Vector velocities (0);





	  typename DoFHandler<dim>::active_cell_iterator
	  cell = sig_diag_dof_handler.begin_active(),
	  endc = sig_diag_dof_handler.end();
	  typename DoFHandler<dim>::active_cell_iterator
	  stokes_cell = stokes_dof_handler.begin_active();
	  typename DoFHandler<dim>::active_cell_iterator
	  p_cell = p_dof_handler.begin_active();



	  for (; cell!=endc; ++cell, ++stokes_cell, ++p_cell)
	  	  {
	        sig_diag_fe_values.reinit (cell);
	        stokes_fe_values.reinit (stokes_cell);
            p_fe_values.reinit (p_cell);

	        local_matrix = 0;
	        local_rhs = 0;

            stokes_fe_values[velocities].get_function_gradients
			((update_viscosity_every_subcycle == true) ? stokes_solution : sub_old_stokes_solution,
            grads_velocity);


	        p_fe_values.get_function_values      (p_solution,
	                                              value_pressure);

	        sig_diag_fe_values.get_function_values (old_sig_diag_solution,
          		                                    old_sig_diag_value);



	        for (unsigned int q=0; q<n_q_points; ++q)
	          {
        		double eta  = compute_viscosity (grads_velocity[q], value_pressure[q]);

        		if (rebuild_sig_diag_matrix == true)
	        		for (unsigned int i=0; i<dofs_per_cell; ++i)
	        		  for (unsigned int j=0; j<dofs_per_cell; ++j)
	        		    local_matrix(i,j) +=
											 sig_diag_fe_values.shape_value (i,q) * sig_diag_fe_values.shape_value (j,q)
											 *(coef + coef3)
											 *sig_diag_fe_values.JxW (q);



	        	for (unsigned int i=0; i<dofs_per_cell; ++i)
	        	{

	        		local_rhs(i) += (
	        				         al2 * eta *
									 (
									  sig_diag_fe_values.shape_value (i,q) * (grads_velocity[q][0][1]
									  			                         	 + grads_velocity[q][1][0])/2
									 )
									 +coef *
									 (
									  sig_diag_fe_values.shape_value (i,q) * old_sig_diag_value[q]
									 )
									 )*sig_diag_fe_values.JxW (q);
	        	}
	          }
	        cell->get_dof_indices (local_dof_indices);

	        if (rebuild_sig_diag_matrix == true)
	          sig_diag_constraints.distribute_local_to_global (local_matrix,
	                                                           local_rhs,
	                                                           local_dof_indices,
	                                                           sig_diag_matrix,
	                                                           sig_diag_rhs);
	        else
	          sig_diag_constraints.distribute_local_to_global (local_rhs,
	                                                           local_dof_indices,
	                                                           sig_diag_rhs);

	  	  }

	    rebuild_sig_diag_matrix = false;
	    std::cout << std::endl;
  }
  template <int dim>
  void SeaIceRheologyProblem<dim>::assemble_thick_conc_system ()
  {
	  if (rebuild_thick_matrix == true)
	  {
		  thick_matrix=0;
		  conc_matrix =0;
	  }

	  thick_rhs=0;
	  conc_rhs =0;

	  const QGauss<dim> quadrature_formula(thick_degree+2);

	  FEValues<dim>     stokes_fe_values (stokes_fe, quadrature_formula,
			                              update_values |
										  update_gradients);

	  FEValues<dim>     thick_fe_values    (thick_fe, quadrature_formula,
			                              update_gradients |
	  			                          update_values |
	  								      update_JxW_values);

	  FEValues<dim>     conc_fe_values    (conc_fe, quadrature_formula,
	  			                            update_values);

	  const unsigned int   dofs_per_cell   = thick_fe.dofs_per_cell;
	  const unsigned int   n_q_points      = quadrature_formula.size();

	  FullMatrix<double>   local_matrix (dofs_per_cell, dofs_per_cell);
	  Vector<double>       local_rhs    (dofs_per_cell),
			          conc_local_rhs    (dofs_per_cell);
	  std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

	  std::vector<Tensor<2,dim> > grads_velocity (n_q_points);
	  std::vector<Tensor<1,dim> > velocity (n_q_points);
	  std::vector<double>         old_thick_values (n_q_points);
	  std::vector<double>         old_conc_values (n_q_points);


	  const FEValuesExtractors::Vector velocities (0);

	  typename DoFHandler<dim>::active_cell_iterator
	  cell = thick_dof_handler.begin_active(),
	  endc = thick_dof_handler.end();
	  typename DoFHandler<dim>::active_cell_iterator
	  stokes_cell = stokes_dof_handler.begin_active();
	  typename DoFHandler<dim>::active_cell_iterator
	  conc_cell = conc_dof_handler.begin_active();


	  for (; cell!=endc; ++cell, ++stokes_cell, ++conc_cell)
	  	  {
	        stokes_fe_values.reinit (stokes_cell);
	        thick_fe_values.reinit (cell);
	        conc_fe_values.reinit (conc_cell);


	        local_matrix = 0;
	        local_rhs = 0;
            conc_local_rhs = 0;

	        stokes_fe_values[velocities].get_function_gradients (stokes_solution,
	                                                             grads_velocity);
	        stokes_fe_values[velocities].get_function_values    (stokes_solution,
	                                                             velocity);
	        thick_fe_values.get_function_values (old_thick_solution,
	        		                             old_thick_values);

	        conc_fe_values.get_function_values  (old_conc_solution,
	        		                             old_conc_values);

	        for (unsigned int q=0; q<n_q_points; ++q)
	          {

	        	if (rebuild_thick_matrix == true)
	        		for (unsigned int i=0; i<dofs_per_cell; ++i)
	        		  for (unsigned int j=0; j<dofs_per_cell; ++j)
	        		  {
	        			  local_matrix(i,j) +=  thick_fe_values.shape_value(i,q) *
	        					               (thick_fe_values.shape_grad(j,q)*velocity[q] +
	        					                thick_fe_values.shape_value(j,q)*(1/EquationData::dt +
	        					                grads_velocity[q][0][0] + grads_velocity[q][1][1])) *
											    thick_fe_values.JxW(q);
	        		  }


	        	for (unsigned int i=0; i<dofs_per_cell; ++i)
	        	{
	        		double temp = thick_fe_values.shape_value(i,q) *
    				              1/EquationData::dt*
							      thick_fe_values.JxW(q);
	        		local_rhs(i) += temp * old_thick_values[q];
	           conc_local_rhs(i) += temp * old_conc_values [q];

	        	}

	          }
	        cell->get_dof_indices (local_dof_indices);

	        if (rebuild_thick_matrix == true)
	        	{
	        		thick_constraints.distribute_local_to_global (local_matrix,
	                                                              local_rhs,
	                                                              local_dof_indices,
	                                                              thick_matrix,
	                                                              thick_rhs);

	        		conc_constraints.distribute_local_to_global  (local_matrix,
	        			                                          conc_local_rhs,
	        			                                          local_dof_indices,
	        			                                          conc_matrix,
	        			                                          conc_rhs);
	        	}
	        else
	          {
	        	    thick_constraints.distribute_local_to_global (local_rhs,
	                                                              local_dof_indices,
	                                                              thick_rhs);

	        		conc_constraints.distribute_local_to_global  (conc_local_rhs,
	                                                              local_dof_indices,
	                                                              conc_rhs);
	          }
	  	  }

	    rebuild_thick_matrix = true; // since the local matrix contains the velocity information.
	    std::cout << std::endl;
  }






  template <int dim>
  void SeaIceRheologyProblem<dim>::solve_sigma ()
  {
    std::cout << "   Solving Sigma..." << std::endl;
    std::cout << "   Assembling For Sigma..." << std::endl << std::flush;
    assemble_sigma_system ();
    assemble_sigma_diag_system ();
    {
      deallog.push("DirectKLU");
      Solver_::solver_klu.solve (sig_matrix, sig_solution, sig_rhs);
      sig_constraints.distribute (sig_solution);
      Solver_::solver_klu.solve (sig_diag_matrix, sig_diag_solution, sig_diag_rhs);
      sig_diag_constraints.distribute (sig_diag_solution);
      deallog.pop();
    }
  }

  template <int dim>
  void SeaIceRheologyProblem<dim>::solve_sigma_parallel ()
  {
	  computing_timer.enter_section ("   Solve Sigma Principal system");
	  deallog.push("CG (Trilinos)");
	  pcout << "   Solving Sigma Principal and Diagnal in Parallel..." << std::endl;

	  SolverControl solver_control (sig_matrix.m(),
			                        1e-12*sig_rhs.l2_norm());
	  SolverCG<TrilinosWrappers::MPI::Vector>   cg (solver_control);

	  TrilinosWrappers::PreconditionIC preconditioner;
	  preconditioner.initialize (sig_matrix);

	  TrilinosWrappers::MPI::Vector
	  distributed_sig_solution (sig_rhs);
	  distributed_sig_solution = sig_solution;

	  cg.solve (sig_matrix, distributed_sig_solution,
			    sig_rhs, preconditioner);

	  sig_constraints.distribute (distributed_sig_solution);
	  sig_solution = distributed_sig_solution;

	  pcout << "   "
			  << solver_control.last_step()
			  << " CG iterations for Sigma Principal" << std::endl;
	  computing_timer.exit_section();

	  computing_timer.enter_section ("   Solve Sigma Diagnal system");
	  SolverControl solver_control2 (sig_diag_matrix.m(),
			                         1e-12*sig_diag_rhs.l2_norm());
	  SolverCG<TrilinosWrappers::MPI::Vector>   cg2 (solver_control2);

	  TrilinosWrappers::PreconditionIC preconditioner2;
	  preconditioner2.initialize (sig_diag_matrix);

	  TrilinosWrappers::MPI::Vector
	  distributed_sig_solution2  (sig_diag_rhs);
	  distributed_sig_solution2 = sig_diag_solution;

	  cg.solve (sig_diag_matrix, distributed_sig_solution2,
			    sig_diag_rhs, preconditioner2);

	  sig_diag_constraints.distribute (distributed_sig_solution2);
	  sig_diag_solution = distributed_sig_solution2;


	  pcout   << "   "
			  << solver_control.last_step()
			  << " CG iterations for Sigma Diagnal" << std::endl;

	  deallog.pop();
	  computing_timer.exit_section();
  }

  template <int dim>
  void SeaIceRheologyProblem<dim>::solve_thick_conc ()
  {
    std::cout << "   Solving Thickness h..." << std::endl;
    std::cout << "   Assembling For h and conc..." << std::endl << std::flush;
    assemble_thick_conc_system ();
    {
    	deallog.push("DirectKLU");

    	Solver_::solver_klu.solve (thick_matrix, thick_solution, thick_rhs);
    	thick_constraints.distribute (thick_solution);

    	Solver_::solver_klu.solve (conc_matrix, conc_solution, conc_rhs);
    	conc_constraints.distribute (conc_solution);
        deallog.pop();
    }

  }



  template <int dim>
  void SeaIceRheologyProblem<dim>::solve_thick_conc_parallel ()
  {

	  computing_timer.enter_section ("   Solve h and c");
	  deallog.push("DirectKLU");
	  pcout << "   Solving Height and Concentration in Parallel..." << std::endl;

	  static TrilinosWrappers::SolverDirect::AdditionalData data (false, "Amesos_Klu");

	  SolverControl solver_control (thick_matrix.m(),
	                                1e-12*thick_rhs.l2_norm());

	  TrilinosWrappers::MPI::Vector tmp (thick_rhs);
	  TrilinosWrappers::SolverDirect solver (solver_control, data);

	  if(IceModel::iffix_thick == false)
	  {
		  tmp = thick_solution;
		  solver.solve (thick_matrix, tmp, thick_rhs);
		  thick_constraints.distribute (tmp);
		  thick_solution = tmp;
	  }

	  if(IceModel::iffix_conc == false)
	  {
		  tmp = conc_solution;
		  solver.solve (thick_matrix, tmp, conc_rhs);
		  conc_constraints.distribute (tmp);
		  conc_solution = tmp;
	  }

	  deallog.pop();
	  computing_timer.exit_section();
  }

  template <int dim>
  void SeaIceRheologyProblem<dim>::solve_stokes ()
  {
    std::cout << "   Solving Stokes U and V..." << std::endl;
    {
    	deallog.push("DirectKLU");
    	Solver_::solver_klu.solve (stokes_matrix, stokes_solution, stokes_rhs);
    	stokes_constraints.distribute (stokes_solution);
        deallog.pop();
    }

  }


  template <int dim>
  void SeaIceRheologyProblem<dim>::solve_stokes_parallel ()
  {
	  computing_timer.enter_section ("   Solve Stokes system");
	  pcout << "   Solving Stokes U and V in Parallel..." << std::endl;

	  if(SolverConfig::stokes.compare(std::string("klu"))==0)
	  {
		  deallog.push("DirectKLU");

		  static TrilinosWrappers::SolverDirect::AdditionalData data (false, "Amesos_Klu");
		  SolverControl solver_control (stokes_matrix.m(),
				  1e-12*stokes_rhs.l2_norm());

		  TrilinosWrappers::MPI::Vector tmp (stokes_rhs);
		  TrilinosWrappers::SolverDirect solver (solver_control, data);
		  tmp = stokes_solution;

		  solver.solve (stokes_matrix, tmp, stokes_rhs);
		  stokes_constraints.distribute (tmp);
		  stokes_solution = tmp;
	  }
	  else if(SolverConfig::stokes.compare(std::string("mumps"))==0)
	  {
		  deallog.push("DirectMUMPS");

		  static TrilinosWrappers::SolverDirect::AdditionalData data (false, "Amesos_Mumps");
		  SolverControl solver_control (stokes_matrix.m(),
				  1e-12*stokes_rhs.l2_norm());

		  TrilinosWrappers::MPI::Vector tmp (stokes_rhs);
		  TrilinosWrappers::SolverDirect solver (solver_control, data);
		  tmp = stokes_solution;

		  solver.solve (stokes_matrix, tmp, stokes_rhs);
		  stokes_constraints.distribute (tmp);
		  stokes_solution = tmp;
	  }
	  else if(SolverConfig::stokes.compare(std::string("gmres"))==0)
	  {
		  deallog.push("TrilinosGmres");
		  TrilinosWrappers::PreconditionILU preconditioner;
		  preconditioner.initialize (stokes_matrix);
		  TrilinosWrappers::MPI::Vector tmp (stokes_rhs);
	      tmp = stokes_solution;
	      SolverControl solver_control (stokes_matrix.m(),
	      				  1e-12*stokes_rhs.l2_norm());
          SolverGMRES<TrilinosWrappers::MPI::Vector>
          gmres (solver_control,
                 SolverGMRES<TrilinosWrappers::MPI::Vector >::AdditionalData(100));
          gmres.solve(stokes_matrix, tmp, stokes_rhs, preconditioner);
	      stokes_constraints.distribute (tmp);
	      stokes_solution = tmp;
	  }
	  else
	  {

	  }

	  deallog.pop();
	  computing_timer.exit_section();
  }


  template <int dim>
  void SeaIceRheologyProblem<dim>::solve_p ()
  {
    std::cout << "   Solving Pressure p..." << std::endl;
    std::cout << "   Assembling For p..." << std::endl << std::flush;
    assemble_p_system ();

    SolverControl solver_control (p_matrix.m(),
                                  1e-8*p_rhs.l2_norm());
    SolverCG<TrilinosWrappers::MPI::Vector> cg (solver_control);

    TrilinosWrappers::PreconditionIC preconditioner;
    preconditioner.initialize (p_matrix);

    cg.solve (p_matrix, p_solution,
              p_rhs, preconditioner);
    p_constraints.distribute (p_solution);

    std::cout << "   "
              << solver_control.last_step()
              << " CG iterations for p."
              << std::endl;
  }

  template <int dim>
  void SeaIceRheologyProblem<dim>::solve_p_parallel ()
  {
	  computing_timer.enter_section ("   Solve Pressure system");
	  deallog.push("CG (Trilinos)");
	  pcout << "   Solving Pressure in Parallel..." << std::endl;

	  SolverControl solver_control (p_matrix.m(),
			  1e-12*p_rhs.l2_norm());
	  SolverCG<TrilinosWrappers::MPI::Vector>   cg (solver_control);

	  TrilinosWrappers::PreconditionIC preconditioner;
	  preconditioner.initialize (p_matrix);

	  TrilinosWrappers::MPI::Vector
	  distributed_p_solution (p_rhs);
	  distributed_p_solution = p_solution;

	  cg.solve (p_matrix, distributed_p_solution,
			    p_rhs, preconditioner);

	  p_constraints.distribute (distributed_p_solution);
	  p_solution = distributed_p_solution;

	  pcout << "   "
			  << solver_control.last_step()
			  << " CG iterations for pressure" << std::endl;

	  deallog.pop();
	  computing_timer.exit_section();
  }

  template <int dim>
  void SeaIceRheologyProblem<dim>::output_results ()  const
  {
    if (timestep_number % EquationData::NtExp != 0)
      return;

    std::vector<std::string> stokes_names (dim, "velocity");
//    stokes_names.push_back ("p");
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
    stokes_component_interpretation
    (dim, DataComponentInterpretation::component_is_scalar);
    for (unsigned int i=0; i<dim; ++i)
      stokes_component_interpretation[i]
        = DataComponentInterpretation::component_is_part_of_vector;

    DataOut<dim> data_out;
    data_out.add_data_vector (stokes_dof_handler, stokes_solution,
                              stokes_names, stokes_component_interpretation);

    //data_out.add_data_vector (sig_dof_handler, sig_solution, "sigma");
    //data_out.add_data_vector (sig_diag_dof_handler, sig_diag_solution, "sigma12");
    data_out.add_data_vector (thick_dof_handler, thick_solution, "h");
    data_out.add_data_vector (conc_dof_handler, conc_solution, "c");
    data_out.add_data_vector (p_dof_handler, p_solution, "p");



    data_out.build_patches (std::min(stokes_degree, sig_degree));


    if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)
            == 0){
    std::ostringstream filename;
    filename << "solution-" << Utilities::int_to_string(timestep_number, 4) << ".vtk";

    std::ofstream output (filename.str().c_str());
    data_out.write_vtk (output);
    }
  }

  template <int dim>
  void SeaIceRheologyProblem<dim>::assign_old_stokes_sig_solution ()
  {
	  old_stokes_solution = stokes_solution;
  }

  template <int dim>
  void SeaIceRheologyProblem<dim>::assign_old_solution ()
  {
	  old_stokes_solution = stokes_solution;
	  old_thick_solution  = thick_solution;
	  old_conc_solution   = conc_solution;
  }

  template <int dim>
  double
  SeaIceRheologyProblem<dim>::
  compute_viscosity (const Tensor<2, dim> &u_grad,const double &pice) const
  {
	  double E = compute_E (u_grad);
	  double eta;
	  if(IceModel::ifgra == false)
      {
		  E = (E > IceModel::Estar) ? E : IceModel::Estar;
		  eta = pice/E;
	      eta = (eta > 1) ? eta : 1;
      }
	  else
	  {
		  E = (E > IceModel::Estar) ? E : IceModel::Estar;
		  eta = pice*IceModel::sinp_gra/E;
		  eta = (eta < IceModel::zeta_max_non) ? eta : IceModel::zeta_max_non;
	  }

	  return eta;

  }



  template <int dim>
  double
  SeaIceRheologyProblem<dim>::
  compute_pice (const double &h, const double &c) const
  {
	  return h*std::exp(-IceModel::k*(1-c));
  }

  template <int dim>
  double
  SeaIceRheologyProblem<dim>::
  compute_source_height (const double &h, const double &c) const
  {
	  double hdim = h*IceModel::hC/c;
	  double Vfdim;
      Vfdim = ( (hdim <= IceModel::lim) ? (IceModel::p1*hdim + IceModel::p2) : IceModel::a*std::pow(hdim,IceModel::b));
	  return Vfdim/IceModel::p2;
  }

  template <int dim>
  double
  SeaIceRheologyProblem<dim>::
  compute_E (const Tensor<2, dim> &u_grad) const
  {
	  double gamma = pow(u_grad[0][1] + u_grad[1][0],2)+
			         pow(u_grad[0][0] - u_grad[1][1],2)+
					 (IceModel::ifgra == true ? 0:
					 (pow(IceModel::alpha,2)*
					 pow(u_grad[0][0] + u_grad[1][1],2)));

      return std::sqrt(gamma);
  }



  template <int dim>
  void
  SeaIceRheologyProblem<dim>::refine_mesh ()
  {
    Vector<float> estimated_error_per_cell (triangulation.n_active_cells());

    FEValuesExtractors::Scalar pressure(dim);
    KellyErrorEstimator<dim>::estimate (stokes_dof_handler,
                                        QGauss<dim-1>(stokes_degree+1),
                                        typename FunctionMap<dim>::type(),
                                        stokes_solution,
                                        estimated_error_per_cell,
                                        stokes_fe.component_mask(pressure));

    GridRefinement::refine_and_coarsen_fixed_number (triangulation,
                                                     estimated_error_per_cell,
                                                     0.3, 0.0);
    triangulation.execute_coarsening_and_refinement ();

    rebuild_stokes_matrix         = true;
    rebuild_p_matrix              = true;
    rebuild_stokes_preconditioner = true;
    rebuild_sig_matrix            = true;
    rebuild_thick_matrix          = true;
  }


  template <int dim>
  void SeaIceRheologyProblem<dim>::bound_conc ()
  {
	  tmp_conc_solution = conc_solution;
	  unsigned int it = tmp_conc_solution.local_range().first;

	  for (; it != tmp_conc_solution.local_range().second ; ++it)
	  {
		  if(tmp_conc_solution(it) < 0)
			  tmp_conc_solution (it)=0;
		  else if(tmp_conc_solution(it) > 1.)
			  tmp_conc_solution (it) = 1.;

	  }

	  conc_solution = tmp_conc_solution;
  }

  template <int dim>
  void SeaIceRheologyProblem<dim>::bound_thick ()
  {
	  tmp_thick_solution = thick_solution;
	  unsigned int it = tmp_thick_solution.local_range().first;

	  for (; it != tmp_thick_solution.local_range().second ; ++it)
	  {
		  if(tmp_thick_solution(it) < 0)
			  tmp_thick_solution (it)=0;
	  }

	  thick_solution = tmp_thick_solution;
  }
  template <int dim>
  void SeaIceRheologyProblem<dim>::print_mesh_info(const Triangulation<dim> &tria,
                       const std::string        &filename)
  {
    std::cout << "Mesh info:" << std::endl
              << " dimension: " << dim << std::endl
              << " no. of cells: " << tria.n_active_cells() << std::endl;
    {
      std::map<unsigned int, unsigned int> boundary_count;
      typename Triangulation<dim>::active_cell_iterator
      cell = tria.begin_active(),
      endc = tria.end();
      for (; cell!=endc; ++cell)
        {
          for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
            {
              if (cell->face(face)->at_boundary())
              {
                boundary_count[cell->face(face)->boundary_id()]++;
                //std::cout<<cell->face(face)->boundary_id()<<" "<<std::endl;
              }
            }
        }
      std::cout << " boundary indicators: ";
      for (std::map<unsigned int, unsigned int>::iterator it=boundary_count.begin();
           it!=boundary_count.end();
           ++it)
        {
          std::cout << it->first << "(" << it->second << " times) ";
        }
      std::cout << std::endl;
    }
    std::ofstream out (filename.c_str());
    GridOut grid_out;
    grid_out.write_eps (tria, out);
    std::cout << " written to " << filename
              << std::endl
              << std::endl;
  }

  template <int dim>
  void SeaIceRheologyProblem<dim>::output_results_parallel (const std::string &filename_base)
  {
	  computing_timer.enter_section ("Postprocessing in Parallel");
	  mkdir(output_dir.c_str(), 0777);

	  const FESystem<dim> joint_fe (stokes_fe, 1,
	                                p_fe, 1,
									thick_fe, 1,
									conc_fe,1);

	  DoFHandler<dim> joint_dof_handler (triangulation);
	  joint_dof_handler.distribute_dofs (joint_fe);
	  Assert (joint_dof_handler.n_dofs() ==
	          stokes_dof_handler.n_dofs() + p_dof_handler.n_dofs() +
			  thick_dof_handler.n_dofs() + conc_dof_handler.n_dofs() ,
	          ExcInternalError());


	  TrilinosWrappers::MPI::Vector joint_solution;
	  joint_solution.reinit (joint_dof_handler.locally_owned_dofs(), MPI_COMM_WORLD);
	  {
	    std::vector<types::global_dof_index> local_joint_dof_indices (joint_fe.dofs_per_cell);
	    std::vector<types::global_dof_index> local_stokes_dof_indices (stokes_fe.dofs_per_cell);
	    std::vector<types::global_dof_index> local_p_dof_indices (p_fe.dofs_per_cell);
	    std::vector<types::global_dof_index> local_thick_dof_indices (thick_fe.dofs_per_cell);
	    std::vector<types::global_dof_index> local_conc_dof_indices (conc_fe.dofs_per_cell);

	    typename DoFHandler<dim>::active_cell_iterator
	    joint_cell       = joint_dof_handler.begin_active(),
	    joint_endc       = joint_dof_handler.end(),
	    stokes_cell      = stokes_dof_handler.begin_active(),
	    p_cell           = p_dof_handler.begin_active(),
	    thick_cell       = thick_dof_handler.begin_active(),
	    conc_cell        = conc_dof_handler.begin_active();

	    for (; joint_cell!=joint_endc;
	         ++joint_cell, ++stokes_cell, ++p_cell, ++thick_cell, ++conc_cell)
	    {
	        if (joint_cell->is_locally_owned())
	          {
	            joint_cell->get_dof_indices (local_joint_dof_indices);
	            stokes_cell->get_dof_indices (local_stokes_dof_indices);
	            p_cell->get_dof_indices (local_p_dof_indices);
	            thick_cell->get_dof_indices (local_thick_dof_indices);
	            conc_cell->get_dof_indices (local_conc_dof_indices);

	            for (unsigned int i=0; i<joint_fe.dofs_per_cell; ++i)
	            	switch (joint_fe.system_to_base_index(i).first.first)
	            	{
	            	case 0 :
	            		joint_solution(local_joint_dof_indices[i])
						= stokes_solution(local_stokes_dof_indices
								         [joint_fe.system_to_base_index(i).second]);
	            		break;
	            	case 1 :
	            		joint_solution(local_joint_dof_indices[i])
						= p_solution(local_p_dof_indices
								    [joint_fe.system_to_base_index(i).second]);
	            		break;
	            	case 2 :
	            		joint_solution(local_joint_dof_indices[i])
						= thick_solution(local_thick_dof_indices
								        [joint_fe.system_to_base_index(i).second]);
	            		break;
	            	case 3 :
	            		joint_solution(local_joint_dof_indices[i])
						= conc_solution(local_conc_dof_indices
								    [joint_fe.system_to_base_index(i).second]);
	            		break;
	            	default :
	            		pcout<<"index too big for parallel output "<<joint_fe.system_to_base_index(i).first.first<<" "<<std::endl;
	            	}

	          }

	    }
	  }

	  joint_solution.compress(VectorOperation::insert);
	  IndexSet locally_relevant_joint_dofs(joint_dof_handler.n_dofs());
	  DoFTools::extract_locally_relevant_dofs (joint_dof_handler, locally_relevant_joint_dofs);
	  TrilinosWrappers::MPI::Vector locally_relevant_joint_solution;
	  locally_relevant_joint_solution.reinit (locally_relevant_joint_dofs, MPI_COMM_WORLD);
	  locally_relevant_joint_solution = joint_solution;


	  //return;
	  Postprocessor postprocessor (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD),
	                               p_solution.min());
	  DataOut<dim> data_out;
	  data_out.attach_dof_handler (joint_dof_handler);
	  data_out.add_data_vector (locally_relevant_joint_solution, postprocessor);
	  data_out.build_patches ();

	  static int out_index=0;
	  const std::string filename = ("solution-" +
	                                Utilities::int_to_string (out_index, 5) +
	                                "." +
	                                Utilities::int_to_string
	                                (triangulation.locally_owned_subdomain(), 4) +
	                                ".vtu");
	  std::ofstream output (filename.c_str());
	  data_out.write_vtu (output);

	  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
	  {
		  std::vector<std::string> filenames;
		  for (unsigned int i=0; i<Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD); ++i)
			  filenames.push_back (std::string("solution-") +
					  Utilities::int_to_string (out_index, 5) +
					  "." +
					  Utilities::int_to_string(i, 4) +
					  ".vtu");
		  const std::string
		  pvtu_master_filename = ("solution-" +
				  Utilities::int_to_string (out_index, 5) +
				  ".pvtu");
		  std::ofstream pvtu_master (pvtu_master_filename.c_str());
		  data_out.write_pvtu_record (pvtu_master, filenames);
		  const std::string
		  visit_master_filename = ("solution-" +
				  Utilities::int_to_string (out_index, 5) +
				  ".visit");
		  std::ofstream visit_master (visit_master_filename.c_str());
		  data_out.write_visit_record (visit_master, filenames);
	  }



	  computing_timer.exit_section ();

	  out_index++;
  }

  template <int dim>
    void SeaIceRheologyProblem<dim>::output_results_parallel2 (const std::string &filename_base)
    {
  	  computing_timer.enter_section ("Postprocessing in Parallel");
  	  mkdir(output_dir.c_str(), 0777);
	  static int out_index=0;

  	DataOut<dim> data_out;
  	const std::vector<DataComponentInterpretation::DataComponentInterpretation>
  	data_component_interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);

  	const std::vector<DataComponentInterpretation::DataComponentInterpretation>
	scalar_interpretation(DataComponentInterpretation::component_is_scalar);

  	data_out.add_data_vector(stokes_dof_handler,stokes_solution,
  	                         std::vector<std::string> (dim, "velocity"),
  	                         data_component_interpretation);

	data_out.add_data_vector(p_dof_handler,p_solution,
                             std::string ("pressure"),
                             scalar_interpretation);

	data_out.add_data_vector(thick_dof_handler,thick_solution,
                             std::string ("h"),
                             scalar_interpretation);

	data_out.add_data_vector(conc_dof_handler,conc_solution,
                             std::string ("c"),
                             scalar_interpretation);

  	Vector<float> subdomain(triangulation.n_active_cells());
  	for (unsigned int i = 0; i < subdomain.size(); ++i)
  		subdomain(i) = triangulation.locally_owned_subdomain();
  	data_out.add_data_vector(subdomain, "subdomain");
  	data_out.build_patches();


  	const std::string filename =
  			(output_dir + filename_base + "_"
  					+ Utilities::int_to_string(out_index, 6) + "-"
					+ Utilities::int_to_string(triangulation.locally_owned_subdomain(), 2));
  	std::ofstream output_vtu((filename + ".vtu").c_str());
  	data_out.write_vtu(output_vtu);
  	pcout << output_dir + filename_base << ".pvtu" << std::endl;

  	if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  	{
  		std::vector<std::string> filenames;
  		for (unsigned int i = 0;
  				i < Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD); ++i)
  			filenames.push_back(filename_base + "_"
  					+ Utilities::int_to_string(out_index, 6) + "-"
					+ Utilities::int_to_string(i, 2) +
					".vtu");
  		std::ofstream pvtu_master_output((output_dir + filename_base + '_'
  				+ Utilities::int_to_string(out_index, 6) + ".pvtu").c_str());
  		data_out.write_pvtu_record(pvtu_master_output, filenames);
  		std::ofstream visit_master_output((output_dir + filename_base + '_'
  				+ Utilities::int_to_string(out_index, 6) + ".visit").c_str());
  		data_out.write_visit_record(visit_master_output, filenames);
  	}

  	computing_timer.exit_section ();
  	out_index ++;
    }

  template <int dim>
  void SeaIceRheologyProblem<dim>::run ()
  {


	  GridIn<2> gridin;
	  gridin.attach_triangulation(triangulation);
	  std::ifstream f("mesh.msh");
	  gridin.read_msh(f);

//	    {
//	      std::vector<unsigned int> subdivisions (dim, 1);
//	      subdivisions[0] = 80;
//	      subdivisions[1] = 1;
//
//	      const Point<dim> bottom_left = (dim == 2 ?
//	                                      Point<dim>(UserGeometry::x_inlet,UserGeometry::y_bottom) :
//	                                      Point<dim>(-2,0,-1));
//	      const Point<dim> top_right   = (dim == 2 ?
//	                                      Point<dim>(UserGeometry::x_outlet,UserGeometry::y_top) :
//	                                      Point<dim>(-1,1,0));
//
//	      GridGenerator::subdivided_hyper_rectangle (triangulation,
//	                                                 subdivisions,
//	                                                 bottom_left,
//	                                                 top_right);
//	    }
//
//
//	    for (typename Triangulation<dim>::active_cell_iterator
//	         cell = triangulation.begin_active();
//	         cell != triangulation.end(); ++cell)
//	      for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
//	      {
//	    	  if (cell->face(f)->center()[0] == UserGeometry::x_inlet)
//	    		  cell->face(f)->set_all_boundary_ids(1);
//	    	  if (cell->face(f)->center()[dim-1] == UserGeometry::y_top ||
//	    	      cell->face(f)->center()[dim-1] == UserGeometry::y_bottom)
//	    	  	  cell->face(f)->set_all_boundary_ids(2);
//	    	  if (cell->face(f)->center()[0] == UserGeometry::x_outlet)
//	    		  cell->face(f)->set_all_boundary_ids(3);
//
//	      }
//
//
//	    triangulation.refine_global (1);



	    print_mesh_info(triangulation, "grid-1.eps");

        setup_dofs ();

        EquationData::time = 0;

        for (unsigned int timestep = 0; timestep<EquationData::NT+1;
	         ++timestep)
	      {

	        pcout << "Time step  " << timestep << std::endl;

	        if(pseudo_stokes == true)
	        {
	        	assemble_stokes_system_parallel(false);
	            solve_stokes_parallel ();
	            assemble_p_system_parallel (false);
	            solve_p_parallel ();
	            prd_stokes_solution = stokes_solution;
	            cor_stokes_solution.equ(0.5,prd_stokes_solution,
	            		                0.5,old_stokes_solution);
	            prd_p_solution = p_solution;

//	            tp2_stokes_solution = stokes_solution;
	           unsigned int pse_step=0;
	           do {
	        	   pcout<<pse_step<<" "<<dif_stokes_solution.l2_norm() /stokes_solution.l2_norm()<<" "<<dif_p_solution.l2_norm() /p_solution.l2_norm()<<std::endl;
	        	   pse_step ++ ;
	        	   assemble_stokes_system_parallel (true);
		           solve_stokes_parallel ();
		           assemble_p_system_parallel (true);
		           solve_p_parallel ();
		           dif_stokes_solution.equ(1.0,prd_stokes_solution,
		            		              -1.0,    stokes_solution);
		           cor_stokes_solution.equ(0.5,prd_stokes_solution,
		            		               0.5,    stokes_solution);
		           prd_stokes_solution = stokes_solution;

		           dif_p_solution.equ(1.0,prd_p_solution,
		            		         -1.0,    p_solution);
		           prd_p_solution = p_solution;
	           } while (dif_stokes_solution.l2_norm() /stokes_solution.l2_norm()> IceModel :: tol_pseudo);
	           max_pseudo_step = max_pseudo_step < pse_step ? pse_step : max_pseudo_step;
	           std::cout<<"Current and Maximum (so far) Pseudo time steps taken are "<<pse_step<<" "<<max_pseudo_step<<" "<<std::endl;
	        }
	        else
	        {
	        	assemble_stokes_system_parallel (false);
	            solve_stokes_parallel ();
	            assemble_p_system_parallel (false);
	            solve_p_parallel ();
	        }

	        if(update_viscosity_every_subcycle == false)
	        	sub_old_stokes_solution = stokes_solution;

	        if(IceModel::iffix_thick==false || IceModel::iffix_conc==false)
	        {
	        	assemble_thick_conc_system_parallel ();
		        solve_thick_conc_parallel ();
	        }



	        if(IceModel::iffix_thick == true)
	        {
	        	thick_solution = IceModel::h0;
	        	std::cout<<'    ddddddddddddddddd    '<<std::endl;
	        }
	        else
	        	if(IceModel::ifbound_thick == true)
	        		bound_thick ();

	  	  fputs(IceModel::iffix_thick ? "true" : "false", stdout);
	  	  fputs(IceModel::iffix_conc ? "true" : "false", stdout);


	        if(IceModel::iffix_conc == true)
	        	conc_solution = IceModel::c0;
	        else
	        	if(IceModel::ifbound_conc == true)
	        		bound_conc ();


//	        assemble_p_system_parallel ();
//	        solve_p_parallel ();

	        assign_old_solution ();


	        if(timestep%EquationData::NtExp==0)
	        	output_results_parallel2 (std::string("solution"));

	        EquationData::time += EquationData::dt;

	        pcout << std::endl;
	      }



  }

  template <int dim>
  void SeaIceRheologyProblem<dim>::declare_parameters (ParameterHandler &prm)
  {
	    prm.declare_entry("Reynolds_number", "0.1",
	                      Patterns::Double(),
	                      "Normally from 0 to 2.");

	    prm.declare_entry("seaice_alpha", "1.1",
	                      Patterns::Double(),
	                      "Normally from 0 to 2.");

	    prm.declare_entry("seaice_k",     "20.0",
	                      Patterns::Double(),
	                      "Normally 20.");

	    prm.declare_entry("zeta_min",     "4E8",
	                      Patterns::Double(),
	                      "zeta minimum.");

	    prm.declare_entry("seaice_c0",    "1",
	                      Patterns::Double(),
	                      "From 0 to 1.");
	    prm.declare_entry("seaice_thick_dimensional",    "0.8",
	                      Patterns::Double(),
	                      "Dimensional value of ice thickness.");
	    prm.declare_entry("seaice_tol_pseudo",    "1e-3",
	                      Patterns::Double(),
	                      "Tolerance of pseudo time stepping for pure vp model.");

	    prm.declare_entry("stokes_solve",    "klu",
	                      Patterns::Selection("klu|gmres|mumps"),
	                       "Name of a linear solver for the Stokes System");

	    prm.declare_entry("total_time_steps",    "10000",
	                      Patterns::Integer(),
	                      "");

	    prm.declare_entry("export_time_steps",    "100",
	                      Patterns::Integer(),
	                      "");

	    prm.declare_entry("dt_dimensional",    "0.01",
	                      Patterns::Double(),
	                      "time step (day)");

	    prm.declare_entry("typical_size",    "1e4",
	                      Patterns::Double(),
	                      "typical size of the problem");

	    prm.declare_entry("wind_speed",    "15",
	                      Patterns::Double(),
	                      "typical wind speed (meter per second)");

	    prm.declare_entry("wind_angle",    "0",
	                      Patterns::Double(),
	                      "angle with respect to x-axis (degree)");

	    prm.declare_entry("wind_stress",    "0.2",
	                      Patterns::Double(),
	                      "imposed wind stress (Pascal)");

	    prm.declare_entry("coriolis_parameter",    "1.33E-4",
	                      Patterns::Double(),
	                      "Coriolis parameter (1/second)");

	    prm.declare_entry("T_wind",    "4",
	                      Patterns::Double(),
	                      "Time of the period of the wind");

	    prm.declare_entry("ice_production_rate",    "0.27",
	                      Patterns::Double(),
	                      "Frazil ice production rate (meter per day)");

	    prm.declare_entry("demarcation_thickness",    "0.3",
	                      Patterns::Double(),
	                      "demarcation thickness (meter)");

	    prm.declare_entry("dimension_katabatic",    "1e4",
	                      Patterns::Double(),
	                      "characteristic extent of katabatic wind (meter)");

	    prm.declare_entry ("strict_stokes", "true",
	                       Patterns::Bool(),
	                       "Whether absolutely removing the time-derivative term "
	                       "'true' or 'false'");

	    prm.declare_entry ("water_drag", "true",
	                       Patterns::Bool(),
	                       "Whether applying the water drag "
	                       "'true' or 'false'");

	    prm.declare_entry ("thermal_source", "true",
	                       Patterns::Bool(),
	                       "Whether adding the source term for thickness and concentration equations"
	                       "'true' or 'false'");

	    prm.declare_entry ("thermal_source_cmax", "true",
	                       Patterns::Bool(),
	                       "Whether using cmax or 1"
	                       "'true' or 'false'");

	    prm.declare_entry ("coriolis_force", "true",
	                       Patterns::Bool(),
	                       "Whether adding the Coriolis force"
	                       "'true' or 'false'");

	    prm.declare_entry ("periodic_wind", "true",
	    	               Patterns::Bool(),
	    	               "Whether applying periodic wind forcing "
	                       "'true' or 'false'");

	    prm.declare_entry ("impose_wind_stress", "true",
	    	               Patterns::Bool(),
	    	               "Whether impose or calculate wind stress "
	                       "'true' or 'false'");

	    prm.declare_entry ("fix_ice_thickness", "false",
	    	               Patterns::Bool(),
	    	               "Whether fix the thickness of ice"
	                       "'true' or 'false'");

	    prm.declare_entry ("fix_ice_concentration", "false",
	    	               Patterns::Bool(),
	    	               "Whether fix the concentration of ice"
	                       "'true' or 'false'");

	    prm.declare_entry ("bound_ice_thickness", "false",
	    	               Patterns::Bool(),
	    	               "Whether bound the thickness of ice"
	                       "'true' or 'false'");

	    prm.declare_entry ("bound_ice_concentration", "false",
	    	               Patterns::Bool(),
	    	               "Whether bound the concentration of ice"
	                       "'true' or 'false'");

	    prm.declare_entry ("apply_katabatic_wind", "false",
	    	               Patterns::Bool(),
	    	               "Whether apply the katabatic wind"
	                       "'true' or 'false'");
	    prm.declare_entry ("granular_model", "false",
	    	               Patterns::Bool(),
	    	               "Whether apply the granular rheology model"
	                       "'true' or 'false'");


  }

  template <int dim>
  void SeaIceRheologyProblem<dim>::init_icemodel (const ParameterHandler &prm)
  {
	  EquationData::Re = prm.get_double("Reynolds_number");
	  IceModel::alpha = prm.get_double("seaice_alpha");
	  if(IceModel::ifgra==true)
		  IceModel::alpha = 1;

	  IceModel::k     = prm.get_double("seaice_k");
	  IceModel::zeta_min = prm.get_double("zeta_min");

	  IceModel::c0 = prm.get_double("seaice_c0");

	  IceModel::hinput = prm.get_double("seaice_thick_dimensional"); // meter
	  IceModel::tol_pseudo = prm.get_double("seaice_tol_pseudo");
	  IceModel::ifthermal = prm.get_bool("thermal_source");
	  IceModel::ifthermal_cmax = prm.get_bool("thermal_source_cmax");


	  IceModel::ifcorio = prm.get_bool("coriolis_force");
	  if(IceModel::ifcorio==true)
		  IceModel::fco = prm.get_double("coriolis_parameter");
	  IceModel::ifwdrag = prm.get_bool("water_drag");
	  EquationData::ifstr_stokes = prm.get_bool("strict_stokes");

	  IceModel::ifsts_wd = prm.get_bool("impose_wind_stress");
	  if(IceModel::ifsts_wd == true)
		  IceModel::f = prm.get_double("wind_stress");
	  else
	  {
		  IceModel::uwind = prm.get_double("wind_speed");
		  IceModel::f = IceModel::rho_air*IceModel::Ca*std::pow(IceModel::uwind,2);
	  }

	  IceModel::theta = prm.get_double("wind_angle");
	  IceModel::ifperi_wd = prm.get_bool("periodic_wind");
	  IceModel::iffix_thick = prm.get_bool("fix_ice_thickness");



	  IceModel::iffix_conc = prm.get_bool("fix_ice_concentration");
	  IceModel::ifbound_thick = prm.get_bool("bound_ice_thickness");
	  IceModel::ifbound_conc = prm.get_bool("bound_ice_concentration");
	  IceModel::ifkata = prm.get_bool("apply_katabatic_wind");
	  IceModel::ifgra = prm.get_bool("granular_model");

//	  fputs(IceModel::iffix_thick ? "true" : "false", stdout);
//	  fputs(IceModel::iffix_conc ? "true" : "false", stdout);


	  IceModel::wC = prm.get_double("typical_size");
	  IceModel::Tperi = prm.get_double("T_wind");
	  IceModel::Vf = prm.get_double("ice_production_rate"); // meter per day
	  IceModel::hd = prm.get_double("demarcation_thickness"); // meter
	  if(IceModel::ifkata == true)
		  {
		  	  IceModel::size_kata = prm.get_double("dimension_katabatic"); // meter
		  	  IceModel::size_kata_non = IceModel::size_kata/IceModel::wC;
		  }

	  SolverConfig::stokes = prm.get("stokes_solve");
	  EquationData::NT = prm.get_integer("total_time_steps");
	  EquationData::NtExp = prm.get_integer("export_time_steps");
	  EquationData::dt_dim = prm.get_double("dt_dimensional");


	  IceModel::uC = std::pow(IceModel::alpha,2)*std::pow(IceModel::wC,2)*IceModel::f/IceModel::zeta_min;
	  IceModel::Estar = IceModel::Estar_dim*IceModel::zeta_min/(IceModel::alpha*IceModel::wC*IceModel::f);
      IceModel::beta  = IceModel::rho_water * IceModel::Cw * IceModel::f * std::pow(IceModel::wC,4)
                        /std::pow(IceModel::zeta_min,2);
      IceModel::tC    = IceModel::zeta_min/(IceModel::wC*std::pow(IceModel::alpha,2)*IceModel::f);
      IceModel::Tperi_non = IceModel::Tperi * EquationData::sec_per_day / IceModel::tC;
      IceModel::Sigma = IceModel::Vf/EquationData::sec_per_day/(IceModel::alpha*IceModel::uC*IceModel::f/IceModel::S);
      IceModel::hC = IceModel::alpha*IceModel::wC*IceModel::f/IceModel::S;
      IceModel::hd_non= IceModel::hd/IceModel::hC;
      IceModel::h0 = IceModel::hinput/IceModel::hC;
      IceModel::zeta_max_non = IceModel::zeta_max/IceModel::zeta_min;
	  EquationData::dt = EquationData::dt_dim*EquationData::sec_per_day/IceModel::tC;

	  if(IceModel::ifcorio==true)
		  IceModel::fco_non = IceModel::fco*IceModel::tC;

	  pcout<<"Read parameters "<<IceModel::c0<<" "<<IceModel::h0<<" "<<IceModel::Estar<<" "<<IceModel::beta<<
		                         IceModel::Tperi_non<<IceModel::tC<<std::endl;

	  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)
	          == 0)
	  {
		  std::ofstream myfile;
		  myfile.open ("parameter_calculate.txt");
		  myfile << "Estar="<<IceModel::Estar<<"\n";
		  myfile << "beta="<<IceModel::beta<<"\n";
		  myfile << "t0="<<IceModel::tC/EquationData::sec_per_day<<"day \n";
		  myfile << "Tperi_non="<<IceModel::Tperi_non<<"\n";
		  myfile << "Sigma="<<IceModel::Sigma<<"\n";
		  myfile << "Characteristic ice thickness"<<IceModel::hC<<"\n";
		  myfile << "Imposing wind stress directly="<<IceModel::ifsts_wd<<"\n";
		  if(IceModel::ifsts_wd == true)
		  {
			  myfile << "Imposed wind stress="<<IceModel::f<<"\n";
			  myfile << "Corresponding wind speed (meter per second)="<<std::sqrt(IceModel::f/IceModel::rho_air/IceModel::Ca)<<"\n";
		  }
		  else
		  {
			  myfile << "Imposed wind speed="<<IceModel::uwind<<"\n";
			  myfile << "Corresponding wind stress (Pascal)="<<IceModel::f<<"\n";
		  }
		  if(IceModel::ifkata == true)
			  myfile << "Dimensionless katabatic extent="<<IceModel::size_kata_non<<"\n";



		  myfile.close();
	  }
  }
}



int main (int argc, char *argv[])
{
  using namespace dealii;
  using namespace IceRheo;

  try
    {
      ParameterHandler prm;
      SeaIceRheologyProblem<2>::declare_parameters(prm);

      if (argc != 2)
        {
          std::cerr << "*** Call this program as <./executable input.prm>" << std::endl;
          return 1;
        }
      prm.read_input(argv[1]);

      Utilities::MPI::MPI_InitFinalize mpi_initialization (argc, argv,
                                                           numbers::invalid_unsigned_int);

//      AssertThrow(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)==1,
//                  ExcMessage("This program can only be run in serial, use ./executable"));
      {
    	  SeaIceRheologyProblem<2> flow_problem (prm) ;
    	  flow_problem.init_icemodel (prm);
    	  flow_problem.run ();
      }
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
