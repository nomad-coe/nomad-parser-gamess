import numpy as np            # pylint: disable=unused-import
import typing                 # pylint: disable=unused-import
from nomad.metainfo import (  # pylint: disable=unused-import
    MSection, MCategory, Category, Package, Quantity, Section, SubSection, SectionProxy,
    Reference
)
from nomad.metainfo.legacy import LegacyDefinition

from nomad.datamodel.metainfo import public

m_package = Package(
    name='gamess_nomadmetainfo_json',
    description='None',
    a_legacy=LegacyDefinition(name='gamess.nomadmetainfo.json'))


class x_gamess_section_atom_forces(MSection):
    '''
    section that contains Cartesian forces of the system for a given geometry
    '''

    m_def = Section(validate=False, a_legacy=LegacyDefinition(name='x_gamess_section_atom_forces'))

    x_gamess_atom_x_force = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='newton',
        description='''
        -
        ''',
        a_legacy=LegacyDefinition(name='x_gamess_atom_x_force'))

    x_gamess_atom_y_force = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='newton',
        description='''
        -
        ''',
        a_legacy=LegacyDefinition(name='x_gamess_atom_y_force'))

    x_gamess_atom_z_force = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='newton',
        description='''
        -
        ''',
        a_legacy=LegacyDefinition(name='x_gamess_atom_z_force'))


class x_gamess_section_cis(MSection):
    '''
    Configuration interaction singles excitation energies and oscillator strengths.
    '''

    m_def = Section(validate=False, a_legacy=LegacyDefinition(name='x_gamess_section_cis'))

    x_gamess_cis_excitation_energy = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='joule',
        description='''
        Value of the excitation energies for configuration interaction singles excited
        states.
        ''',
        a_legacy=LegacyDefinition(name='x_gamess_cis_excitation_energy'))

    x_gamess_cis_oscillator_strength = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        Value of the oscillator strengths for configuration interaction singles excited
        states.
        ''',
        a_legacy=LegacyDefinition(name='x_gamess_cis_oscillator_strength'))


class x_gamess_section_ci(MSection):
    '''
    Configuration interaction energies.
    '''

    m_def = Section(validate=False, a_legacy=LegacyDefinition(name='x_gamess_section_ci'))


class x_gamess_section_coupled_cluster(MSection):
    '''
    Coupled cluster energies.
    '''

    m_def = Section(validate=False, a_legacy=LegacyDefinition(name='x_gamess_section_coupled_cluster'))


class x_gamess_section_elstruc_method(MSection):
    '''
    Section containing electronic structure method.
    '''

    m_def = Section(validate=False, a_legacy=LegacyDefinition(name='x_gamess_section_elstruc_method'))

    x_gamess_electronic_structure_method = Quantity(
        type=str,
        shape=[],
        description='''
        Name of electronic structure method.
        ''',
        a_legacy=LegacyDefinition(name='x_gamess_electronic_structure_method'))


class x_gamess_section_excited_states(MSection):
    '''
    Time-dependent DFT and configuration interaction singles results.
    '''

    m_def = Section(validate=False, a_legacy=LegacyDefinition(name='x_gamess_section_excited_states'))

    x_gamess_section_cis = SubSection(
        sub_section=SectionProxy('x_gamess_section_cis'),
        repeats=True,
        a_legacy=LegacyDefinition(name='x_gamess_section_cis'))

    x_gamess_section_tddft = SubSection(
        sub_section=SectionProxy('x_gamess_section_tddft'),
        repeats=True,
        a_legacy=LegacyDefinition(name='x_gamess_section_tddft'))


class x_gamess_section_frequencies(MSection):
    '''
    section for the values of the frequencies, reduced masses and normal mode vectors
    '''

    m_def = Section(validate=False, a_legacy=LegacyDefinition(name='x_gamess_section_frequencies'))

    x_gamess_frequencies = Quantity(
        type=np.dtype(np.float64),
        shape=['number_of_frequencies'],
        description='''
        values of frequencies, in cm-1
        ''',
        a_legacy=LegacyDefinition(name='x_gamess_frequencies'))

    x_gamess_frequency_values = Quantity(
        type=str,
        shape=[],
        description='''
        values of frequencies, in cm-1
        ''',
        a_legacy=LegacyDefinition(name='x_gamess_frequency_values'))

    x_gamess_red_masses = Quantity(
        type=np.dtype(np.float64),
        shape=['number_of_frequencies'],
        description='''
        values of normal mode reduced masses
        ''',
        a_legacy=LegacyDefinition(name='x_gamess_red_masses'))

    x_gamess_reduced_masses = Quantity(
        type=str,
        shape=['number_of_reduced_masses_rows'],
        description='''
        values of normal mode reduced masses
        ''',
        a_legacy=LegacyDefinition(name='x_gamess_reduced_masses'))


class x_gamess_section_geometry_optimization_info(MSection):
    '''
    Specifies whether a geometry optimization is converged.
    '''

    m_def = Section(validate=False, a_legacy=LegacyDefinition(name='x_gamess_section_geometry_optimization_info'))

    x_gamess_geometry_optimization_converged = Quantity(
        type=str,
        shape=[],
        description='''
        Specifies whether a geometry optimization is converged.
        ''',
        a_legacy=LegacyDefinition(name='x_gamess_geometry_optimization_converged'))


class x_gamess_section_geometry(MSection):
    '''
    section that contains Cartesian coordinates of the system for a given geometry
    '''

    m_def = Section(validate=False, a_legacy=LegacyDefinition(name='x_gamess_section_geometry'))

    x_gamess_atom_positions = Quantity(
        type=np.dtype(np.float64),
        shape=['number_of_atoms', 3],
        unit='meter',
        description='''
        Initial positions of all the atoms, in Cartesian coordinates.
        ''',
        a_legacy=LegacyDefinition(name='x_gamess_atom_positions'))

    x_gamess_atom_x_coord = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='meter',
        description='''
        x coordinate for the atoms
        ''',
        a_legacy=LegacyDefinition(name='x_gamess_atom_x_coord'))

    x_gamess_atom_y_coord = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='meter',
        description='''
        y coordinate for the atoms
        ''',
        a_legacy=LegacyDefinition(name='x_gamess_atom_y_coord'))

    x_gamess_atom_z_coord = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='meter',
        description='''
        z coordinate for the atoms
        ''',
        a_legacy=LegacyDefinition(name='x_gamess_atom_z_coord'))


class x_gamess_section_mcscf(MSection):
    '''
    Multiconfigurational SCF energies.
    '''

    m_def = Section(validate=False, a_legacy=LegacyDefinition(name='x_gamess_section_mcscf'))

    x_gamess_energy_mcscf_iteration = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='joule',
        description='''
        Value of the MCSCF total energy, normally CASSCF, during the iterations.
        ''',
        a_legacy=LegacyDefinition(name='x_gamess_energy_mcscf_iteration'))

    x_gamess_mcscf_active_electrons = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        Number of MCSCF active electrons in the calculation.
        ''',
        a_legacy=LegacyDefinition(name='x_gamess_mcscf_active_electrons'))

    x_gamess_mcscf_active_orbitals = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        Number of MCSCF active orbitals in the calculation.
        ''',
        a_legacy=LegacyDefinition(name='x_gamess_mcscf_active_orbitals'))

    x_gamess_mcscf_inactive_orbitals = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        Number of MCSCF inactive orbitals in the calculation.
        ''',
        a_legacy=LegacyDefinition(name='x_gamess_mcscf_inactive_orbitals'))


class x_gamess_section_moller_plesset(MSection):
    '''
    Perturbative Moller-Plesset energies.
    '''

    m_def = Section(validate=False, a_legacy=LegacyDefinition(name='x_gamess_section_moller_plesset'))


class x_gamess_section_mrpt2(MSection):
    '''
    Multiference multiconfigurational energies at second order of perturbation theory.
    '''

    m_def = Section(validate=False, a_legacy=LegacyDefinition(name='x_gamess_section_mrpt2'))

    x_gamess_mrpt2_active_electrons = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        Number of active electrons in MRPT2 calculation.
        ''',
        a_legacy=LegacyDefinition(name='x_gamess_mrpt2_active_electrons'))

    x_gamess_mrpt2_active_orbitals = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        Number of active orbitals in MRPT2 calculation.
        ''',
        a_legacy=LegacyDefinition(name='x_gamess_mrpt2_active_orbitals'))

    x_gamess_mrpt2_doubly_occupied_orbitals = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        Number of doubly occupied orbitals in MRPT2 calculation.
        ''',
        a_legacy=LegacyDefinition(name='x_gamess_mrpt2_doubly_occupied_orbitals'))

    x_gamess_mrpt2_external_orbitals = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        Number of external orbitals in MRPT2 calculation.
        ''',
        a_legacy=LegacyDefinition(name='x_gamess_mrpt2_external_orbitals'))

    x_gamess_mrpt2_frozen_core_orbitals = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        Number of frozen core orbitals in MRPT2 calculation.
        ''',
        a_legacy=LegacyDefinition(name='x_gamess_mrpt2_frozen_core_orbitals'))

    x_gamess_mrpt2_frozen_virtual_orbitals = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        Number of frozen virtual orbitals in MRPT2 calculation.
        ''',
        a_legacy=LegacyDefinition(name='x_gamess_mrpt2_frozen_virtual_orbitals'))

    x_gamess_mrpt2_method_type = Quantity(
        type=str,
        shape=[],
        description='''
        Determinant (MRPT2) or CSF (MC-QDPT) method for second-order perturbation theory
        calculation
        ''',
        a_legacy=LegacyDefinition(name='x_gamess_mrpt2_method_type'))


class x_gamess_section_scf_hf_method(MSection):
    '''
    Section containing type of SCF method employed (RHF,UHF,ROHF or GVB).
    '''

    m_def = Section(validate=False, a_legacy=LegacyDefinition(name='x_gamess_section_scf_hf_method'))

    x_gamess_scf_hf_method = Quantity(
        type=str,
        shape=[],
        description='''
        Type of SCF method employed.
        ''',
        a_legacy=LegacyDefinition(name='x_gamess_scf_hf_method'))


class x_gamess_section_tddft(MSection):
    '''
    Time-dependent DFT excitation energies and oscillator strengths.
    '''

    m_def = Section(validate=False, a_legacy=LegacyDefinition(name='x_gamess_section_tddft'))

    x_gamess_tddft_excitation_energy = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='joule',
        description='''
        Value of the excitation energies for time-dependent DFT excited states.
        ''',
        a_legacy=LegacyDefinition(name='x_gamess_tddft_excitation_energy'))

    x_gamess_tddft_oscillator_strength = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        Value of the oscillator strengths for time-dependent DFT excited states.
        ''',
        a_legacy=LegacyDefinition(name='x_gamess_tddft_oscillator_strength'))


class section_eigenvalues(public.section_eigenvalues):

    m_def = Section(validate=False, extends_base_section=True, a_legacy=LegacyDefinition(name='section_eigenvalues'))

    x_gamess_alpha_eigenvalues_values = Quantity(
        type=str,
        shape=[],
        description='''
        values of eigenenergies for alpha MOs
        ''',
        a_legacy=LegacyDefinition(name='x_gamess_alpha_eigenvalues_values'))

    x_gamess_beta_eigenvalues_values = Quantity(
        type=str,
        shape=[],
        description='''
        values of eigenenergies for occupied beta MOs
        ''',
        a_legacy=LegacyDefinition(name='x_gamess_beta_eigenvalues_values'))


class section_system(public.section_system):

    m_def = Section(validate=False, extends_base_section=True, a_legacy=LegacyDefinition(name='section_system'))

    x_gamess_atom_positions_initial = Quantity(
        type=np.dtype(np.float64),
        shape=['number_of_atoms', 3],
        unit='meter',
        description='''
        Initial positions of all the atoms, in Cartesian coordinates.
        ''',
        categories=[public.configuration_core],
        a_legacy=LegacyDefinition(name='x_gamess_atom_positions_initial'))

    x_gamess_atom_x_coord_initial = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='meter',
        description='''
        x coordinate for the atoms of the initial geometry
        ''',
        a_legacy=LegacyDefinition(name='x_gamess_atom_x_coord_initial'))

    x_gamess_atom_y_coord_initial = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='meter',
        description='''
        y coordinate for the atoms of the initial geometry
        ''',
        a_legacy=LegacyDefinition(name='x_gamess_atom_y_coord_initial'))

    x_gamess_atom_z_coord_initial = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='meter',
        description='''
        z coordinate for the atoms of the initial geometry
        ''',
        a_legacy=LegacyDefinition(name='x_gamess_atom_z_coord_initial'))

    x_gamess_number_of_electrons = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        number of electrons for system
        ''',
        a_legacy=LegacyDefinition(name='x_gamess_number_of_electrons'))

    x_gamess_atomic_number = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        atomic number for atoms
        ''',
        a_legacy=LegacyDefinition(name='x_gamess_atomic_number'))

    x_gamess_memory = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        Total memory for GAMESS job
        ''',
        a_legacy=LegacyDefinition(name='x_gamess_memory'))

    x_gamess_spin_target_multiplicity = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        Target (user-imposed) value of the spin multiplicity $M=2S+1$, where $S$ is the
        total spin. It is an integer value. This value is not necessarly the value
        obtained at the end of the calculation. See spin_S2 for the converged value of the
        spin moment.
        ''',
        a_legacy=LegacyDefinition(name='x_gamess_spin_target_multiplicity'))

    x_gamess_total_charge = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        Total charge of the system.
        ''',
        a_legacy=LegacyDefinition(name='x_gamess_total_charge'))


class section_method(public.section_method):

    m_def = Section(validate=False, extends_base_section=True, a_legacy=LegacyDefinition(name='section_method'))

    x_gamess_basis_set_diffsp = Quantity(
        type=str,
        shape=[],
        description='''
        Include a set of SP diffuse functions on heavy atoms or not
        ''',
        a_legacy=LegacyDefinition(name='x_gamess_basis_set_diffsp'))

    x_gamess_basis_set_diffs = Quantity(
        type=str,
        shape=[],
        description='''
        Incluse a set of S diffuse functions on light atoms or not
        ''',
        a_legacy=LegacyDefinition(name='x_gamess_basis_set_diffs'))

    x_gamess_basis_set_gbasis = Quantity(
        type=str,
        shape=[],
        description='''
        Gaussian basis set main name
        ''',
        a_legacy=LegacyDefinition(name='x_gamess_basis_set_gbasis'))

    x_gamess_basis_set_igauss = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        Number of main Gaussians
        ''',
        a_legacy=LegacyDefinition(name='x_gamess_basis_set_igauss'))

    x_gamess_basis_set_ndfunc = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        Number of polarization d function sets on heavy atoms
        ''',
        a_legacy=LegacyDefinition(name='x_gamess_basis_set_ndfunc'))

    x_gamess_basis_set_nffunc = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        Number of polarization f function sets on heavy atoms
        ''',
        a_legacy=LegacyDefinition(name='x_gamess_basis_set_nffunc'))

    x_gamess_basis_set_npfunc = Quantity(
        type=str,
        shape=[],
        description='''
        Number of polarization p function sets on light atoms
        ''',
        a_legacy=LegacyDefinition(name='x_gamess_basis_set_npfunc'))

    x_gamess_basis_set_polar = Quantity(
        type=str,
        shape=[],
        description='''
        Exponents of polarization functions
        ''',
        a_legacy=LegacyDefinition(name='x_gamess_basis_set_polar'))

    x_gamess_cctype = Quantity(
        type=str,
        shape=[],
        description='''
        Type of coupled cluster method employed
        ''',
        a_legacy=LegacyDefinition(name='x_gamess_cctype'))

    x_gamess_cistep = Quantity(
        type=str,
        shape=[],
        description='''
        Determinant or CSF method for multiconfigurational SCF calculation
        ''',
        a_legacy=LegacyDefinition(name='x_gamess_cistep'))

    x_gamess_citype = Quantity(
        type=str,
        shape=[],
        description='''
        Type of configuration interaction method employed
        ''',
        a_legacy=LegacyDefinition(name='x_gamess_citype'))

    x_gamess_comp_method = Quantity(
        type=str,
        shape=[],
        description='''
        Control if the G3MP2 composite method has been defined
        ''',
        a_legacy=LegacyDefinition(name='x_gamess_comp_method'))

    x_gamess_mcscf_casscf = Quantity(
        type=str,
        shape=[],
        description='''
        This indicates whether the multiconfigurational SCF calculation is a complete
        active space SCF calculation or not
        ''',
        a_legacy=LegacyDefinition(name='x_gamess_mcscf_casscf'))

    x_gamess_method = Quantity(
        type=str,
        shape=[],
        description='''
        String identifying in an unique way the WF method used for the final
        wavefunctions.
        ''',
        a_legacy=LegacyDefinition(name='x_gamess_method'))

    x_gamess_mplevel = Quantity(
        type=str,
        shape=[],
        description='''
        Level of second-orden Moller-Plesset perturbation theory
        ''',
        a_legacy=LegacyDefinition(name='x_gamess_mplevel'))

    x_gamess_pptype = Quantity(
        type=str,
        shape=[],
        description='''
        Name of the pseudopotential employed
        ''',
        a_legacy=LegacyDefinition(name='x_gamess_pptype'))

    x_gamess_relatmethod = Quantity(
        type=str,
        shape=[],
        description='''
        Type of relativistic method employed
        ''',
        a_legacy=LegacyDefinition(name='x_gamess_relatmethod'))

    x_gamess_scf_method = Quantity(
        type=str,
        shape=[],
        description='''
        String identifying in an unique way the SCF method used in the calculation.
        ''',
        a_legacy=LegacyDefinition(name='x_gamess_scf_method'))

    x_gamess_scf_type = Quantity(
        type=str,
        shape=[],
        description='''
        String identifying in an unique way the SCF method used in the calculation.
        ''',
        a_legacy=LegacyDefinition(name='x_gamess_scf_type'))

    x_gamess_tddfttype = Quantity(
        type=str,
        shape=[],
        description='''
        Type of time-dependent DFT calculation
        ''',
        a_legacy=LegacyDefinition(name='x_gamess_tddfttype'))

    x_gamess_vbtype = Quantity(
        type=str,
        shape=[],
        description='''
        Type of valence bond method employed
        ''',
        a_legacy=LegacyDefinition(name='x_gamess_vbtype'))

    x_gamess_xc = Quantity(
        type=str,
        shape=[],
        description='''
        String identifying in an unique way the XC method used for the final
        wavefunctions.
        ''',
        a_legacy=LegacyDefinition(name='x_gamess_xc'))

    x_gamess_section_elstruc_method = SubSection(
        sub_section=SectionProxy('x_gamess_section_elstruc_method'),
        repeats=True,
        a_legacy=LegacyDefinition(name='x_gamess_section_elstruc_method'))

    x_gamess_section_scf_hf_method = SubSection(
        sub_section=SectionProxy('x_gamess_section_scf_hf_method'),
        repeats=True,
        a_legacy=LegacyDefinition(name='x_gamess_section_scf_hf_method'))


class section_run(public.section_run):

    m_def = Section(validate=False, extends_base_section=True, a_legacy=LegacyDefinition(name='section_run'))

    x_gamess_program_execution_date = Quantity(
        type=str,
        shape=[],
        description='''
        -
        ''',
        a_legacy=LegacyDefinition(name='x_gamess_program_execution_date'))

    x_gamess_program_implementation = Quantity(
        type=str,
        shape=[],
        description='''
        -
        ''',
        a_legacy=LegacyDefinition(name='x_gamess_program_implementation'))

    x_gamess_section_geometry_optimization_info = SubSection(
        sub_section=SectionProxy('x_gamess_section_geometry_optimization_info'),
        repeats=True,
        a_legacy=LegacyDefinition(name='x_gamess_section_geometry_optimization_info'))


class section_scf_iteration(public.section_scf_iteration):

    m_def = Section(validate=False, extends_base_section=True, a_legacy=LegacyDefinition(name='section_scf_iteration'))

    x_gamess_energy_scf = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='joule',
        description='''
        Final value of the total electronic energy calculated with the method described in
        XC_method.
        ''',
        a_legacy=LegacyDefinition(name='x_gamess_energy_scf'))

    x_gamess_energy_total_scf_iteration = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='joule',
        description='''
        Value of the total electronic energy calculated with the method described in
        XC_method during each self-consistent field (SCF) iteration.
        ''',
        a_legacy=LegacyDefinition(name='x_gamess_energy_total_scf_iteration'))


class section_single_configuration_calculation(public.section_single_configuration_calculation):

    m_def = Section(validate=False, extends_base_section=True, a_legacy=LegacyDefinition(name='section_single_configuration_calculation'))

    x_gamess_section_atom_forces = SubSection(
        sub_section=SectionProxy('x_gamess_section_atom_forces'),
        repeats=True,
        a_legacy=LegacyDefinition(name='x_gamess_section_atom_forces'))

    x_gamess_section_ci = SubSection(
        sub_section=SectionProxy('x_gamess_section_ci'),
        repeats=True,
        a_legacy=LegacyDefinition(name='x_gamess_section_ci'))

    x_gamess_section_coupled_cluster = SubSection(
        sub_section=SectionProxy('x_gamess_section_coupled_cluster'),
        repeats=True,
        a_legacy=LegacyDefinition(name='x_gamess_section_coupled_cluster'))

    x_gamess_section_excited_states = SubSection(
        sub_section=SectionProxy('x_gamess_section_excited_states'),
        repeats=True,
        a_legacy=LegacyDefinition(name='x_gamess_section_excited_states'))

    x_gamess_section_frequencies = SubSection(
        sub_section=SectionProxy('x_gamess_section_frequencies'),
        repeats=True,
        a_legacy=LegacyDefinition(name='x_gamess_section_frequencies'))

    x_gamess_section_geometry = SubSection(
        sub_section=SectionProxy('x_gamess_section_geometry'),
        repeats=True,
        a_legacy=LegacyDefinition(name='x_gamess_section_geometry'))

    x_gamess_section_mcscf = SubSection(
        sub_section=SectionProxy('x_gamess_section_mcscf'),
        repeats=True,
        a_legacy=LegacyDefinition(name='x_gamess_section_mcscf'))

    x_gamess_section_moller_plesset = SubSection(
        sub_section=SectionProxy('x_gamess_section_moller_plesset'),
        repeats=True,
        a_legacy=LegacyDefinition(name='x_gamess_section_moller_plesset'))

    x_gamess_section_mrpt2 = SubSection(
        sub_section=SectionProxy('x_gamess_section_mrpt2'),
        repeats=True,
        a_legacy=LegacyDefinition(name='x_gamess_section_mrpt2'))


m_package.__init_metainfo__()
