# Copyright 2016-2018 Rosendo Valero, Fawzi Mohamed, Danio Brambila
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

from __future__ import division
from builtins import str
from builtins import range
from builtins import object
from functools import reduce
from nomadcore.simple_parser import mainFunction, SimpleMatcher as SM
from nomadcore.local_meta_info import loadJsonFile, InfoKindEl
from nomadcore.caching_backend import CachingLevel
from nomadcore.unit_conversion.unit_conversion import convert_unit
import os, sys, json, logging
import numpy as np
import ase
import re

import warnings
warnings.simplefilter('ignore', category=FutureWarning)

############################################################
# This is the parser for the output file of GAMESS.
############################################################

logger = logging.getLogger("nomad.GAMESSParser")

# description of the output
mainFileDescription = SM(
    name = 'root',
    weak = True,
    forwardMatch = True,
    startReStr = "",
    subMatchers = [
        SM(name = 'newRun',
           startReStr = r"\s*\*\s*GAMESS VERSION \=\s*|\s*\*\s*Firefly version",
           repeats = True,
           required = True,
           forwardMatch = True,
           fixedStartValues={ 'program_name': 'GAMESS', 'program_basis_set_type': 'gaussians' },
           sections   = ['section_run'],
           subMatchers = [
               SM(name = 'header',
                  startReStr = r"\s*\*\s*GAMESS VERSION \=\s*|\s*\*\s*Firefly version",
                  forwardMatch = True,
                  subMatchers = [
                      SM(r"\s*\*\s*GAMESS VERSION \=\s*(?P<program_version>[0-9]+\s*[A-Z]+\s*[0-9]+)"),
                      SM(r"\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\s*(?P<x_gamess_program_implementation>[0-9]+\s*[A-Z]+\s*[A-Z]+\s*[A-Z]+)"),
                      SM(r"\s*EXECUTION OF GAMESS BEGUN\s*(?P<x_gamess_program_execution_date>[a-zA-Z]+\s*[a-zA-Z]+\s*[0-9]+\s*[0-9][0-9][:][[0-9][0-9][:][0-9][0-9]\s*[0-9]+)"),
                      SM(r"\s*\*\s*Firefly version\s*(?P<program_version>[0-9.]+)"),
                      ]),
               SM(name = 'memory',
               sections = ['section_system'],
                  startReStr = r"\s*ECHO OF THE FIRST FEW INPUT CARDS",
                  forwardMatch = True,
                  subMatchers = [
                      SM(r"\s*(?P<x_gamess_memory>[0-9]+)\s*WORDS OF MEMORY AVAILABLE"),
                      ]
             ),
               SM (name = 'SectionMethodBasisSet',
               sections = ['section_method'],
                   startReStr = r"\s*BASIS OPTIONS",
                   forwardMatch = False,
                   subMatchers = [
                      SM(r"\s*GBASIS=(?P<x_gamess_basis_set_gbasis>[A-Z0-9-]+)\s*IGAUSS=\s*(?P<x_gamess_basis_set_igauss>[0-9]+)\s*POLAR=(?P<x_gamess_basis_set_polar>[A-Z]+)"),
                      SM(r"\s*NDFUNC=\s*(?P<x_gamess_basis_set_ndfunc>[0-9]+)\s*NFFUNC=\s*(?P<x_gamess_basis_set_nffunc>[0-9]+)\s*DIFFSP=\s*(?P<x_gamess_basis_set_diffsp>[TF])"),
                      SM(r"\s*NPFUNC=\s*(?P<x_gamess_basis_set_npfunc>[0-9]+)\s*DIFFS=\s*(?P<x_gamess_basis_set_diffs>[TF])"),
                       ]
             ),
               SM(name = 'charge_multiplicity_atoms',
               sections = ['section_system'],
                  startReStr = r"\s*ATOM      ATOMIC",
                  forwardMatch = True,
                  subMatchers = [
                      SM(r"\s*ATOM      ATOMIC"),
                      SM(r"\s*[A-Z0-9?]+\s+(?P<x_gamess_atomic_number>\d+\.\d)\s+(?P<x_gamess_atom_x_coord_initial__bohr>[-+0-9.]+)\s+(?P<x_gamess_atom_y_coord_initial__bohr>[-+0-9.]+)\s+(?P<x_gamess_atom_z_coord_initial__bohr>[-+0-9.]+)",repeats = True),
                      SM(r"\s*INTERNUCLEAR DISTANCES"),
                      SM(r"\s*NUMBER OF ELECTRONS\s*=\s*(?P<x_gamess_number_of_electrons>[0-9]+)"),
                      SM(r"\s*CHARGE OF MOLECULE\s*=\s*(?P<x_gamess_total_charge>[0-9-]+)"),
                      SM(r"\s*SPIN MULTIPLICITY\s*=\s*(?P<x_gamess_spin_target_multiplicity>[0-9]+)"),
                      SM(r"\s*TOTAL NUMBER OF ATOMS\s*=\s*(?P<number_of_atoms>[0-9]+)"),
                      ]
             ),
               SM (name = 'SectionMethodElecStruc',
               sections = ['section_method'],
                   startReStr = r"\s*SCFTYP=",
                   forwardMatch = True,
                   subMatchers = [
                       SM(r"\s*SCFTYP=(?P<x_gamess_scf_type>[A-Z]+)\s*RUNTYP=(?P<x_gamess_comp_method>[0-9a-zA-Z]+)\s*EXETYP=[A-Z]+"),
                       SM(r"\s*MPLEVL=\s*(?P<x_gamess_mplevel>[0-9])\s*CITYP =(?P<x_gamess_citype>[A-Z]+)\s*CCTYP =(?P<x_gamess_cctype>[-A-Z]+)\s*VBTYP =(?P<x_gamess_vbtype>[A-Z]+)"),
                       SM(r"\s*DFTTYP=(?P<XC_functional>[-0-9A-Z]+)\s*TDDFT =(?P<x_gamess_tddfttype>[A-Z]+)"),
                       SM(r"\s*PP    =(?P<x_gamess_pptype>[A-Z]+)\s*RELWFN=(?P<x_gamess_relatmethod>[A-Z]+)"),
                       ]
             ),
            SM (name = 'SingleConfigurationCalculationWithSystemDescription',
                startReStr = r"\s*\$SYSTEM OPTIONS|\s*COORDINATES OF ALL ATOMS",
                repeats = False,
                forwardMatch = True,
                subMatchers = [
                SM (name = 'SingleConfigurationCalculation',
                  startReStr = r"\s*\$SYSTEM OPTIONS|\s*COORDINATES OF ALL ATOMS",
                  repeats = True,
                  forwardMatch = False,
                  sections = ['section_single_configuration_calculation'],
                  subMatchers = [
                  SM(name = 'geometry',
                   sections  = ['x_gamess_section_geometry'],
                   startReStr = r"\s*COORDINATES OF ALL ATOMS",
                   endReStr = r"\s*THE CURRENT FULLY SUBSTITUTED Z-MATRIX IS",
                      subMatchers = [
                      SM(r"\s*[A-Z]+\s+[0-9.]+\s+(?P<x_gamess_atom_x_coord__angstrom>[-+0-9.]+)\s+(?P<x_gamess_atom_y_coord__angstrom>[-+0-9.]+)\s+(?P<x_gamess_atom_z_coord__angstrom>[-+0-9.]+)",repeats = True),
                      SM(r"\s*THE CURRENT FULLY SUBSTITUTED Z-MATRIX IS"),
                    ]
                ),
                  SM(name = 'TotalEnergyScfGamess',
                   sections  = ['section_scf_iteration'],
                    startReStr = r"\s*[-A-Z0-9]+\s*SCF CALCULATION",
                    forwardMatch = False,
                    subMatchers = [
                     SM(r"ITER EX DEM     TOTAL ENERGY"),
                     SM(r"(\s+[0-9^.]+)(\s+[0-9^.]+)(\s+[0-9^.])\s*(?P<x_gamess_energy_total_scf_iteration__hartree>(-\d+\.\d{10}))\s*[-+0-9.]+\s*[-+0-9.]+\s*[-+0-9.]+",repeats = True),
                     SM(r"\s*(?P<single_configuration_calculation_converged>DENSITY CONVERGED)"),
                     SM(r"\s*FINAL\s*[-A-Z0-9]+\s*ENERGY IS\s*(?P<x_gamess_energy_scf__hartree>[-+0-9.]+)"),
                    ]
                ),
                    SM(name = 'OrbitalEnergies',
                    sections = ['section_eigenvalues'],
                    startReStr = r"\s*EIGENVECTORS",
                    endReStr = r"\s*UHF NATURAL ORBITALS AND OCCUPATION NUMBERS|MULLIKEN AND LOWDIN POPULATION ANALYSES",
                    forwardMatch = False,
                    subFlags = SM.SubFlags.Sequenced,
                    subMatchers = [
                          SM(r"\s*(?P<x_gamess_alpha_eigenvalues_values>\s*(-?\d+\.\d{4})\s{2,}(-?\d+\.\d{4})?\s{2,}(-?\d+\.\d{4})?\s{2,}(-?\d+\.\d{4})?\s{2,}(-?\d+\.\d{4})?)", repeats = True),
                          SM(r"\s*\-\-\-\-\-\s*BETA SET"),
                          SM(r"\s*(?P<x_gamess_beta_eigenvalues_values>\s*(-?\d+\.\d{4})\s{2,}(-?\d+\.\d{4})?\s{2,}(-?\d+\.\d{4})?\s{2,}(-?\d+\.\d{4})?\s{2,}(-?\d+\.\d{4})?)", repeats = True),
                     ]
                ),
                    SM(name = 'PerturbationEnergies',
                    sections = ['x_gamess_section_moller_plesset'],
                    startReStr = r"\s*RESULTS OF MOLLER-PLESSET 2ND ORDER CORRECTION ARE|\s*DISTRIBUTED DATA MP2 GRADIENT",
                    forwardMatch = False,
                    subMatchers = [
                     SM(r"\s*E\(MP2\)=\s*(?P<energy_total__hartree>[-0-9.]+)"),
                     ]
                ),
                    SM(name = 'GroundStateCoupledClusterEnergies',
                    sections = ['x_gamess_section_coupled_cluster'],
                    startReStr = r"\s*.......\s*DONE WITH CC AMPLITUDE ITERATIONS\s*",
                    forwardMatch = False,
                    subMatchers = [
                     SM(r"\s*CCSD    ENERGY:\s*(?P<energy_total__hartree>[-+0-9.]+)"),
                     SM(r"\s*CR-CC(2,3) OR CR-CCSD(T)_L ENERGY:\s*(?P<energy_total__hartree>[-+0-9.]+)"),
                     ]
                ),
                    SM(name = 'MCSCFStates',
                    sections = ['x_gamess_section_mcscf'],
                    startReStr = r"\s*MCSCF CALCULATION",
                    forwardMatch = False,
                    subFlags = SM.SubFlags.Sequenced,
                    subMatchers = [
                     SM(r"\s*NUMBER OF CORE ORBITALS          =\s*(?P<x_gamess_mcscf_inactive_orbitals>[0-9]+)"),
                     SM(r"\s*THE MAXIMUM ELECTRON EXCITATION WILL BE\s*(?P<x_gamess_mcscf_active_electrons>[0-9]+)"),
                     SM(r"\s*SYMMETRIES FOR THE\s*(?P<x_gamess_mcscf_inactive_orbitals>[0-9]+)\s*CORE,\s*(?P<x_gamess_mcscf_active_orbitals>[0-9]+)\s*ACTIVE"),
                     SM(r"\s*NUMBER OF ACTIVE ORBITALS        =\s*(?P<x_gamess_mcscf_active_orbitals>[0-9]+)"),
                     SM(r"\s*NUMBER\s*OF\s*[A-Z]+\s*ELECTRONS\s*=\s*[0-9]+\s*\(\s*(?P<x_gamess_mcscf_active_electrons>[0-9]+)", repeats = True),
                     SM(r"\s*STATE #\s*[0-9]+\s*ENERGY =\s*[-+0-9.]+", repeats = True),
                     SM(r"\s*STATE\s*[0-9]+\s*ENERGY=\s*", repeats = True),
                     SM(r"\s*ITER\s*TOTAL\s*ENERGY"),
                     SM(r"\s*[0-9^.]+\s*(?P<x_gamess_energy_mcscf_iteration__hartree>(-\d+\.\d{9}))", repeats = True),
                     SM(r"\s*(?P<single_configuration_calculation_converged>ENERGY CONVERGED)"),
                     SM(r"\s*(?P<single_configuration_calculation_converged>LAGRANGIAN CONVERGED)"),
                     SM(r"\s*STATE #\s*[0-9]+\s*ENERGY =\s*(?P<energy_total__hartree>[-+0-9.]+)", repeats = True),
                     SM(r"\s*STATE\s*[0-9]+\s*ENERGY=\s*(?P<energy_total__hartree>[-+0-9.]+)", repeats = True),
                     ]
                ),
                    SM(name = 'OrbitalEnergies',
                    sections = ['section_eigenvalues'],
                    startReStr = r"\s*MCSCF OPTIMIZED ORBITALS",
                    endReStr = r"\s*MULLIKEN AND LOWDIN POPULATION ANALYSES",
                    forwardMatch = False,
                    subFlags = SM.SubFlags.Sequenced,
                    subMatchers = [
                          SM(r"\s*(?P<x_gamess_alpha_eigenvalues_values>\s*(-?\d+\.\d{4})\s{2,}(-?\d+\.\d{4})?\s{2,}(-?\d+\.\d{4})?\s{2,}(-?\d+\.\d{4})?\s{2,}(-?\d+\.\d{4})?)", repeats = True),
                          SM(r"\s*\-\-\-\-\-\s*BETA SET"),
                          SM(r"\s*(?P<x_gamess_beta_eigenvalues_values>\s*(-?\d+\.\d{4})\s{2,}(-?\d+\.\d{4})?\s{2,}(-?\d+\.\d{4})?\s{2,}(-?\d+\.\d{4})?\s{2,}(-?\d+\.\d{4})?)", repeats = True),
                     ]
                ),
                    SM(name = 'MRPT2States',
                    sections = ['x_gamess_section_mrpt2'],
                    startReStr = r"\s*MC-QDPT2|\s*DETERMINANTAL MULTIREFERENCE 2ND ORDER PERTURBATION THEORY",
                    forwardMatch = False,
                    subMatchers = [
                     SM(r"\s*# OF FROZEN CORE ORBITALS     =\s*(?P<x_gamess_mrpt2_frozen_core_orbitals>[0-9]+)"),
                     SM(r"\s*# OF DOUBLY OCCUPIED ORBITALS =\s*(?P<x_gamess_mrpt2_doubly_occupied_orbitals>[0-9]+)"),
                     SM(r"\s*# OF ACTIVE ORBITALS          =\s*(?P<x_gamess_mrpt2_active_orbitals>[0-9]+)"),
                     SM(r"\s*# OF EXTERNAL ORBITALS        =\s*(?P<x_gamess_mrpt2_external_orbitals>[0-9]+)"),
                     SM(r"\s*# OF FROZEN VIRTUAL ORBITALS  =\s*(?P<x_gamess_mrpt2_frozen_virtual_orbitals>[0-9]+)"),
                     SM(r"\s*NUMBER OF CORE     ORBITALS      =\s*(?P<x_gamess_mrpt2_frozen_core_orbitals>[0-9]+)"),
                     SM(r"\s*NUMBER OF VALENCE  ORBITALS      =\s*(?P<x_gamess_mrpt2_doubly_occupied_orbitals>[0-9]+)"),
                     SM(r"\s*NUMBER OF ACTIVE   ORBITALS      =\s*(?P<x_gamess_mrpt2_active_orbitals>[0-9]+)"),
                     SM(r"\s*NUMBER OF EXTERNAL ORBITALS      =\s*(?P<x_gamess_mrpt2_external_orbitals>[0-9]+)"),
                     SM(r"\s*NUMBER OF ALPHA ELECTRONS\s*=\s*[0-9]+\s*\(\s*[0-9]+\s*VALENCE\)\s*\(\s*(?P<x_gamess_mrpt2_active_electrons>[0-9]+)"),
                     SM(r"\s*NUMBER OF BETA ELECTRONS\s*=\s*[0-9]+\s*\(\s*[0-9]+\s*VALENCE\)\s*\(\s*(?P<x_gamess_mrpt2_active_electrons>[0-9]+)"),
                     SM(r"\s*###   MRMP2 RESULTS"),
                     SM(r"\s*AMES LABORATORY DETERMINANTAL FULL CI"),
                     SM(r"\s*THE DETERMINANT\s*(?P<x_gamess_mrpt2_method_type>([MRPT]+))"),
                     SM(r"\s*\*\*\*\s*(?P<x_gamess_mrpt2_method_type>[0-9A-Z]+)\s*ENERGY"),
                     SM(r"\s*[0-9]+\s*E\(MCSCF\)=\s*[-0-9.]+\s*E\(MP2\)=\s*(?P<energy_total__hartree>[-0-9.]+)", repeats = True),
                     SM(r"\s*TOTAL MRPT2, E\(MP2\) 0TH\s*\+\s*1ST\s*\+\s*2ND ORDER ENERGY =\s*(?P<energy_total__hartree>[-0-9.]+)", repeats = True),
                     ]
                ),
                    SM(name = 'TDDFTStatesSection',
                    sections = ['x_gamess_section_excited_states'],
                    startReStr = r"\s*SINGLET EXCITATIONS|\s*PRINTING CIS COEFFICIENTS",
                    forwardMatch = False,
                    repeats = False,
                    subMatchers = [
                       SM(name = 'TDDFTStates',
                       sections = ['x_gamess_section_tddft'],
                       startReStr = r"\s*STATE #\s*[0-9]+\s*ENERGY =",
                       forwardMatch = True,
                       repeats = True,
                       subMatchers = [
                        SM(r"\s*STATE #\s*[0-9]+\s*ENERGY =\s*(?P<x_gamess_tddft_excitation_energy__eV>[-0-9.]+)"),
                        SM(r"\s*OSCILLATOR STRENGTH =\s*(?P<x_gamess_tddft_oscillator_strength>[0-9.]+)"),
                        ]
                   ),
                       SM(name = 'CISStates',
                       sections = ['x_gamess_section_cis'],
                       startReStr = r"\s*CI-SINGLES EXCITATION ENERGIES",
                       forwardMatch = False,
                       repeats = True,
                       subMatchers = [
                        SM(r"\s*[0-9A-Z']+\s*(?P<x_gamess_cis_excitation_energy__hartree>[0-9.]+)"),
                        SM(r"\s*OSCILLATOR STRENGTH =\s*(?P<x_gamess_cis_oscillator_strength>[0-9.]+)"),
                        ]
                   ),
                        ]
                   ),
                       SM(name = 'EnergyGradients',
                       sections = ['x_gamess_section_atom_forces'],
                       startReStr = r"\s*GRADIENT \(HARTREE|\s*UNITS ARE HARTREE",
                       forwardMatch = False,
                       repeats = True,
                       subMatchers = [
                        SM(r"\s*[0-9]+\s*[A-Z0-9]+\s*(\d+\.\d{1})\s*(?P<x_gamess_atom_x_force>[+-.0-9]+)\s*(?P<x_gamess_atom_y_force>[+-.0-9]+)\s*(?P<x_gamess_atom_z_force>[+-.0-9]+)", repeats = True),
                        SM(r"\s*[0-9]+\s*[A-Z0-9]+\s*(?P<x_gamess_atom_x_force>[+-.0-9]+)\s*(?P<x_gamess_atom_y_force>[+-.0-9]+)\s*(?P<x_gamess_atom_z_force>[+-.0-9]+)", repeats = True),
                        SM(r"\s*MAXIMUM GRADIENT ="),
                        SM(r"\s*INTERNAL COORDINATES"),
                        ]
                   ),
                       SM(name = 'Geometry_optimization',
                       sections  = ['x_gamess_section_geometry_optimization_info'],
                       startReStr = r"\s*\*\*\*\*\*\s*EQUILIBRIUM GEOMETRY LOCATED",
                       forwardMatch = True,
                       subMatchers = [
                        SM(r"\s*\*\*\*\*\*\s*(?P<x_gamess_geometry_optimization_converged>EQUILIBRIUM GEOMETRY LOCATED)"),
                        SM(r"\s*\*\*\*\*\*\s*(?P<x_gamess_geometry_optimization_converged>FAILURE TO LOCATE STATIONARY POINT)"),
                        ]
                   ),
                       SM (name = 'Frequencies',
                       sections = ['x_gamess_section_frequencies'],
                       startReStr = r"\s*FREQUENCIES IN CM",
                       endReStr = r"\s*THERMOCHEMISTRY",
                       forwardMatch = False,
                       repeats = False,
                       subFlags = SM.SubFlags.Unordered,
                       subMatchers = [
                        SM(r"\s*FREQUENCY:\s*(?P<x_gamess_frequency_values>([0-9]+\.\d{2}\s*[A-Z]?)\s*([0-9]+\.\d{2})?\s*([0-9]+\.\d{2})?\s*([0-9]+\.\d{2})?\s*([0-9]+\.\d{2})?)", repeats = True),
                        SM(r"\s*REDUCED MASS:\s*(?P<x_gamess_reduced_masses>([0-9]+\.\d{5})\s*([0-9]+\.\d{5})?\s*([0-9]+\.\d{5})?\s*([0-9]+\.\d{5})?\s*([0-9]+\.\d{5})?)", repeats = True),
                         ]
                   ),

          ])
        ])
      ])
    ])

parserInfo = {
  "name": "parser_gamess",
  "version": "1.0"
}

class GAMESSParserContext(object):
      """Context for parsing GAMESS output file.

        This class keeps tracks of several GAMESS settings to adjust the parsing to them.
        The onClose_ functions allow processing and writing of cached values after a section is closed.
        They take the following arguments:
        backend: Class that takes care of writing and caching of metadata.
        gIndex: Index of the section that is closed.
        section: The cached values and sections that were found in the section that is closed.
      """
      def __init__(self):
        # dictionary of energy values, which are tracked between SCF iterations and written after convergence
        self.totalEnergyList = {
                               }
        self.skip_system_onclose = False

      def initialize_values(self):
        """Initializes the values of certain variables.

        This allows a consistent setting and resetting of the variables,
        when the parsing starts and when a section_run closes.
        """
        self.secMethodIndex = None
        self.secSystemDescriptionIndex = None
        # start with -1 since zeroth iteration is the initialization
        self.scfIterNr = -1
        self.singleConfCalcs = []
        self.scfConvergence = False
        self.geoConvergence = False
        self.scfenergyconverged = 0.0
        self.periodicCalc = False

      def startedParsing(self, path, parser):
        self.parser = parser
        # save metadata
        self.metaInfoEnv = self.parser.parserBuilder.metaInfoEnv
        # allows to reset values if the same superContext is used to parse different files
        self.initialize_values()

      def onClose_section_run(self, backend, gIndex, section):
          """Trigger called when section_run is closed.

          Write convergence of geometry optimization.
          Variables are reset to ensure clean start for new run.
          """
          # write geometry optimization convergence
          gIndexTmp = backend.openSection('section_frame_sequence')
          backend.addValue('geometry_optimization_converged', self.geoConvergence)
          backend.closeSection('section_frame_sequence', gIndexTmp)
          # frame sequence
          if self.geoConvergence:
              sampling_method = "geometry_optimization"
          elif len(self.singleConfCalcs) > 1:
              pass # to do
          else:
              return
          samplingGIndex = backend.openSection("section_sampling_method")
          backend.addValue("sampling_method", sampling_method)
          backend.closeSection("section_sampling_method", samplingGIndex)
          frameSequenceGIndex = backend.openSection("section_frame_sequence")
          backend.addValue("frame_sequence_to_sampling_ref", samplingGIndex)
          backend.addArrayValues("frame_sequence_local_frames_ref", np.asarray(self.singleConfCalcs))
          backend.closeSection("section_frame_sequence", frameSequenceGIndex)
          # reset all variables
          self.initialize_values()

      def onOpen_section_single_configuration_calculation(self, backend, gIndex, section):
          self.singleConfCalcs.append(gIndex)

      def onClose_x_gamess_section_geometry(self, backend, gIndex, section):

         if(section["x_gamess_atom_x_coord"]):
            xCoord = section["x_gamess_atom_x_coord"]
            yCoord = section["x_gamess_atom_y_coord"]
            zCoord = section["x_gamess_atom_z_coord"]
            atom_coords = np.zeros((len(xCoord),3), dtype=float)
            atom_numbers = np.zeros(len(xCoord), dtype=int)
            for i in range(len(xCoord)):
               atom_coords[i,0] = xCoord[i]
               atom_coords[i,1] = yCoord[i]
               atom_coords[i,2] = zCoord[i]
            gIndexTmp = backend.openSection("section_system")
            backend.addArrayValues("atom_positions", atom_coords)
            self.skip_system_onclose = True
            backend.closeSection("section_system", gIndexTmp)
            self.skip_system_onclose = False

      def onClose_x_gamess_section_atom_forces(self, backend, gIndex, section):
        xForce = section["x_gamess_atom_x_force"]
        yForce = section["x_gamess_atom_y_force"]
        zForce = section["x_gamess_atom_z_force"]
        atom_forces = np.zeros((len(xForce),3), dtype=float)
        for i in range(len(xForce)):
           atom_forces[i,0] = -xForce[i]
           atom_forces[i,1] = -yForce[i]
           atom_forces[i,2] = -zForce[i]
        backend.addArrayValues("atom_forces_raw", atom_forces)

      def onOpen_section_system(self, backend, gIndex, section):
          # keep track of the latest system description section
          self.secSystemDescriptionIndex = gIndex

      def onClose_section_system(self, backend, gIndex, section):
        if self.skip_system_onclose:
          return
        if(section["x_gamess_atom_x_coord_initial"]):
          xCoord = section["x_gamess_atom_x_coord_initial"]
          yCoord = section["x_gamess_atom_y_coord_initial"]
          zCoord = section["x_gamess_atom_z_coord_initial"]
          numbers = section["x_gamess_atomic_number"]
          atom_coords = np.zeros((len(xCoord),3), dtype=float)
          atom_numbers = np.zeros(len(xCoord), dtype=int)
          atomic_symbols = np.empty((len(xCoord)), dtype=object)
          for i in range(len(xCoord)):
             atom_coords[i,0] = xCoord[i]
             atom_coords[i,1] = yCoord[i]
             atom_coords[i,2] = zCoord[i]
          for i in range(len(xCoord)):
            atom_numbers[i] = numbers[i]
            atomic_symbols[i] = ase.data.chemical_symbols[atom_numbers[i]]
          backend.addArrayValues("atom_labels", atomic_symbols)
          backend.addArrayValues("atom_positions", atom_coords)

      def onOpen_section_method(self, backend, gIndex, section):
        # keep track of the latest method section
        self.secMethodIndex = gIndex

      def onClose_section_method(self, backend, gIndex, section):
       # handling of xc functional
       # Dictionary for conversion of xc functional name in Gaussian to metadata format.
       # The individual x and c components of the functional are given as dictionaries.
       # Possible key of such a dictionary is 'name'.

       #density functionals

       xcDict = {
              'SLATER':       [{'name': 'LDA_X'}],
              'VWN':          [{'name': 'LDA_C_VWN_5'}],
              'VWN3':         [{'name': 'LDA_C_VWN_3'}],
              'VWN1RPA':      [{'name': 'LDA_C_VWN1RPA'}],
              'BECKE':        [{'name': 'GGA_X_B88'}],
              'OPTX':         [{'name': 'GGA_X_OPTX'}],
              'GILL':         [{'name': 'GGA_X_G96'}],
              'PW91X':        [{'name': 'GGA_X_PW91'}],
              'PBEX':         [{'name': 'GGA_X_PBE'}],
              'PZ81':         [{'name': 'GGA_C_PZ'}],
              'P86':          [{'name': 'GGA_C_P86'}],
              'LYP':          [{'name': 'GGA_C_LYP'}],
              'PW91C':        [{'name': 'GGA_C_PW91'}],
              'PBEC':         [{'name': 'GGA_C_PBE'}],
              'OP':           [{'name': 'GGA_C_OP'}],
              'SVWN':         [{'name': 'LDA_C_VWN_5'}, {'name': 'LDA_X'}],
              'SVWN1RPA':     [{'name': 'LDA_C_VWN1RPA'}, {'name': 'LDA_X'}],
              'SVWN3':        [{'name': 'LDA_C_VWN_3'}, {'name': 'LDA_X'}],
              'SPZ81':        [{'name': 'GGA_C_PZ'}, {'name': 'LDA_X'}],
              'SP86':         [{'name': 'GGA_C_P86'}, {'name': 'LDA_X'}],
              'SLYP':         [{'name': 'GGA_C_LYP'}, {'name': 'LDA_X'}],
              'SPW91':        [{'name': 'GGA_C_PW91'}, {'name': 'LDA_X'}],
              'SPBE':         [{'name': 'GGA_C_PBE'}, {'name': 'LDA_X'}],
              'SOP':          [{'name': 'GGA_C_OP'}, {'name': 'LDA_X'}],
              'BVWN':         [{'name': 'LDA_C_VWN_5'}, {'name': 'LDA_X_B88'}],
              'BVWN1RPA':     [{'name': 'LDA_C_VWN1RPA'}, {'name': 'LDA_X_B88'}],
              'BVWN3':        [{'name': 'LDA_C_VWN_3'}, {'name': 'LDA_X_B88'}],
              'BPZ81':        [{'name': 'GGA_C_PZ'}, {'name': 'LDA_X_B88'}],
              'BP86':         [{'name': 'GGA_C_P86'}, {'name': 'LDA_X_B88'}],
              'BLYP':         [{'name': 'GGA_C_LYP'}, {'name': 'LDA_X_B88'}],
              'BPW91':        [{'name': 'GGA_C_PW91'}, {'name': 'LDA_X_B88'}],
              'BPBE':         [{'name': 'GGA_C_PBE'}, {'name': 'LDA_X_B88'}],
              'BOP':          [{'name': 'GGA_C_OP'}, {'name': 'LDA_X_B88'}],
              'GVWN':         [{'name': 'LDA_C_VWN_5'}, {'name': 'GGA_X_G96'}],
              'GVWN1RPA':     [{'name': 'LDA_C_VWN1RPA'}, {'name': 'GGA_X_G96'}],
              'GVWN3':        [{'name': 'LDA_C_VWN_3'}, {'name': 'GGA_X_G96'}],
              'GPZ81':        [{'name': 'GGA_C_PZ'}, {'name': 'GGA_X_G96'}],
              'GP86':         [{'name': 'GGA_C_P86'}, {'name': 'GGA_X_G96'}],
              'GLYP':         [{'name': 'GGA_C_LYP'}, {'name': 'GGA_X_G96'}],
              'GPW91':        [{'name': 'GGA_C_PW91'}, {'name': 'GGA_X_G96'}],
              'GPBE':         [{'name': 'GGA_C_PBE'}, {'name': 'GGA_X_G96'}],
              'GOP':          [{'name': 'GGA_C_OP'}, {'name': 'GGA_X_G96'}],
              'OVWN':         [{'name': 'LDA_C_VWN_5'}, {'name': 'GGA_X_OPTX'}],
              'OVWN1RPA':     [{'name': 'LDA_C_VWN1RPA'}, {'name': 'GGA_X_OPTX'}],
              'OVWN3':        [{'name': 'LDA_C_VWN_3'}, {'name': 'GGA_X_OPTX'}],
              'OPZ81':        [{'name': 'GGA_C_PZ'}, {'name': 'GGA_X_OPTX'}],
              'OP86':         [{'name': 'GGA_C_P86'}, {'name': 'GGA_X_OPTX'}],
              'OLYP':         [{'name': 'GGA_C_LYP'}, {'name': 'GGA_X_OPTX'}],
              'OPW91':        [{'name': 'GGA_C_PW91'}, {'name': 'GGA_X_OPTX'}],
              'OPBE':         [{'name': 'GGA_C_PBE'}, {'name': 'GGA_X_OPTX'}],
              'OOP':          [{'name': 'GGA_C_OP'}, {'name': 'GGA_X_OPTX'}],
              'PW91VWN':      [{'name': 'LDA_C_VWN_5'}, {'name': 'GGA_X_PW91'}],
              'PW91VWN1RPA':  [{'name': 'LDA_C_VWN1RPA'}, {'name': 'GGA_X_PW91'}],
              'PW91VWN3':     [{'name': 'LDA_C_VWN_3'}, {'name': 'GGA_X_PW91'}],
              'PW91PZ81':     [{'name': 'GGA_C_PZ'}, {'name': 'GGA_X_PW91'}],
              'PW91P86':      [{'name': 'GGA_C_P86'}, {'name': 'GGA_X_PW91'}],
              'PW91LYP':      [{'name': 'GGA_C_LYP'}, {'name': 'GGA_X_PW91'}],
              'PW91':         [{'name': 'GGA_C_PW91'}, {'name': 'GGA_X_PW91'}],
              'PW91PBE':      [{'name': 'GGA_C_PBE'}, {'name': 'GGA_X_PW91'}],
              'PW91OP':       [{'name': 'GGA_C_OP'}, {'name': 'GGA_X_PW91'}],
              'PBEVWN':       [{'name': 'LDA_C_VWN_5'}, {'name': 'GGA_X_PBE'}],
              'PBEVWN1RPA':   [{'name': 'LDA_C_VWN1RPA'}, {'name': 'GGA_X_PBE'}],
              'PBEVWN3':      [{'name': 'LDA_C_VWN_3'}, {'name': 'GGA_X_PBE'}],
              'PBEPZ81':      [{'name': 'GGA_C_PZ'}, {'name': 'GGA_X_PBE'}],
              'PBEP86':       [{'name': 'GGA_C_P86'}, {'name': 'GGA_X_PBE'}],
              'PBELYP':       [{'name': 'GGA_C_LYP'}, {'name': 'GGA_X_PBE'}],
              'PBEPW91':      [{'name': 'GGA_C_PW91'}, {'name': 'GGA_X_PBE'}],
              'PBE':          [{'name': 'GGA_C_PBE'}, {'name': 'GGA_X_PBE'}],
              'PBEOP':        [{'name': 'GGA_C_OP'}, {'name': 'GGA_X_PBE'}],
              'EDF1':         [{'name': 'GGA_XC_EDF1'}],
              'REVPBE':       [{'name': 'GGA_XC_REVPBE'}],
              'RPBE':         [{'name': 'GGA_XC_RPBE'}],
              'PBESOL':       [{'name': 'GGA_XC_PBESOL'}],
              'HCTH93':       [{'name': 'GGA_XC_HCTH_93'}],
              'HCTH147':      [{'name': 'GGA_XC_HCTH_147'}],
              'HCTH407':      [{'name': 'GGA_XC_HCTH_407'}],
              'SOGGA':        [{'name': 'GGA_XC_SOGGA'}],
              'SOGGA11':      [{'name': 'GGA_XC_SOGGA11'}],
              'MOHLYP':       [{'name': 'GGA_XC_MOHLYP'}],
              'B97-D':        [{'name': 'GGA_XC_B97D'}],
              'BHHLYP':       [{'name': 'HYB_GGA_XC_BHANDHLYP'}],
              'B3PW91':       [{'name': 'HYB_GGA_XC_B3PW91'}],
              'B3LYP':        [{'name': 'HYB_GGA_XC_B3LYP'}],
              'B3LYPV1R':     [{'name': 'HYB_GGA_XC_B3LYPVWN1RPA'}],
              'B3LYPV3':      [{'name': 'HYB_GGA_XC_B3LYPVWN3'}],
              'B3P86':        [{'name': 'HYB_GGA_XC_B3P86'}],
              'B3P86V1R':     [{'name': 'HYB_GGA_XC_B3P86VWN1RPA'}],
              'B3P86V5':      [{'name': 'HYB_GGA_XC_B3P86VWN5'}],
              'B97':          [{'name': 'HYB_GGA_XC_B97'}],
              'B97-1':        [{'name': 'HYB_GGA_XC_B971'}],
              'B97-2':        [{'name': 'HYB_GGA_XC_B972'}],
              'B97-3':        [{'name': 'HYB_GGA_XC_B973'}],
              'B97-K':        [{'name': 'HYB_GGA_XC_B97K'}],
              'B98':          [{'name': 'HYB_GGA_XC_B98'}],
              'PBE0':         [{'name': 'HYB_GGA_XC_PBEH'}],
              'X3LYP':        [{'name': 'HYB_GGA_XC_X3LYP'}],
              'SOGGA11X':     [{'name': 'HYB_GGA_XC_SOGGA11X'}],
              'CAMB3LYP':     [{'name': 'CAM-B3LYP'}],
              'WB97':         [{'name': 'WB97'}],
              'WB97X':        [{'name': 'WB97X'}],
              'WB97X-D':      [{'name': 'WB97XD'}],
              'B2-PLYP':      [{'name': 'B2PLYP'}],
              'B2K-PLYP':     [{'name': 'B2KPLYP'}],
              'B2T-PLYP':     [{'name': 'B2TPLYP'}],
              'B2GP-PLYP':    [{'name': 'B2GPPLYP'}],
              'WB97X-2':      [{'name': 'WB97X2'}],
              'WB97X-2L':     [{'name': 'WB97X2L'}],
              'VS98':         [{'name': 'MGGA_XC_VSXC'}],
              'PKZB':         [{'name': 'MGGA_XC_PKZB'}],
              'THCTH':        [{'name': 'MGGA_XC_TAU_HCTH'}],
              'THCTHHYB':     [{'name': 'MGGA_XC_TAU_HCTHHYB'}],
              'BMK':          [{'name': 'MGGA_XC_BMK'}],
              'TPSS':         [{'name': 'MGGA_XC_TPSS'}],
              'TPSSH':        [{'name': 'MGGA_XC_TPSSHYB'}],
              'TPSSM':        [{'name': 'MGGA_XC_TPSSMOD'}],
              'REVTPSS':      [{'name': 'MGGA_XC_REVISEDTPSS'}],
              'DLDF':         [{'name': 'MGGA_XC_DLDF'}],
              'M05':          [{'name': 'HYB_MGGA_XC_M05'}],
              'M05-2X':       [{'name': 'HYB_MGGA_XC_M05_2X'}],
              'M06':          [{'name': 'HYB_MGGA_XC_M06'}],
              'M06-L':        [{'name': 'MGGA_C_M06_L'}, {'name': 'MGGA_X_M06_L'}],
              'M06-2X':       [{'name': 'HYB_MGGA_XC_M06_2X'}],
              'M06-HF':       [{'name': 'HYB_MGGA_XC_M06_HF'}],
              'M08-HX':       [{'name': 'HYB_MGGA_XC_M08_HX'}],
              'M08-SO':       [{'name': 'HYB_MGGA_XC_M08_SO'}],
              'M11-L':        [{'name': 'MGGA_C_M11_L'}, {'name': 'MGGA_X_M11_L'}],
              'M11':          [{'name': 'MGGA_C_M11_L'}, {'name': 'MGGA_X_M11'}],
              'RHF':          [{'name': 'RHF_X'}],
              'UHF':          [{'name': 'UHF_X'}],
              'ROHF':         [{'name': 'ROHF_X'}],
              'LCBOPLRD':     [{'name': 'HYB_GGA_XC_HSE03'}],
              'XALPHA':       [{'name': 'XALPHA_X_GRIDFREE'}],
              'DEPRISTO':     [{'name': 'DEPRISTO_X_GRIDFREE'}],
              'CAMA':         [{'name': 'CAMA_X_GRIDFREE'}],
              'HALF':         [{'name': 'HYB_GGA_XC_HALF_GRIDFREE'}],
              'VWN':          [{'name': 'LDA_C_VWN5_GRIDFREE'}],
              'PWLOC':        [{'name': 'PWLOC_C_GRIDFREE'}],
              'BPWLOC':       [{'name': 'PWLOC_C_GRIDFREE'}, {'name': 'GGA_X_B88_GRIDFREE'}],
              'CAMB':         [{'name': 'GGA_C_CAMBRIDGE_GRIDFREE'}, {'name': 'GGA_X_CAMA_GRIDFREE'}],
              'XVWN':         [{'name': 'LDA_C_VWN5_GRIDFREE'}, {'name': 'XALPHA_X_GRIDFREE'}],
              'XPWLOC':       [{'name': 'GGA_C_PW91_GRIDFREE'}, {'name': 'XALPHA_X_GRIDFREE'}],
              'SPWLOC':       [{'name': 'PWLOC_C_GRIDFREE'}, {'name': 'LDA_X_GRIDFREE'}],
              'WIGNER':       [{'name': 'GGA_XC_WIGNER_GRIDFREE'}],
              'WS':           [{'name': 'GGA_XC_WIGNER_GRIDFREE'}],
              'WIGEXP':       [{'name': 'GGA_XC_WIGNER_GRIDFREE'}],
              'NONE':         [{'name': 'NONE'}],
             }

       methodDict = {
              'RHF':       [{'name': 'RHF'}],
              'UHF':       [{'name': 'UHF'}],
              'ROHF':      [{'name': 'ROHF'}],
              'GVB':       [{'name': 'GVB'}],
              'MCSCF':     [{'name': 'MCSCF'}],
              'EXCITE':    [{'name': 'TDDFT'}],
              'SPNFLP':    [{'name': 'SF-TDDFT'}],
              'POL':       [{'name': 'HYPERPOL'}],
              'VB2000':    [{'name': 'VB'}],
              'CIS':       [{'name': 'CIS'}],
              'SFCIS':     [{'name': 'SF-CIS'}],
              'ALDET':     [{'name': 'DET-MCSCF'}],
              'ORMAS':     [{'name': 'ORMAS'}],
              'FSOCI':     [{'name': 'SECONDORDER-CI'}],
              'GENCI':     [{'name': 'GENERAL-CI'}],
              'LCCD':      [{'name': 'L-CCSD'}],
              'CCD':       [{'name': 'CCD'}],
              'CCSD':      [{'name': 'CCSD'}],
              'CCSD(T)':   [{'name': 'CCSD(T)'}],
              'R-CC':      [{'name': 'R-CCSD(T)&R-CCSD[T]'}],
              'CR-CC':     [{'name': 'CR-CCSD(T)&CR-CCSD[T]'}],
              'CR-CCL':    [{'name': 'CR-CC(2,3)'}],
              'CCSD(TQ)':  [{'name': 'CCSD(TQ)&R-CCSD(TQ)'}],
              'CR-CC(Q)':  [{'name': 'CR-CCSD(TQ)'}],
              'EOM-CCSD':  [{'name': 'EOM-CCSD'}],
              'CR-EOM':    [{'name': 'CR-EOMCCSD(T)'}],
              'CR-EOML':   [{'name': 'CR-EOMCC(2,3)'}],
              'IP-EOM2':   [{'name': 'IP-EOMCCSD'}],
              'IP-EOM3A':  [{'name': 'IP-EOMCCSDt'}],
              'EA-EOM2':   [{'name': 'EA-EOMCCSD'}],
              'EA-EOM3A':  [{'name': 'EA-EOMCCSDt'}],
              'IOTC':      [{'name': 'SCALAR_RELATIVISTIC'}],
              'DK':        [{'name': 'SCALAR_RELATIVISTIC'}],
              'RESC':      [{'name': 'SCALAR_RELATIVISTIC'}],
              'NESC':      [{'name': 'SCALAR_RELATIVISTIC'}],
              'SBKJC':     [{'name': 'PSEUDO_SCALAR_RELATIVISTIC'}],
              'HW':        [{'name': 'PSEUDO_SCALAR_RELATIVISTIC'}],
              'MCP':       [{'name': 'PSEUDO_SCALAR_RELATIVISTIC'}],
              'TAMMD':     [{'name': 'TAMM-DANCOFF'}],
              'DFTB':      [{'name': 'DFTB'}],
              'MP2':       [{'name': 'MP2'}],
              'RIMP2':     [{'name': 'RESOLUTIONOFIDENTITY-MP2'}],
              'CPHF':      [{'name': 'COUPLEDPERTURBED-HF'}],
              'G3MP2':     [{'name': 'G3(MP2)'}],
              'G32CCSD':   [{'name': 'G3(MP2,CCSD(T))'}],
              'G4MP2':     [{'name': 'G4(MP2)'}],
              'G4MP2-6X':  [{'name': 'G4(MP2)-6X'}],
              'CCCA-S4':   [{'name': 'CCCA-S4'}],
              'CCCA-CCL':  [{'name': 'CCCA-CC(2,3)'}],
              'MCQDPT':    [{'name': 'MCQDPT2'}],
              'DETMRPT':   [{'name': 'MRPT2'}],
              'FORS':      [{'name': 'CASSCF'}],
              'FOCI':      [{'name': 'FIRSTORDER-CI'}],
              'SOCI':      [{'name': 'SECONDORDER-CI'}],
              'DM':        [{'name': 'TRANSITIONMOMENTS'}],
              'HSO1':      [{'name': 'ONEELEC-SOC'}],
              'HSO2P':     [{'name': 'PARTIALTWOELEC-SOC'}],
              'HSO2':      [{'name': 'TWOELEC-SOC'}],
              'HSO2FF':    [{'name': 'TWOELECFORMFACTOR-SOC'}],
              'GUGA':      [{'name': 'GUGA-CI'}],
              'NONE':      [{'name': 'NOCORRELATEDMETHOD'}],
             }

       basissetDict = {
              'STO':          [{'name': 'STO-2G'}],
              'STO':          [{'name': 'STO-3G'}],
              'STO':          [{'name': 'STO-4G'}],
              'STO':          [{'name': 'STO-5G'}],
              'STO':          [{'name': 'STO-6G'}],
              'N21':          [{'name': 'N21'}],
              'N31':          [{'name': 'N31'}],
              'N31':          [{'name': 'N31'}],
              'N311':         [{'name': 'N311'}],
              'N21':          [{'name': 'N21'}],
              'N31':          [{'name': 'N31'}],
              'N311':         [{'name': 'N311'}],
              'MINI':         [{'name': 'MINI'}],
              'MIDI':         [{'name': 'MIDI'}],
              'DZV':          [{'name': 'DZV'}],
              'DH':           [{'name': 'DH'}],
              'TZV':          [{'name': 'TZV'}],
              'MC':           [{'name': 'MC'}],
              'G3L':          [{'name': 'G3MP2LARGE'}],
              'G3LX':         [{'name': 'G3MP2LARGEXP'}],
              'CCD':          [{'name': 'CC_PVDZ'}],
              'CCT':          [{'name': 'CC_PVTZ'}],
              'CCQ':          [{'name': 'CC_PVQZ'}],
              'CC5':          [{'name': 'CC_PV5Z'}],
              'CC6':          [{'name': 'CC_PV6Z'}],
              'ACCD':         [{'name': 'AUG-CC-PVDZ'}],
              'ACCT':         [{'name': 'AUG-CC-PVTZ'}],
              'ACCQ':         [{'name': 'AUG-CC-PVQZ'}],
              'ACC5':         [{'name': 'AUG-CC-PV5Z'}],
              'ACC6':         [{'name': 'AUG-CC-PV6Z'}],
              'CCDC':         [{'name': 'CC-PCVDZ'}],
              'CCTC':         [{'name': 'CC-PCVTZ'}],
              'CCQC':         [{'name': 'CC-PCVQZ'}],
              'CC5C':         [{'name': 'CC-PCV5Z'}],
              'CC6C':         [{'name': 'CC-PCV6Z'}],
              'ACCDC':        [{'name': 'AUG-CC-PCVDZ'}],
              'ACCTC':        [{'name': 'AUG-CC-PCVTZ'}],
              'ACCQC':        [{'name': 'AUG-CC-PCVQZ'}],
              'ACC5C':        [{'name': 'AUG-CC-PCV5Z'}],
              'ACC6C':        [{'name': 'AUG-CC-PCV6Z'}],
              'CCDWC':        [{'name': 'CC-PWCVDZ'}],
              'CCTWC':        [{'name': 'CC-PWCVTZ'}],
              'CCQWC':        [{'name': 'CC-PWCVQZ'}],
              'CC5WC':        [{'name': 'CC-PWCV5Z'}],
              'CC6WC':        [{'name': 'CC-PWCV6Z'}],
              'ACCDWC':       [{'name': 'AUG-CC-PWCVDZ'}],
              'ACCTWC':       [{'name': 'AUG-CC-PWCVTZ'}],
              'ACCQWC':       [{'name': 'AUG-CC-PWCVQZ'}],
              'ACC5WC':       [{'name': 'AUG-CC-PWCV5Z'}],
              'ACC6WC':       [{'name': 'AUG-CC-PWCV6Z'}],
              'PCSEG-0':      [{'name': 'PCSEG-0'}],
              'PCSEG-1':      [{'name': 'PCSEG-1'}],
              'PCSEG-2':      [{'name': 'PCSEG-2'}],
              'PCSEG-3':      [{'name': 'PCSEG-3'}],
              'PCSEG-4':      [{'name': 'PCSEG-4'}],
              'APCSEG-0':     [{'name': 'AUG-PCSEG-0'}],
              'APCSEG-1':     [{'name': 'AUG-PCSEG-1'}],
              'APCSEG-2':     [{'name': 'AUG-PCSEG-2'}],
              'APCSEG-3':     [{'name': 'AUG-PCSEG-3'}],
              'APCSEG-4':     [{'name': 'AUG-PCSEG-4'}],
              'SPK-DZP':      [{'name': 'SPK-DZP'}],
              'SPK-DTP':      [{'name': 'SPK-DTP'}],
              'SPK-DQP':      [{'name': 'SPK-DQP'}],
              'SPK-ADZP':     [{'name': 'AUG-SPK-DZP'}],
              'SPK-ATZP':     [{'name': 'AUG-SPK-TZP'}],
              'SPK-AQZP':     [{'name': 'AUG-SPK-QZP'}],
              'SPKRDZP':      [{'name': 'SPK-RELDZP'}],
              'SPKRDTP':      [{'name': 'SPK-RELDTP'}],
              'SPKRDQP':      [{'name': 'SPK-RELDQP'}],
              'SPKRADZP':     [{'name': 'AUG-SPK-RELDZP'}],
              'SPKRATZP':     [{'name': 'AUG-SPK-RELTZP'}],
              'SPKRAQZP':     [{'name': 'AUG-SPK-RELQZP'}],
              'SPK-DZC':      [{'name': 'SPK-DZC'}],
              'SPK-TZC':      [{'name': 'SPK-TZC'}],
              'SPK-QZC':      [{'name': 'SPK-QZC'}],
              'SPK-DZCD':     [{'name': 'SPK-DZCD'}],
              'SPK-TZCD':     [{'name': 'SPK-TZCD'}],
              'SPK-QZCD':     [{'name': 'SPK-QZCD'}],
              'SPKRDZC':      [{'name': 'SPK-RELDZC'}],
              'SPKRTZC':      [{'name': 'SPK-RELTZC'}],
              'SPKRQZC':      [{'name': 'SPK-RELQZC'}],
              'SPKRDZCD':     [{'name': 'SPK-RELDZCD'}],
              'SPKRTZCD':     [{'name': 'SPK-RELTZCD'}],
              'SPKRQZCD':     [{'name': 'SPK-RELQZCD'}],
              'KTZV':         [{'name': 'KARLSRUHETZV'}],
              'KTZVP':        [{'name': 'KARLSRUHETZVP'}],
              'KTZVPP':       [{'name': 'KARLSRUHETZVPP'}],
              'MCP-DZP':      [{'name': 'MCP-DZP'}],
              'MCP-TZP':      [{'name': 'MCP-TZP'}],
              'MCP-QZP':      [{'name': 'MCP-QZP'}],
              'MCP-ATZP':     [{'name': 'AUG-MCP-TZP'}],
              'MCP-AQZP':     [{'name': 'AUG-MCP-QZP'}],
              'MCPCDZP':      [{'name': 'MCPCDZP'}],
              'MCPCTZP':      [{'name': 'MCPCTZP'}],
              'MCPCQZP':      [{'name': 'MCPCQZP'}],
              'MCPACDZP':     [{'name': 'AUG-MCPCDZP'}],
              'MCPACTZP':     [{'name': 'AUG-MCPCTZP'}],
              'MCPACQZP':     [{'name': 'AUG-MCPCQZP'}],
              'IMCP-SR1':     [{'name': 'IMPROVEDMCP-SCALREL1'}],
              'IMCP-SR2':     [{'name': 'IMPROVEDMCP-SCALREL2'}],
              'ZFK3-DK3':     [{'name': 'ZFK3-DK3'}],
              'ZFK4-DK3':     [{'name': 'ZFK4-DK3'}],
              'ZFK5-DK3':     [{'name': 'ZFK5-DK3'}],
              'ZFK3LDK3':     [{'name': 'ZFK3LDK3'}],
              'ZFK4LDK3':     [{'name': 'ZFK4LDK3'}],
              'ZFK5LDK3':     [{'name': 'ZFK5LDK3'}],
              'SLATER-MNDO':  [{'name': 'SLATER-MNDO'}],
              'AM1':          [{'name': 'SLATER-AM1'}],
              'PM3':          [{'name': 'SLATER-PM3'}],
              'RM1':          [{'name': 'SLATER-RM1'}],
              'DFTB':         [{'name': 'SLATER-DFTB'}]
             }


       global basisset, basissetWrite, basissetreal, basissetname, nrofdfunctions, nrofffunctions, spdiffuselogical, nrofpfunctionslight, sdiffuselightlogical
       basisset = None
       basissetWrite = False
       basissetreal = None
       basissetname = None
       nrofgaussians = 0
       nrofdfunctions = 0
       nrofffunctions = 0
       spdiffuselogical = 'F'
       sdiffuselightlogical = 'F'
       nrofpfunctionslight = 0

       if(section['x_gamess_basis_set_gbasis']):

        basisset = str(section['x_gamess_basis_set_gbasis']).replace("[","").replace("]","").replace("'","").upper()
        basissetreal = basissetDict.get([basisset][-1])

        nrofgaussians = str(section['x_gamess_basis_set_igauss']).replace("[","").replace("]","").replace("'","")
        nrofdfunctions = int(str(section['x_gamess_basis_set_ndfunc']).replace("[","").replace("]","").replace("'",""))
        nrofffunctions = int(str(section['x_gamess_basis_set_nffunc']).replace("[","").replace("]","").replace("'",""))
        spdiffuselogical = str(section['x_gamess_basis_set_diffsp']).replace("[","").replace("]","").replace("'","")
        nrofpfunctionslight = int(str(section['x_gamess_basis_set_npfunc']).replace("[","").replace("]","").replace("'",""))
        sdiffuselightlogical = str(section['x_gamess_basis_set_diffs']).replace("[","").replace("]","").replace("'","")

        if(basisset == 'STO'):
            symbol = 'G'
            basissetreal = basisset + '-' + nrofgaussians + symbol
        elif(basisset == 'N21' or basisset == 'N31' or basisset == 'N311'):
            basissetname = basisset[1:]
            symbol = 'G'
        elif(basisset == 'DZV' or basisset == 'DH' or basisset == 'TZV' or basisset == 'MC'):
            symbol = ''
            nrofgaussians = ''
            basissetname = basisset[:]
        if(nrofdfunctions == 0):
            if(spdiffuselogical == 'T' or spdiffuselogical == 'TRUE'):
               basissetreal = nrofgaussians + '-' + basissetname + '+' + symbol
               if(sdiffuselightlogical == 'T' or sdiffuselightlogical == 'TRUE'):
                   basissetreal = nrofgaussians + '-' + basissetname + '++' + symbol
        if(nrofdfunctions == 1):
            basissetreal = nrofgaussians + '-' + basissetname + symbol + '(d)'
            if(nrofffunctions == 1):
                if(nrofpfunctionslight == 0):
                   basissetreal = nrofgaussians + '-' + basissetname + symbol + '(df)'
                   if(spdiffuselogical == 'T' or spdiffuselogical == 'TRUE'):
                       basissetreal = nrofgaussians + '-' + basissetname + '+' + symbol + '(df)'
                       if(sdiffuselightlogical == 'T' or sdiffuselightlogical == 'TRUE'):
                           basissetreal = nrofgaussians + '-' + basissetname + '++' + symbol + '(df)'
                elif(nrofpfunctionslight == 1):
                   basissetreal = nrofgaussians + '-' + basissetname + symbol + '(df,p)'
                   if(spdiffuselogical == 'T' or spdiffuselogical == 'TRUE'):
                       basissetreal = nrofgaussians + '-' + basissetname + '+' + symbol + '(df,p)'
                       if(sdiffuselightlogical == 'T' or sdiffuselightlogical == 'TRUE'):
                           basissetreal = nrofgaussians + '-' + basissetname + '++' + symbol + '(df,p)'
                elif(nrofpfunctionslight == 2):
                   basissetreal = nrofgaussians + '-' + basissetname + symbol + '(df,2p)'
                   if(spdiffuselogical == 'T' or spdiffuselogical == 'TRUE'):
                       basissetreal = nrofgaussians + '-' + basissetname + '+' + symbol + '(df,2p)'
                       if(sdiffuselightlogical == 'T' or sdiffuselightlogical == 'TRUE'):
                           basissetreal = nrofgaussians + '-' + basissetname + '++' + symbol + '(df,2p)'
                elif(nrofpfunctionslight == 3):
                   basissetreal = nrofgaussians + '-' + basissetname + symbol + '(df,3p)'
                   if(spdiffuselogical == 'T' or spdiffuselogical == 'TRUE'):
                       basissetreal = nrofgaussians + '-' + basissetname + '+' + symbol + '(df,3p)'
                       if(sdiffuselightlogical == 'T' or sdiffuselightlogical == 'TRUE'):
                           basissetreal = nrofgaussians + '-' + basissetname + '++' + symbol + '(df,3p)'
            elif(nrofffunctions == 0):
                if(nrofpfunctionslight == 0):
                   basissetreal = nrofgaussians + '-' + basissetname + symbol + '(d)'
                   if(spdiffuselogical == 'T' or spdiffuselogical == 'TRUE'):
                       basissetreal = nrofgaussians + '-' + basissetname + '+' + symbol + '(d)'
                       if(sdiffuselightlogical == 'T' or sdiffuselightlogical == 'TRUE'):
                           basissetreal = nrofgaussians + '-' + basissetname + '++' + symbol + '(d)'
                if(nrofpfunctionslight == 1):
                   basissetreal = nrofgaussians + '-' + basissetname + symbol + '(d,p)'
                   if(spdiffuselogical == 'T' or spdiffuselogical == 'TRUE'):
                       basissetreal = nrofgaussians + '-' + basissetname + '+' + symbol + '(d,p)'
                       if(sdiffuselightlogical == 'T' or sdiffuselightlogical == 'TRUE'):
                           basissetreal = nrofgaussians + '-' + basissetname + '++' + symbol + '(d,p)'
                if(nrofpfunctionslight == 2):
                   basissetreal = nrofgaussians + '-' + basissetname + symbol + '(d,2p)'
                   if(spdiffuselogical == 'T' or spdiffuselogical == 'TRUE'):
                       basissetreal = nrofgaussians + '-' + basissetname + '+' + symbol + '(d,2p)'
                       if(sdiffuselightlogical == 'T' or sdiffuselightlogical == 'TRUE'):
                           basissetreal = nrofgaussians + '-' + basissetname + '++' + symbol + '(d,2p)'
                if(nrofpfunctionslight == 3):
                   basissetreal = nrofgaussians + '-' + basissetname + symbol + '(d,3p)'
                   if(spdiffuselogical == 'T' or spdiffuselogical == 'TRUE'):
                       basissetreal = nrofgaussians + '-' + basissetname + '+' + symbol + '(d,3p)'
                       if(sdiffuselightlogical == 'T' or sdiffuselightlogical == 'TRUE'):
                           basissetreal = nrofgaussians + '-' + basissetname + '++' + symbol + '(d,3p)'

        if(nrofdfunctions == 2):
            basissetreal = nrofgaussians + '-' + basissetname + symbol + '(2d)'
            if(nrofffunctions == 1):
                if(nrofpfunctionslight == 0):
                   basissetreal = nrofgaussians + '-' + basissetname + symbol + '(2df)'
                   if(spdiffuselogical == 'T' or spdiffuselogical == 'TRUE'):
                       basissetreal = nrofgaussians + '-' + basissetname + '+' + symbol + '(2df)'
                       if(sdiffuselightlogical == 'T' or sdiffuselightlogical == 'TRUE'):
                           basissetreal = nrofgaussians + '-' + basissetname + '++' + symbol + '(2df)'
                elif(nrofpfunctionslight == 1):
                   basissetreal = nrofgaussians + '-' + basissetname + symbol + '(2df,p)'
                   if(spdiffuselogical == 'T' or spdiffuselogical == 'TRUE'):
                       basissetreal = nrofgaussians + '-' + basissetname + '+' + symbol + '(2df,p)'
                       if(sdiffuselightlogical == 'T' or sdiffuselightlogical == 'TRUE'):
                           basissetreal = nrofgaussians + '-' + basissetname + '++' + symbol + '(2df,p)'
                elif(nrofpfunctionslight == 2):
                   basissetreal = nrofgaussians + '-' + basissetname + symbol + '(2df,2p)'
                   if(spdiffuselogical == 'T' or spdiffuselogical == 'TRUE'):
                       basissetreal = nrofgaussians + '-' + basissetname + '+' + symbol + '(2df,2p)'
                       if(sdiffuselightlogical == 'T' or sdiffuselightlogical == 'TRUE'):
                           basissetreal = nrofgaussians + '-' + basissetname + '++' + symbol + '(2df,2p)'
                elif(nrofpfunctionslight == 3):
                   basissetreal = nrofgaussians + '-' + basissetname + symbol + '(2df,3p)'
                   if(spdiffuselogical == 'T' or spdiffuselogical == 'TRUE'):
                       basissetreal = nrofgaussians + '-' + basissetname + '+' + symbol + '(2df,3p)'
                       if(sdiffuselightlogical == 'T' or sdiffuselightlogical == 'TRUE'):
                           basissetreal = nrofgaussians + '-' + basissetname + '++' + symbol + '(2df,3p)'
            elif(nrofffunctions == 0):
                if(nrofpfunctionslight == 0):
                   basissetreal = nrofgaussians + '-' + basissetname + symbol + '(2d)'
                   if(spdiffuselogical == 'T' or spdiffuselogical == 'TRUE'):
                       basissetreal = nrofgaussians + '-' + basissetname + '+' + symbol + '(2d)'
                       if(sdiffuselightlogical == 'T' or sdiffuselightlogical == 'TRUE'):
                           basissetreal = nrofgaussians + '-' + basissetname + '++' + symbol + '(2d)'
                if(nrofpfunctionslight == 1):
                   basissetreal = nrofgaussians + '-' + basissetname + symbol + '(2d,p)'
                   if(spdiffuselogical == 'T' or spdiffuselogical == 'TRUE'):
                       basissetreal = nrofgaussians + '-' + basissetname + '+' + symbol + '(2d,p)'
                       if(sdiffuselightlogical == 'T' or sdiffuselightlogical == 'TRUE'):
                           basissetreal = nrofgaussians + '-' + basissetname + '++' + symbol + '(2d,p)'
                if(nrofpfunctionslight == 2):
                   basissetreal = nrofgaussians + '-' + basissetname + symbol + '(2d,2p)'
                   if(spdiffuselogical == 'T' or spdiffuselogical == 'TRUE'):
                       basissetreal = nrofgaussians + '-' + basissetname + '+' + symbol + '(2d,2p)'
                       if(sdiffuselightlogical == 'T' or sdiffuselightlogical == 'TRUE'):
                           basissetreal = nrofgaussians + '-' + basissetname + '++' + symbol + '(2d,2p)'
                if(nrofpfunctionslight == 3):
                   basissetreal = nrofgaussians + '-' + basissetname + symbol + '(2d,3p)'
                   if(spdiffuselogical == 'T' or spdiffuselogical == 'TRUE'):
                       basissetreal = nrofgaussians + '-' + basissetname + '+' + symbol + '(2d,3p)'
                       if(sdiffuselightlogical == 'T' or sdiffuselightlogical == 'TRUE'):
                           basissetreal = nrofgaussians + '-' + basissetname + '++' + symbol + '(2d,3p)'

        if(nrofdfunctions == 3):
            basissetreal = nrofgaussians + '-' + basissetname + symbol + '(3d)'
            if(nrofffunctions == 1):
                if(nrofpfunctionslight == 0):
                   basissetreal = nrofgaussians + '-' + basissetname + symbol + '(3df)'
                   if(spdiffuselogical == 'T' or spdiffuselogical == 'TRUE'):
                       basissetreal = nrofgaussians + '-' + basissetname + '+' + symbol + '(3df)'
                       if(sdiffuselightlogical == 'T' or sdiffuselightlogical == 'TRUE'):
                           basissetreal = nrofgaussians + '-' + basissetname + '++' + symbol + '(3df)'
                elif(nrofpfunctionslight == 1):
                   basissetreal = nrofgaussians + '-' + basissetname + symbol + '(3df,p)'
                   if(spdiffuselogical == 'T' or spdiffuselogical == 'TRUE'):
                       basissetreal = nrofgaussians + '-' + basissetname + '+' + symbol + '(3df,p)'
                       if(sdiffuselightlogical == 'T' or sdiffuselightlogical == 'TRUE'):
                           basissetreal = nrofgaussians + '-' + basissetname + '++' + symbol + '(3df,p)'
                elif(nrofpfunctionslight == 2):
                   basissetreal = nrofgaussians + '-' + basissetname + symbol + '(3df,2p)'
                   if(spdiffuselogical == 'T' or spdiffuselogical == 'TRUE'):
                       basissetreal = nrofgaussians + '-' + basissetname + '+' + symbol + '(3df,2p)'
                       if(sdiffuselightlogical == 'T' or sdiffuselightlogical == 'TRUE'):
                           basissetreal = nrofgaussians + '-' + basissetname + '++' + symbol + '(3df,2p)'
                elif(nrofpfunctionslight == 3):
                   basissetreal = nrofgaussians + '-' + basissetname + symbol + '(3df,3p)'
                   if(spdiffuselogical == 'T' or spdiffuselogical == 'TRUE'):
                       basissetreal = nrofgaussians + '-' + basissetname + '+' + symbol + '(3df,3p)'
                       if(sdiffuselightlogical == 'T' or sdiffuselightlogical == 'TRUE'):
                           basissetreal = nrofgaussians + '-' + basissetname + '++' + symbol + '(3df,3p)'
            elif(nrofffunctions == 0):
                if(nrofpfunctionslight == 0):
                   basissetreal = nrofgaussians + '-' + basissetname + symbol + '(3d)'
                   if(spdiffuselogical == 'T' or spdiffuselogical == 'TRUE'):
                       basissetreal = nrofgaussians + '-' + basissetname + '+' + symbol + '(3d)'
                       if(sdiffuselightlogical == 'T' or sdiffuselightlogical == 'TRUE'):
                           basissetreal = nrofgaussians + '-' + basissetname + '++' + symbol + '(3d)'
                if(nrofpfunctionslight == 1):
                   basissetreal = nrofgaussians + '-' + basissetname + symbol + '(3d,p)'
                   if(spdiffuselogical == 'T' or spdiffuselogical == 'TRUE'):
                       basissetreal = nrofgaussians + '-' + basissetname + '+' + symbol + '(3d,p)'
                       if(sdiffuselightlogical == 'T' or sdiffuselightlogical == 'TRUE'):
                           basissetreal = nrofgaussians + '-' + basissetname + '++' + symbol + '(3d,p)'
                if(nrofpfunctionslight == 2):
                   basissetreal = nrofgaussians + '-' + basissetname + symbol + '(3d,2p)'
                   if(spdiffuselogical == 'T' or spdiffuselogical == 'TRUE'):
                       basissetreal = nrofgaussians + '-' + basissetname + '+' + symbol + '(3d,2p)'
                       if(sdiffuselightlogical == 'T' or sdiffuselightlogical == 'TRUE'):
                           basissetreal = nrofgaussians + '-' + basissetname + '++' + symbol + '(3d,2p)'
                if(nrofpfunctionslight == 3):
                   basissetreal = nrofgaussians + '-' + basissetname + symbol + '(3d,3p)'
                   if(spdiffuselogical == 'T' or spdiffuselogical == 'TRUE'):
                       basissetreal = nrofgaussians + '-' + basissetname + '+' + symbol + '(3d,3p)'
                       if(sdiffuselightlogical == 'T' or sdiffuselightlogical == 'TRUE'):
                           basissetreal = nrofgaussians + '-' + basissetname + '++' + symbol + '(3d,3p)'

       basissetWrite = True

        #Write basis sets to metadata

       if basissetreal is not None:
          # check if only one method keyword was found in output
          if len([basisset]) > 1:
              logger.error("Found %d settings for the basis set: %s. This leads to an undefined behavior of the calculation and no metadata can be written for the basis set." % (len(basisset), basisset))
          else:
            if(gIndex == 0):
              backend.superBackend.addValue('basis_set', basissetreal)
          basissetList = basissetDict.get([basisset][-1])
          if basissetWrite:
               if basissetList is not None:
        # loop over the basis set components
                  for basissetItem in basissetList:
                        basissetName = basissetItem.get('name')
                        if basissetName is not None:
                 # write section and basis set name(s)
                           gIndexTmp = backend.openSection('section_basis_set_atom_centered')
                           backend.addValue('basis_set_atom_centered_short_name', basisset)
                           backend.closeSection('section_basis_set_atom_centered', gIndexTmp)
                        else:
                              logger.error("The dictionary for basis set '%s' does not have the key 'name'. Please correct the dictionary basissetDict in %s." % (basisset[-1], os.path.basename(__file__)))
               else:
                      logger.error("The basis set '%s' could not be converted for the metadata. Please add it to the dictionary basissetDict in %s." % (basisset[-1], os.path.basename(__file__)))


       global xc, xcWrite, methodWrite, scfmethod, mplevel, casscfkey, relatmethod, relatpseudo, methodci, methodcc, methodvb, methodmr, methodtddft

       xc = None
       xcWrite = True
       scfmethod = None
       methodWrite = True
       mplevel = 0
       casscfkey = None
       relatmethod = None
       relatpseudo = None
       methodci = None
       methodcc = None
       methodvb = None
       methodtddft = None

# functionals where hybrid_xc_coeff are written

       xc = str(section["XC_functional"]).replace("[","").replace("]","").replace("'","").upper()

       if xc is not None:
          # check if only one xc keyword was found in output
          if len([xc]) > 1:
              logger.error("Found %d settings for the xc functional: %s. This leads to an undefined behavior of the calculation and no metadata can be written for xc." % (len(xc), xc))
          else:
              backend.superBackend.addValue('x_gamess_xc', [xc][-1])
              if xcWrite:
              # get list of xc components according to parsed value
                  xcList = xcDict.get([xc][-1])
                  if xcList is not None:
                    # loop over the xc components
                      for xcItem in xcList:
                          xcName = xcItem.get('name')
                          if xcName is not None:
                          # write section and XC_functional_name
                             gIndexTmp = backend.openSection('section_XC_functionals')
                             if xcName != 'NONE':
                                backend.addValue('XC_functional_name', xcName)
                             backend.closeSection('section_XC_functionals',gIndexTmp)
                              # write hybrid_xc_coeff for PBE1PBE into XC_functional_parameters
                          else:
                              logger.error("The dictionary for xc functional '%s' does not have the key 'name'. Please correct the dictionary xcDict in %s." % (xc[-1], os.path.basename(__file__)))
                  else:
                      logger.error("The xc functional '%s' could not be converted for the metadata. Please add it to the dictionary xcDict in %s." % (xc[-1], os.path.basename(__file__)))


# Write electronic structure method to metadata

       scfmethod = str(section["x_gamess_scf_type"]).replace("[","").replace("]","").replace("'","").upper()

       methodci = str(section["x_gamess_citype"]).replace("[","").replace("]","").replace("'","").upper()
       methodcc = str(section["x_gamess_cctype"]).replace("[","").replace("]","").replace("'","").upper()
       methodvb = str(section["x_gamess_vbtype"]).replace("[","").replace("]","").replace("'","").upper()
       methodtddft = str(section["x_gamess_tddfttype"]).replace("[","").replace("]","").replace("'","").upper()
       methodcomp = str(section["x_gamess_comp_method"]).replace("[","").replace("]","").replace("'","").upper()
       relatmethod = str(section["x_gamess_relatmethod"]).replace("[","").replace("]","").replace("'","").upper()
       relatpseudo = str(section["x_gamess_pptype"]).replace("[","").replace("]","").replace("'","").upper()

       if relatmethod != 'NONE' or relatpseudo != 'NONE':
          # check if only one method keyword was found in output
          if len([relatmethod]) > 1:
              logger.error("Found %d settings for the method: %s. This leads to an undefined behavior of the calculation and no metadata can be written for the method." % (len(relatmethod), relatmethod))
          if len([relatpseudo]) > 1:
              logger.error("Found %d settings for the method: %s. This leads to an undefined behavior of the calculation and no metadata can be written for the method." % (len(relatpseudo), relatpseudo))
          methodList = methodDict.get([relatmethod][-1])
          methodList = methodDict.get([relatpseudo][-1])
          if methodWrite:
               if methodList is not None:
        # loop over the method components
                  for methodItem in methodList:
                        methodName = methodItem.get('name')
                        if methodName is not None:
                 # write section and method name
                           backend.addValue('relativity_method', methodName)
                        else:
                           logger.error("The dictionary for method '%s' does not have the key 'name'. Please correct the dictionary methodDict in %s." % (relatmethod[-1], os.path.basename(__file__)))
               else:
                    logger.error("The method '%s' could not be converted for the metadata. Please add it to the dictionary methodDict in %s." % (relatmethod[-1], os.path.basename(__file__)))

       mplevel = str(section["x_gamess_mplevel"]).replace("[","").replace("]","").replace("'","")
       casscfkey = str(section["x_gamess_mcscf_casscf"]).replace("[","").replace("]","").replace("'","")

       if scfmethod != 'NONE' and scfmethod != 'MCSCF' and methodci == 'NONE' and methodcc == 'NONE' and methodvb == 'NONE' and methodtddft == 'NONE' and mplevel != '2':
          # check if only one method keyword was found in output
          if len([scfmethod]) > 1:
              logger.error("Found %d settings for the method: %s. This leads to an undefined behavior of the calculation and no metadata can be written for the method." % (len(scfmethod), scfmethod))
          else:
              backend.superBackend.addValue('x_gamess_scf_method', [scfmethod][-1])
          methodList = methodDict.get([scfmethod][-1])
          if methodWrite:
               if methodList is not None:
        # loop over the method components
                  for methodItem in methodList:
                        methodName = methodItem.get('name')
                        if methodName is not None:
                 # write section and method name
                           gIndexTmp = backend.openSection('x_gamess_section_elstruc_method')
                           backend.addValue('x_gamess_electronic_structure_method', methodName)
                           backend.closeSection('x_gamess_section_elstruc_method', gIndexTmp)
                        else:
                           logger.error("The dictionary for method '%s' does not have the key 'name'. Please correct the dictionary methodDict in %s." % (method[-1], os.path.basename(__file__)))
               else:
                    logger.error("The method '%s' could not be converted for the metadata. Please add it to the dictionary methodDict in %s." % (method[-1], os.path.basename(__file__)))


       if scfmethod == 'MCSCF' and mplevel != '2':
          # check if only one method keyword was found in output
          if len([scfmethod]) > 1:
              logger.error("Found %d settings for the method: %s. This leads to an undefined behavior of the calculation and no metadata can be written for the method." % (len(scfmethod), scfmethod))
          else:
              backend.superBackend.addValue('x_gamess_method', [scfmethod][-1])
          methodList = methodDict.get([scfmethod][-1])
          if methodWrite:
               if methodList is not None:
        # loop over the method components
                  for methodItem in methodList:
                        methodName = methodItem.get('name')
                        if methodName is not None:
                 # write section and method name
                           if(scfmethod == 'MCSCF' and casscfkey == 'T'):
                               methodName = 'CASSCF'
                           gIndexTmp = backend.openSection('x_gamess_section_elstruc_method')
                           backend.addValue('x_gamess_electronic_structure_method', methodName)
                           backend.closeSection('x_gamess_section_elstruc_method', gIndexTmp)
                        else:
                           logger.error("The dictionary for method '%s' does not have the key 'name'. Please correct the dictionary methodDict in %s." % (method[-1], os.path.basename(__file__)))
               else:
                    logger.error("The method '%s' could not be converted for the metadata. Please add it to the dictionary methodDict in %s." % (method[-1], os.path.basename(__file__)))

       if scfmethod == 'MCSCF' and mplevel == '2':
          gIndexTmp = backend.openSection('x_gamess_section_elstruc_method')
          backend.addValue('x_gamess_electronic_structure_method', 'MRPT2')
          backend.closeSection('x_gamess_section_elstruc_method', gIndexTmp)

       if (methodci != 'NONE' or methodcc != 'NONE' or methodvb != 'NONE' or methodtddft != 'NONE' or methodcomp == 'COMP' or methodcomp == 'G3MP2') and mplevel != 2:
          # check if only one method keyword was found in output
          if len([methodci]) > 1 or len([methodcc]) > 1 or len([methodvb]) > 1 or len([methodtddft]) > 1 or len([methodcomp]) > 1:
              logger.error("There is an undefined behavior of the calculation and no metadata can be written for the method.")
          else:
              if(methodci != 'NONE'):
                 backend.superBackend.addValue('x_gamess_method', [methodci][-1])
              elif(methodcc != 'NONE'):
                 backend.superBackend.addValue('x_gamess_method', [methodcc][-1])
              elif(methodvb != 'NONE'):
                 backend.superBackend.addValue('x_gamess_method', [methodvb][-1])
              elif(methodtddft != 'NONE'):
                 backend.superBackend.addValue('x_gamess_method', [methodtddft][-1])
              elif(methodcomp == 'COMP' or methodcomp == 'G3MP2'):
                 backend.superBackend.addValue('x_gamess_method', [methodcomp][-1])
          if(methodci != 'NONE'):
              methodList = methodDict.get([methodci][-1])
          elif(methodcc != 'NONE'):
              methodList = methodDict.get([methodcc][-1])
          elif(methodvb != 'NONE'):
              methodList = methodDict.get([methodvb][-1])
          elif(methodtddft != 'NONE'):
              methodList = methodDict.get([methodtddft][-1])
          elif(methodcomp == 'COMP' or methodcomp == 'G3MP2'):
              methodList = methodDict.get([methodcomp][-1])
          if methodWrite:
               if methodList is not None:
        # loop over the method components
                  for methodItem in methodList:
                        methodName = methodItem.get('name')
                        if methodName is not None:
                 # write section and method name
                           gIndexTmp = backend.openSection('x_gamess_section_elstruc_method')
                           backend.addValue('x_gamess_electronic_structure_method', methodName)
                           backend.closeSection('x_gamess_section_elstruc_method', gIndexTmp)
                        else:
                           logger.error("The dictionary for method '%s' does not have the key 'name'. Please correct the dictionary methodDict in %s." % (methodci[-1], os.path.basename(__file__)))
               else:
                    logger.error("The method '%s' could not be converted for the metadata. Please add it to the dictionary methodDict in %s." % (methodci[-1], os.path.basename(__file__)))

       if (scfmethod == 'RHF' or scfmethod == 'UHF' or scfmethod == 'ROHF') and mplevel == '2':
          # check if only one method keyword was found in output
          backend.superBackend.addValue('x_gamess_method', 'MP2')
          gIndexTmp = backend.openSection('x_gamess_section_elstruc_method')
          backend.addValue('x_gamess_electronic_structure_method', 'MP2')
          backend.closeSection('x_gamess_section_elstruc_method', gIndexTmp)


      def onClose_section_eigenvalues(self, backend, gIndex, section):
          eigenenergies = str(section["x_gamess_alpha_eigenvalues_values"])
          eigenen1 = []
          energy = [float(f) for f in eigenenergies[:].replace("'","").replace(",","").replace("]","").replace("[","").replace("one","").replace("\\n","").replace("-"," -").split()]
          eigenen1 = np.append(eigenen1, energy)
          eigenenconalp = convert_unit(eigenen1, "hartree", "J")
          occupationsalp = np.zeros(len(eigenen1), dtype = float)
          for number in range(len(eigenen1)):
            if(section["x_gamess_beta_eigenvalues_values"]):
              if(eigenen1[number] < 0.0):
                  occupationsalp[number] = 1.0
            else:
              if(eigenen1[number] < 0.0):
                  occupationsalp[number] = 2.0

          eigenenconalpnew = np.reshape(eigenenconalp,(1, 1, len(eigenenconalp)))
          occupationsalpnew = np.reshape(occupationsalp,(1, 1, len(occupationsalp)))
          if(section["x_gamess_beta_eigenvalues_values"]):
             pass
          else:
             backend.addArrayValues("eigenvalues_values", eigenenconalpnew)
             backend.addArrayValues("eigenvalues_occupation", occupationsalpnew)

          if(section["x_gamess_beta_eigenvalues_values"]):
            eigenenergies = str(section["x_gamess_beta_eigenvalues_values"])
            eigenen1 = []
            energy = [float(f) for f in eigenenergies[:].replace("'","").replace(",","").replace("]","").replace("[","").replace("one","").replace("\\n","").replace("-"," -").split()]
            eigenen1 = np.append(eigenen1, energy)
            eigenenconbet = convert_unit(eigenen1, "hartree", "J")
            occupationsbet = np.zeros(len(eigenen1), dtype = float)
            for number in range(len(eigenen1)):
              if(eigenen1[number] < 0.0):
                occupationsbet[number] = 1.0

            eigenenall = np.concatenate((eigenenconalp,eigenenconbet), axis=0)
            occupall = np.concatenate((occupationsalp,occupationsbet), axis=0)
            eigenenall = np.reshape(eigenenall,(2, 1, len(eigenenconalp)))
            occupall = np.reshape(occupall,(2, 1, len(occupationsalp)))
            backend.addArrayValues("eigenvalues_values", eigenenall)
            backend.addArrayValues("eigenvalues_occupation", occupall)


      def onClose_x_gamess_section_geometry_optimization_info(self, backend, gIndex, section):
        # check for geometry optimization convergence
        if section['x_gamess_geometry_optimization_converged'] is not None:
           if section['x_gamess_geometry_optimization_converged'] == ['EQUILIBRIUM GEOMETRY LOCATED']:
              self.geoConvergence = True
           elif section['x_gamess_geometry_optimization_converged'] != ['FAILURE TO LOCATE STATIONARY POINT']:
              self.geoConvergence = False


      def onClose_section_scf_iteration(self, backend, gIndex, section):
        # count number of SCF iterations
        self.scfIterNr += 1
        # check for SCF convergence
        if section['single_configuration_calculation_converged'] is not None:
           self.scfConvergence = True
           if section['x_gamess_energy_scf']:
               self.scfenergyconverged = float(str(section['x_gamess_energy_scf']).replace("[","").replace("]","").replace("D","E"))
               scfmethod = str(section["x_gamess_scf_type"]).replace("[","").replace("]","").replace("'","").upper()
               if (scfmethod != ['RHF'] and scfmethod != ['UHF'] and scfmethod != ['ROHF']):
                  self.energytotal = self.scfenergyconverged
                  backend.addValue('energy_total', self.energytotal)
               else:
                  pass

      def onClose_section_single_configuration_calculation(self, backend, gIndex, section):
        """Trigger called when section_single_configuration_calculation is closed.
         Write number of SCF iterations and convergence.
         Check for convergence of geometry optimization.
        """
        # write SCF convergence and reset
        if(self.scfConvergence == 'True'):
           backend.addValue('single_configuration_calculation_converged', self.scfConvergence)
           self.scfConvergence = False
        # start with -1 since zeroth iteration is the initialization
        self.scfIterNr = -1
        # write the references to section_method and section_system
        backend.addValue('single_configuration_to_calculation_method_ref', self.secMethodIndex)
        backend.addValue('single_configuration_calculation_to_system_ref', self.secSystemDescriptionIndex)

      def onClose_x_gamess_section_mrpt2(self, backend, gIndex, section):

         mrpt2type = None
         mrpt2 = section['x_gamess_mrpt2_method_type']

         if mrpt2 == ['MRPT']:
           mrpt2type = 'MRPT2'
         elif mrpt2 == ['MRMP2']:
           mrpt2type = 'MC-QDPT'
         gIndexTmp = backend.openSection('section_method')
         backend.addValue("x_gamess_method", mrpt2type)
         backend.closeSection('section_method', gIndexTmp)

      def onClose_x_gamess_section_frequencies(self, backend, gIndex, section):
          frequencies = str(section["x_gamess_frequency_values"])
          vibfreqs = []
          frequencies = frequencies.replace("'","").replace(",","").replace("[","").replace("]","").replace("\\n","").split()
          if(frequencies[1] == 'I'):
             frequencies[0] = -float(frequencies[0])

          if(frequencies[1] == 'I'):
             for i in range(2,len(frequencies)):
               frequencies[i-1] = frequencies[i]

          n = len(frequencies)

          for freqs in frequencies[:n-1]:
              vibfreqs = np.append(vibfreqs,float(freqs))

          vibfreqs = convert_unit(vibfreqs, "inversecm", "J")
          backend.addArrayValues("x_gamess_frequencies", vibfreqs)

          masses = str(section["x_gamess_reduced_masses"])
          vibreducedmasses = []
          reduced = [float(f) for f in masses[:].replace("'","").replace(",","").replace("]","").replace("[","").replace("\\n","").split()]
          vibreducedmasses = np.append(vibreducedmasses, reduced)
          vibreducedmasses = convert_unit(vibreducedmasses, "amu", "kilogram")
          backend.addArrayValues("x_gamess_red_masses", vibreducedmasses)


# which values to cache or forward (mapping meta name -> CachingLevel)

cachingLevelForMetaName = {
        "basis_set_atom_centered_short_name": CachingLevel.ForwardAndCache,
        "section_basis_set_atom_centered": CachingLevel.Forward,
        "x_gamess_atom_x_coord": CachingLevel.Cache,
        "x_gamess_atom_y_coord": CachingLevel.Cache,
        "x_gamess_atom_z_coord": CachingLevel.Cache,
        "x_gamess_atomic_number": CachingLevel.Cache,
        "x_gamess_section_geometry": CachingLevel.Forward,
        "XC_functional_name": CachingLevel.ForwardAndCache,
        "x_gamess_scf_type": CachingLevel.ForwardAndCache,
        "x_gamess_method": CachingLevel.ForwardAndCache,
        "relativistic_method": CachingLevel.ForwardAndCache,
        "x_gamess_section_geometry_optimization_info": CachingLevel.Forward,
        "x_gamess_geometry_optimization_converged": CachingLevel.ForwardAndCache,
}


class GamessParser():
   """ A proper class envolop for running this parser from within python. """
   def __init__(self, backend, **kwargs):
       self.backend_factory = backend

   def parse(self, mainfile):
       from unittest.mock import patch
       logging.info('gamess parser started')
       logging.getLogger('nomadcore').setLevel(logging.WARNING)
       backend = self.backend_factory("gamess.nomadmetainfo.json")
       with patch.object(sys, 'argv', ['<exe>', '--uri', 'nmd://uri', mainfile]):
           mainFunction(
               mainFileDescription,
               None,
               parserInfo,
               cachingLevelForMetaName = cachingLevelForMetaName,
               superContext=GAMESSParserContext(),
               superBackend=backend)

       return backend
