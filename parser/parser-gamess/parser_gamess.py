from __future__ import division
from builtins import str
from builtins import range
from builtins import object
from functools import reduce
import setup_paths
from nomadcore.simple_parser import mainFunction, SimpleMatcher as SM
from nomadcore.local_meta_info import loadJsonFile, InfoKindEl
from nomadcore.caching_backend import CachingLevel
from nomadcore.unit_conversion.unit_conversion import convert_unit
import os, sys, json, logging
import numpy as np
import ase
import re

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
           startReStr = r"\s*GAMESS temporary binary files",
           repeats = True,
           required = True,
           forwardMatch = True,
           fixedStartValues={ 'program_name': 'GAMESS', 'program_basis_set_type': 'gaussians' },
           sections   = ['section_run'],
           subMatchers = [
               SM(name = 'header',
                  startReStr = r"\s*GAMESS temporary binary files",
                  forwardMatch = False,
                  subMatchers = [
                      SM(r"\s*\*\s*GAMESS VERSION \=\s*(?P<program_version>[0-9]+\s*[A-Z]+\s*[0-9]+)"),
                      SM(r"\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\s*(?P<x_gamess_program_implementation>[0-9]+\s*[A-Z]+\s*[A-Z]+\s*[A-Z]+)"),
                      SM(r"\s*EXECUTION OF GAMESS BEGUN\s*(?P<x_gamess_program_execution_date>[a-zA-Z]+\s*[a-zA-Z]+\s*[0-9]+\s*[0-9][0-9][:][[0-9][0-9][:][0-9][0-9]\s*[0-9]+)"),
                      ]),
               SM(name = 'charge_multiplicity_atoms',
               sections = ['section_system'],
                  startReStr = r"\s*ECHO OF THE FIRST FEW INPUT CARDS",
                  forwardMatch = True,
                  subMatchers = [
                      SM(r"\s*(?P<x_gamess_memory>[0-9]+)\s*WORDS OF MEMORY AVAILABLE"),
                      SM(r"\s*BASIS OPTIONS"),
                      SM(r"\s*GBASIS=(?P<x_gamess_basis_set_gbasis>[A-Z0-9-]+)\s*IGAUSS=\s*(?P<x_gamess_basis_set_igauss>[0-9]+)\s*POLAR=(?P<x_gamess_basis_set_polar>[A-Z]+)"),
                      SM(r"\s*NDFUNC=\s*(?P<x_gamess_basis_set_ndfunc>[0-9]+)\s*NFFUNC=\s*(?P<x_gamess_basis_set_nffunc>[0-9]+)\s*DIFFSP=\s*(?P<x_gamess_basis_set_diffsp>[TF])"),
                      SM(r"\s*NPFUNC=\s*(?P<x_gamess_basis_set_npfunc>[0-9]+)\s*DIFFS=\s*(?P<x_gamess_basis_set_diffs>[TF])"),
                      SM(r"\s*THE MOMENTS OF INERTIA ARE"),
                      SM(r"\s*IXX=\s*(?P<x_gamess_moment_of_inertia_X>[0-9.]+)\s*IYY=\s*(?P<x_gamess_moment_of_inertia_Y>[0-9.]+)\s*IZZ=\s*(?P<x_gamess_moment_of_inertia_Z>[0-9.]+)"),
                      SM(r"\s*ATOM      ATOMIC"),
                      SM(r"\s*CHARGE         X"),
                      SM(r"\s*[A-Z]+\s+(?P<x_gamess_atomic_number>[0-9.]+)\s+(?P<x_gamess_atom_x_coord__bohr>(\-?\d+\.\d{10}))\s+(?P<x_gamess_atom_y_coord__bohr>(\-?\d+\.\d{10}))\s+(?P<x_gamess_atom_z_coord__bohr>(\-?\d+\.\d{10}))",repeats = True),
                      SM(r"\s*NUMBER OF ELECTRONS\s*=\s*(?P<number_of_electrons>[0-9]+)"),
                      SM(r"\s*CHARGE OF MOLECULE\s*=\s*(?P<x_gaussian_total_charge>[0-9-]+)"),
                      SM(r"\s*SPIN MULTIPLICITY\s*=\s*(?P<x_gamess_spin_target_multiplicity>[0-9]+)"),
                      SM(r"\s*TOTAL NUMBER OF ATOMS\s*=\s*(?P<number_of_atoms>[0-9]+)")
                      ]
             ),
            SM (name = 'SingleConfigurationCalculationWithSystemDescription',
                startReStr = "\s*BEGINNING GEOMETRY SEARCH POINT NSERCH=",
                repeats = False,
                forwardMatch = True,
                subMatchers = [
                SM (name = 'SingleConfigurationCalculation',
                  startReStr = "\s*COORDINATES OF SYMMETRY UNIQUE ATOMS",
                  repeats = True,
                  forwardMatch = False,
                  sections = ['section_single_configuration_calculation'],
                  subMatchers = [
                  SM(name = 'geometry',
                   sections  = ['section_system'],
                   startReStr = r"\s*COORDINATES OF ALL ATOMS",
                      subMatchers = [
                      SM(r"\s*ATOM   CHARGE       X"),
                      SM(r"\s*[A-Z]+\s+(?P<x_gamess_atomic_number>[0-9.]+)\s+(?P<x_gamess_atom_x_coord__bohr>[-+0-9.]+)\s+(?P<x_gamess_atom_y_coord__bohr>[-+0-9.]+)\s+(?P<x_gamess_atom_z_coord__bohr>[-+0-9.]+)",repeats = True),
                      SM(r"\s*THE CURRENT FULLY SUBSTITUTED Z-MATRIX IS")
                    ]
                ),
          ])
        ])
      ])
    ])

# loading metadata from nomad-meta-info/meta_info/nomad_meta_info/gamess.nomadmetainfo.json
metaInfoPath = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"../../../../nomad-meta-info/meta_info/nomad_meta_info/gamess.nomadmetainfo.json"))
metaInfoEnv, warnings = loadJsonFile(filePath = metaInfoPath, dependencyLoader = None, extraArgsHandling = InfoKindEl.ADD_EXTRA_ARGS, uri = None)

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
        self.scfkineticenergyconverged = 0.0
        self.scfelectrostaticenergy = 0.0
        self.periodicCalc = False

      def startedParsing(self, path, parser):
        self.parser = parser
        # save metadata
        self.metaInfoEnv = self.parser.parserBuilder.metaInfoEnv
        # allows to reset values if the same superContext is used to parse different files
        self.initialize_values()

      def onClose_section_system(self, backend, gIndex, section):
        xCoord = section["x_gamess_atom_x_coord"]
        yCoord = section["x_gamess_atom_y_coord"]
        zCoord = section["x_gamess_atom_z_coord"]
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

        #basis sets

        basissetDict = {
              'STO-2G':      [{'name': 'STO'}],
              'STO-3G':      [{'name': 'STO'}],
              '3-21G':       [{'name': 'N21'}],
              '4-31G':       [{'name': 'N31'}],
              '5-31G':       [{'name': 'N31'}],
              '6-311G':       [{'name': 'N311'}],
              '6-21G':       [{'name': 'N21'}],
              '6-31G':       [{'name': 'N31'}],
              '6-311G':       [{'name': 'N311'}],
             }

        global basisset, basissetWrite, basissetreal
        basisset = None
        basissetWrite = False
        basissetreal = None

        basisset = str(section['x_gamess_basis_set_gbasis']).replace("[","").replace("]","").replace("'","")
        nrofgaussians = str(section['x_gamess_basis_set_igauss']).replace("[","").replace("]","").replace("'","") 
        if(basisset == 'STO'):
            basissetreal = basisset + '-' + nrofgaussians + 'G'
        elif(basisset == 'N21' or basisset == 'N31' or basisset == 'N311'):
            basissetreal = nrofgaussians + '-' + basisset[1:] + 'G'

        if basissetreal in basissetDict.keys():
            basissetWrite = True

        #Write basis sets to metadata

        if basisset is not None:
          # check if only one method keyword was found in output
          if len([basissetreal]) > 1:
              logger.error("Found %d settings for the basis set: %s. This leads to an undefined behavior of the calculation and no metadata can be written for the basis set." % (len(method), method))
          else:
              backend.superBackend.addValue('basis_set', basissetreal)
          basissetList = basissetDict.get([basissetreal][-1])
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
 

# which values to cache or forward (mapping meta name -> CachingLevel)

cachingLevelForMetaName = {
        "basis_set_atom_centered_short_name": CachingLevel.ForwardAndCache,
        "section_basis_set_atom_centered": CachingLevel.Forward
}

if __name__ == "__main__":
    mainFunction(mainFileDescription, metaInfoEnv, parserInfo,
                 cachingLevelForMetaName = cachingLevelForMetaName,
                 superContext = GAMESSParserContext())
