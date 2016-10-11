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
           startReStr = r"\s*GAMESS temporary binary files|\s*\*\s*Firefly version",
           repeats = True,
           required = True,
           forwardMatch = True,
           fixedStartValues={ 'program_name': 'GAMESS', 'program_basis_set_type': 'gaussians' },
           sections   = ['section_run'],
           subMatchers = [
               SM(name = 'header',
                  startReStr = r"\s*GAMESS temporary binary files|\s*\*\s*Firefly version",
                  forwardMatch = True,
                  subMatchers = [
                      SM(r"\s*GAMESS temporary binary files"),
                      SM(r"\s*\*\s*GAMESS VERSION \=\s*(?P<program_version>[0-9]+\s*[A-Z]+\s*[0-9]+)"),
                      SM(r"\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\s*(?P<x_gamess_program_implementation>[0-9]+\s*[A-Z]+\s*[A-Z]+\s*[A-Z]+)"),
                      SM(r"\s*EXECUTION OF GAMESS BEGUN\s*(?P<x_gamess_program_execution_date>[a-zA-Z]+\s*[a-zA-Z]+\s*[0-9]+\s*[0-9][0-9][:][[0-9][0-9][:][0-9][0-9]\s*[0-9]+)"),
                      SM(r"\s*\*\s*Firefly version\s*(?P<program_version>[0-9.]+)"),
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
                      SM(r"\s*ATOM      ATOMIC"),
                      SM(r"\s*[A-Z0-9?]+\s+(?P<x_gamess_atomic_number>\d+\.\d)\s+(?P<x_gamess_atom_x_coord_initial__bohr>[-+0-9.]+)\s+(?P<x_gamess_atom_y_coord_initial__bohr>[-+0-9.]+)\s+(?P<x_gamess_atom_z_coord_initial__bohr>[-+0-9.]+)",repeats = True),
                      SM(r"\s*INTERNUCLEAR DISTANCES"),
                      SM(r"\s*NUMBER OF ELECTRONS\s*=\s*(?P<number_of_electrons>[0-9]+)"),
                      SM(r"\s*CHARGE OF MOLECULE\s*=\s*(?P<x_gamess_total_charge>[0-9-]+)"),
                      SM(r"\s*SPIN MULTIPLICITY\s*=\s*(?P<x_gamess_spin_target_multiplicity>[0-9]+)"),
                      SM(r"\s*TOTAL NUMBER OF ATOMS\s*=\s*(?P<number_of_atoms>[0-9]+)"),
                      ]
             ),
               SM (name = 'SectionMethod',
               sections = ['section_method'],
                   startReStr = r"\s*\$CONTRL OPTIONS",
                   forwardMatch = False,
                   subMatchers = [
                       SM(r"\s*SCFTYP=[A-Z]+\s*RUNTYP=[A-Z]+\s*EXETYP=[A-Z]+"),
                       SM(r"\s*MPLEVL=\s*[0-9]\s*CITYP =[A-Z]+\s*CCTYP =[A-Z]+\s*VBTYP =[A-Z]+"),
                       SM(r"\s*DFTTYP=(?P<XC_functional>[-A-Z0-9]+)\s*TDDFT =[A-Z]+"),
                       SM(r"\s*PP    =[A-Z]+\s*RELWFN=[A-Z]+"),
                       ]
             ),
            SM (name = 'SingleConfigurationCalculationWithSystemDescription',
                startReStr = r"\s*COORDINATES OF ALL ATOMS|\s*[-A-Z0-9]+\s*SCF CALCULATION",
                repeats = False,
                forwardMatch = True,
                subMatchers = [
                SM (name = 'SingleConfigurationCalculation',
                  startReStr = r"\s*COORDINATES OF ALL ATOMS|\s*[-A-Z0-9]+\s*SCF CALCULATION",
                  repeats = True,
                  forwardMatch = True,
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
                    repeats = True,
                    subMatchers = [
                     SM(r"(\s+[0-9]+)(\s+[0-9]+)(\s+[0-9]+)?\s*(?P<energy_total_scf_iteration>[-+0-9.]+)",repeats = True),
                     SM(r"\s*(?P<single_configuration_calculation_converged>DENSITY CONVERGED)"),
                     SM(r"\s*FINAL\s*[-A-Z0-9]+\s*ENERGY IS\s*(?P<x_gamess_energy_scf>[-+0-9.]+)"),
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
        self.scfkineticenergyconverged = 0.0
        self.scfelectrostaticenergy = 0.0
        self.periodicCalc = False

      def startedParsing(self, path, parser):
        self.parser = parser
        # save metadata
        self.metaInfoEnv = self.parser.parserBuilder.metaInfoEnv
        # allows to reset values if the same superContext is used to parse different files
        self.initialize_values()

      def onClose_x_gamess_section_geometry(self, backend, gIndex, section):

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

      def onClose_section_system(self, backend, gIndex, section):
       if self.skip_system_onclose:
         return
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

        #basis sets

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
              'SPKAQZP':      [{'name': 'AUG-SPK-QZP'}],
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
              'SBKJC':        [{'name': 'SBKJC'}],
              'HW':           [{'name': 'HAYWADT'}],
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

       basissetreal = basissetDict.get([basisset][-1])

       if(section['x_gamess_basis_set_gbasis']):

        basisset = str(section['x_gamess_basis_set_gbasis']).replace("[","").replace("]","").replace("'","").upper()
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

       if basisset is not None:
          # check if only one method keyword was found in output
          if len([basisset]) > 1:
              logger.error("Found %d settings for the basis set: %s. This leads to an undefined behavior of the calculation and no metadata can be written for the basis set." % (len(method), method))
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
              'IOTC':      [{'name': 'INFINITEORDERTWOCOMPONENT'}],
              'DK':        [{'name': 'DOUGLASKROLL'}],
              'RESC':      [{'name': 'RELATIVISTICELIMINATIONOFSMALLCOMPONENT'}],
              'NESC':      [{'name': 'NORMALISEDELIMINTIONOFSMALLCOMPONENT'}],
              'TAMMD':     [{'name': 'TAMM-DANCOFF'}],
              'DFTB':      [{'name': 'DFTB'}],
              'MP2':       [{'name': 'MP2'}],
              'RIMP2':     [{'name': 'RESOLUTIONOFIDENTITY-MP2'}],
              'CPHF':      [{'name': 'COUPLEDPERTURBED-HF'}],
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
             }

       global xc,xcWrite

       xc = None
       xcWrite = True

# functionals where hybrid_xc_coeff are written

       xc = str(section["XC_functional"]).replace("[","").replace("]","").replace("'","")

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
                              backend.addValue('XC_functional_name', xcName)
                              # write hybrid_xc_coeff for PBE1PBE into XC_functional_parameters
                          else:
                              backend.closeSection('section_XC_functionals', gIndexTmp)
                              logger.error("The dictionary for xc functional '%s' does not have the key 'name'. Please correct the dictionary xcDict in %s." % (xc[-1], os.path.basename(__file__)))
                  else:
                      logger.error("The xc functional '%s' could not be converted for the metadata. Please add it to the dictionary xcDict in %s." % (xc[-1], os.path.basename(__file__)))



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
}

if __name__ == "__main__":
    mainFunction(mainFileDescription, metaInfoEnv, parserInfo,
                 cachingLevelForMetaName = cachingLevelForMetaName,
                 superContext = GAMESSParserContext())
