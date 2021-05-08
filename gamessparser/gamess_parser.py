#
# Copyright The NOMAD Authors.
#
# This file is part of NOMAD.
# See https://nomad-lab.eu for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import numpy as np
import re
from ase.data import atomic_names, chemical_symbols

from nomad.units import ureg
from nomad.parsing.file_parser import BasicParser


class GamessParser(BasicParser):
    def __init__(self):
        re_f = r'\-*\d+\.\d+E*\-*\+*\d*'

        def get_positions(val):
            # gamess and firefly use diff units
            unit = ureg.bohr if 'BOHR' in val else ureg.angstrom
            res = re.findall(rf'(\w+)\s*{re_f}\s*({re_f} +{re_f} +{re_f})', val)
            labels, positions = [], []
            if res:
                try:
                    labels, positions = list(zip(*res))
                    # firefly prints full name
                    labels = [chemical_symbols[atomic_names.index(s.title())] if len(s) > 2 else s for s in labels]
                    positions = np.array([p.split() for p in positions], dtype=np.dtype(np.float64)) * unit
                except Exception:
                    pass
            return dict(atom_positions=positions, atom_labels=labels)

        super().__init__(
            specifications=dict(
                name='parsers/gamess', code_name='GAMESS',
                code_homepage='https://www.msg.chem.iastate.edu/gamess/versions.html',
                mainfile_contents_re=(
                    r'\s*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\**\s*'
                    r'\s*\*\s*GAMESS VERSION =\s*(.*)\*\s*'
                    r'\s*\*\s*FROM IOWA STATE UNIVERSITY\s*\*\s*')),
            units_mapping=dict(length=ureg.bohr, energy=ureg.hartree),
            # include code name to distinguish gamess and firefly
            program_version=r'GAMESS VERSION *\= *(.+?) *\*|(Firefly version [\d\.]+)',
            atom_labels_atom_positions=(
                r'(COORDINATES OF ALL ATOMS ARE \(ANGS\)\s*ATOM.+\s*[\s\S]+?)\n *\n|'
                # not sure why \n *\n does not work
                r'(ATOM *ATOMIC *COORDINATES \(BOHR\)\s*CHARGE *X *Y *Z[\s\S]+?)INTER', get_positions),
            energy_total=rf'TOTAL ENERGY \=\s*({re_f})',
            atom_forces=r'GRADIENT \(HARTREE\/BOHR\)([\s\S]+?)\n *\n')
