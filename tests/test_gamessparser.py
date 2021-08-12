#
# Copyright The NOMAD Authors.
#
# This file is part of NOMAD. See https://nomad-lab.eu for further info.
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

import pytest

from nomad.datamodel import EntryArchive
from gamessparser import GamessParser


def approx(value, abs=0, rel=1e-6):
    return pytest.approx(value, abs=abs, rel=rel)


@pytest.fixture(scope='module')
def parser():
    return GamessParser()


def test_gamess(parser):
    archive = EntryArchive()

    parser.parse('tests/data/gamessus/exam01.out', archive, None)

    sec_run = archive.run[0]
    assert sec_run.program.version == '1 MAY 2013 (R1)'

    sec_systems = archive.run[0].system
    assert sec_systems[0].atoms.labels == ['C', 'H', 'H']
    assert sec_systems[0].atoms.positions[1][0].magnitude == approx(-8.92875664e-11)
    assert sec_systems[4].atoms.positions[2][2].magnitude == approx(5.69562232e-11)

    sec_sccs = sec_run.calculation
    assert len(sec_sccs) == 7
    assert sec_sccs[0].energy.total.value.magnitude == approx(-1.62323183e-16)
    assert sec_sccs[2].forces.total.value[1][2].magnitude == approx(-1.04055078e-10)


def test_firefly(parser):
    archive = EntryArchive()

    parser.parse('tests/data/firefly/bench01.out', archive, None)

    sec_run = archive.run[0]
    assert sec_run.program.version == 'Firefly version 8.2.0'

    sec_systems = archive.run[0].system
    assert sec_systems[0].atoms.labels[5] == 'H'
    assert sec_systems[0].atoms.positions[7][2].magnitude == approx(-1.219788e-10)

    sec_sccs = sec_run.calculation
    assert len(sec_sccs) == 1
    assert sec_sccs[0].energy.total.value.magnitude == approx(-1.60476447e-15)
