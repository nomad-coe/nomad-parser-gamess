#!/usr/bin/env python

# xXXXXxx This file copied from GULP  XXXxxXxxXXXX

from __future__ import print_function
from glob import glob
import re
from optparse import OptionParser
from subprocess import Popen, PIPE
import os
from ase.parallel import world
import platform

p = OptionParser()
opts, args = p.parse_args()

#if len(args) == 0:
testfiles = glob('../../test/examples/gamessus/*.out')
testfiles.sort()

    #tests = {}

    #testfiles.sort()
    #print(testfiles)

    #pat = re.compile('test/outputs/example(\d+)[ab]?.got')

    #for fname in testfiles:
    #    print(fname, pat.match(fname))
    #testfiles.sort(key=lambda fname:
    #               int(pat.match(fname).group(1)))
    #for f in testfiles:
    #    print(f)
#else:
#testfiles = argv

for i, testfile in enumerate(testfiles):
    if i % world.size != world.rank:
        continue
    dirname, basename = os.path.split(testfile)
    #print(testfile)
    #print(basename)
    py = 'python'
    if platform.node() == 'labdev-nomad':
        py = '/labEnv3/bin/python'
    args = [py, 'parser_gamess.py', '--annotate']
    args.append(testfile)
    print(' '.join(args))
    proc = Popen(args, stdout=PIPE)
    txt = proc.stdout.read()
    with open('%s.json' % testfile, 'w') as fd:
        fd.write(txt.decode('ascii'))
