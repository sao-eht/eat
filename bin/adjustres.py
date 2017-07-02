#!/usr/bin/env python

# adjust output resolution for DiFX bands based on bandwidth
# 2017-05-22 Lindy Blackburn

from __future__ import division
from builtins import str
import sys
import optparse # dewi has optparse, no argparse

parser = optparse.OptionParser(description='Filter DiFX input file and change resolution for certain channels based on bandwidth. Can also update an original channel bandwidth to the desired bandwidth and resolution. By default read and write to STDIN/STDOUT')
parser.add_option('-b', '--bandwidth', type=float, default=58.59375, help='channel BW [MHz]')
parser.add_option('-r', '--resolution', type=float, default=0.46875, help='output spectral resolution [MHz]')
parser.add_option('-u', '--update', type=float, default=58.0, help='also update channels with original BW [MHz]')
parser.add_option('-i', '--input-file', dest='input_file', default=None, help='input file (default STDIN)')
parser.add_option('-o', '--output-file', dest='output_file', default=None, help='output file (default STDOUT)')
(args, ignore) = parser.parse_args()

if args.input_file is None:
    IN = sys.stdin
else:
    IN = open(args.input_file)
if args.output_file is None:
    OUT = sys.stdout
else:
    OUT = open(args.output_file, 'w')

obw = dict() # original BW
nbw = dict() # new BW
fftres = dict() # resolution of channel (fixed)

echo = False
for line in IN:
    if echo:
        OUT.write(line)
        continue
    # BW (MHZ) 0:         62.50000000000
    if line[:len("BW (MHZ)")] == "BW (MHZ)":
        tok = line.split(':')
        idx = tok[0].split()[-1]
        obw[idx] = float(tok[1]) # original bandwidth
        if obw[idx] == args.update:
            nbw[idx] = args.bandwidth # set to new bandwidth
            line2 = line[:len("BW (MHZ) 0:         ")] + str(nbw[idx]) + '\n'
            OUT.write(line2)
            continue
        else:
            nbw[idx] = obw[idx]
        # print "**** MSG1 %s %f" % (tok[0].split()[-1], float(tok[1]))
    # NUM CHANNELS 0:     4000
    if line[:len("NUM CHANNELS")] == "NUM CHANNELS":
        tok = line.split(':')
        idx = tok[0].split()[-1]
        if nbw[idx] == args.bandwidth: # we will want to change resolution for this channel later
            fftres[idx] = obw[idx] / float(tok[1]) # fftres based on original channel bw
            if obw[idx] != nbw[idx]: # we need to update nchan
                nchan = int(0.5 + nbw[idx] / fftres[idx])
                line2 = line[:len("NUM CHANNELS 0:     ")] + str(nchan) + '\n'
                OUT.write(line2)
                continue
        # print "**** MSG2 %s %d" % (tok[0].split()[-1], int(tok[1]))
    # CHANS TO AVG 0:     32
    if line[:len("CHANS TO AVG")] == "CHANS TO AVG":
        tok = line.split(':')
        idx = tok[0].split()[-1]
        if nbw[idx] == args.bandwidth: # we want to update this resolution
            navg = int(0.5 + (args.resolution / fftres[idx]))
            line2 = line[:len("CHANS TO AVG 0:     ")] + str(navg) + '\n'
            # print "**** MSG3 %s %d %f %f" % (idx, navg, bw[idx], args.bandwidth)
            OUT.write(line2)
            continue
    # skip rest
    if line[:len("# TELESCOPE TABLE ##!")] == "# TELESCOPE TABLE ##!":
        echo = True
    OUT.write(line)
