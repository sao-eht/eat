# hopstools
tools to convert data from the HOPS format

=============================

To run: 

python hops2uvfits.py 

Flags: 

--skip_bl: skip regenerating each baseline uvfits files

--clean: remove each old baseline uvfits files in case the data/baselines have changed

--rot_rate: de-rotate the fringe rate to the original values in order to form reliable closure phase. Do not do this if you want to average in time after loading the uvfits files. 

--rot_delay: de-rotate the fringe delay-rate to the original values in order to form reliable closure phase. Do not do this if you want to average down in frequency after loading the uvfits files. 

=============================

For Katie: 

cd /Users/klbouman/Research/vlbi_imaging/software/hops/build
source hops.bash

run this file from: /Users/klbouman/Research/vlbi_imaging/software/hops/eat