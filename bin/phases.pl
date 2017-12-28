#!/usr/bin/perl

use warnings;
use diagnostics;
use strict;
use Getopt::Std;
use POSIX;
use List::MoreUtils qw(uniq);
our $VERSION = 0.02;

my %opts;
$opts{'f'} = 'NONE';
$opts{'s'} = 0;
$opts{'q'} = 0;
$opts{'n'} = 0;
$opts{'v'} = 0;
$opts{'l'} = 0;
$opts{'w'} = 1;
my $USAGE="Usage: $0 [options]

where the required options are

  -f  input param file
      The parameter file should have, in this order,
        qcode, snr, phase, corel_phase(1-30)
      The file can be produced by aedit using
        param 72 54 55 22
        pwrite filename

  -s  minimum snr          [$opts{'s'}]
  -q  minimum qcode        [$opts{'q'}]
  -n  # of channels        [from file]
  -w  exponent on weight   [$opts{'w'}]

  -l  use linear averaging [defaults to complex averaging]
  -v  verbose

  Complex averaging will handle phases near +/-180 deg better than linear.

  If referring to one telescope, use the line where the other telescope comes first.

  Version $VERSION
  v0.02: Added complex averaging as default
";


if ( $#ARGV < 0 || $ARGV[0] eq "--help" ) { print "$USAGE"; exit(0); }

getopts('f:s:q:n:vlw:b',\%opts);
my($infile,$minsnr,$minqcode,$nchan,$verbose,$useweight,$opposite,$dolinear);
$infile    = $opts{'f'};
$minsnr    = $opts{'s'};
$minqcode  = $opts{'q'};
$nchan     = $opts{'n'};
$verbose   = $opts{'v'};
$useweight = $opts{'w'};
$dolinear  = $opts{'l'};
$opposite  = 1;  # set this true by default

die "Must specify input file with -f option\n" if ($infile eq 'NONE');
die "Required input file ($infile) is missing\n" if ( ! -f $infile);

open(FILIN,$infile);
my(@entry,$i,@phases,@fringe,@source,@nfreqs,@qcode,@snr,@residphase,$n);
my(@baseline,$temp,$tempsnr,$tempqcode,$lastchannel);

if ($nchan==0){
    while (<FILIN>){
	chomp;
	if (/^\*/){
	    @entry = split();
	    if ($entry[1] eq "Parameters"){
		$nchan = $entry[7];
		$nchan =~ s/corel_phase\(1-//;
		$nchan =~ s/\)//;
	    }
	}
    }
}
close(FILIN);


open(FILIN,$infile);
$i = 0;
while (<FILIN>) {
    chomp;
    next if (/^\*/);       # skip comments
    @entry          =  split();
    $tempsnr        =  $entry[4];
    $tempqcode      =  $entry[3];
    $lastchannel    =  5+$nchan;
    if (($tempsnr >= $minsnr) && ($tempqcode >= $minqcode)){
	$phases[$i]     =  [ @entry[6..$lastchannel] ];
	$fringe[$i]     =  $entry[0];
	$source[$i]     =  $entry[1];
	$nfreqs[$i]     =  $entry[2];
	$nfreqs[$i]     =~ s/://;
	$qcode[$i]      =  $entry[3];
	$snr[$i]        =  $entry[4];
	$residphase[$i] =  $entry[5];
	$temp           =  $fringe[$i];
	$temp           =~ s/.*\///;
	$temp           =~ s/\..*//;
	$baseline[$i]   =  $temp;
	$i++;
    }
}
$n = $i;
close (FILIN);

my ($j,$k);

for ($i=0;$i<$n;$i++){
    for ($j=0;$j<$nchan;$j++){
	$phases[$i][$j] -= $residphase[$i];
        if ($phases[$i][$j]>180){$phases[$i][$j]-=360;}
        if ($phases[$i][$j]<=-180){$phases[$i][$j]+=360;}
    }
}

my @baselineset = sort (uniq(@baseline));


if ($verbose){
    #for ($i=0;$i<=$#baselineset;$i++){
    #	print "*$baselineset[$i] ";
    #}
    #print "\n\n\n";
    for ($i=0;$i<$n;$i++){
	print "*$baseline[$i] ";
	for ($j=0;$j<$nchan;$j++){
	    printf ("%9.3f",$phases[$i][$j]);
	}
	print "\n";
    }
    print "\n\n\n";
}


my ($weight,@averagephase,$thisweight,$deg2rad);
my (@averagereal,@averageimag,@averagecomplexphase);
$deg2rad = 3.14159265/180;
for ($k=0;$k<=$#baselineset;$k++){
    $weight = 0;
    for ($j=0;$j<$nchan;$j++){
        #averagephase computes naive averages with wrap errors at +/-180
	$averagephase[$j]        = 0;
        #averagecomplexphase should deal more gracefully with wraps
	$averagecomplexphase[$j] = 0;
	$averagereal[$j]         = 0;
	$averageimag[$j]         = 0;
    }

    for ($i=0;$i<$n;$i++){
	if ($baseline[$i] eq $baselineset[$k]){
	    $thisweight = $snr[$i]**$useweight;
	    $weight += $thisweight;
	    for ($j=0;$j<$nchan;$j++){
		$averagephase[$j] += ($phases[$i][$j]*$thisweight);
		$averagereal[$j]  += (cos($phases[$i][$j]*$deg2rad)*$thisweight);
		$averageimag[$j]  += (sin($phases[$i][$j]*$deg2rad)*$thisweight);
	    }
	}
    }

    if ($weight > 0){
	print "$baselineset[$k] ";
	for ($j=0;$j<$nchan;$j++){
	    $averagephase[$j] /= $weight;
	    $averagecomplexphase[$j] = atan2($averageimag[$j],$averagereal[$j])/$deg2rad;
	    if ($dolinear) {printf ("%9.3f",$averagephase[$j]);}
	    else           {printf ("%9.3f",$averagecomplexphase[$j]);}
	}
	print "\n";
	if ($opposite){
	    print substr($baselineset[$k],1,1);
	    print substr($baselineset[$k],0,1);
	    print " ";
	    for ($j=0;$j<$nchan;$j++){
		if ($dolinear) {printf ("%9.3f",-$averagephase[$j]);}
		else           {printf ("%9.3f",-$averagecomplexphase[$j]);}
	    }
	    print "\n";
	}
    }

}

 
