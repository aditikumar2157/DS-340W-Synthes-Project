#!/bin/bash

########################################################################
##                                                                    ##
## This script full automates RMSD calculation for pose               ##
## reproducibility accessment in V-SYNTHES                            ##
## Version 02082022  -  Antonina L. Nazarova   nazarova@usc.edu       ## 
## If using this script, please cite Antonina L. Nazarova             ##
## nazarova@usc.edu, Depqrtment of Quantitative and Computational     ##
## Biology, University of Southern California, LA, USA, 90089         ##
########################################################################



## Make CapSelect files directory - All input and output files of CapSelect are stored here
mkdir -p RMSD

## Execute icm_generate_CapSelect_files.icm using ICM (icm-3.9-2b) - ICMHOME folder might eventually be updated down the road
#/usr/icm-3.9-2b/icm64 icm_generate_CapSelect_files.icm > output_icm_generate_CapSelect_files.log

##Fallback options if no inputs are passed along with the bash script
nproc=16				#Default is 16 CPU cores
sdfPr1=list_1.sdf	#list of MEL fragments 
sdfPr2=list_2.sdf      #list of corresponding fully enumerated REAL cmpds
ICM_BIN=${ICM_BIN:-$(command -v icm64 2>/dev/null)}
#sdfFrag=CapSelect_files/fragments.sdf

while getopts f:P:p:c: flag
do
	case "${flag}" in
	    #f) sdfFrag=${OPTARG};;
	    P) sdfPr1=${OPTARG};;
	    p) sdfPr2=${OPTARG};;
	    c) nproc=${OPTARG};;
  esac
done

echo "NOTE: Starting CapSelectMP ..."
##Check if input files exist
if [ -e "$sdfPr1" ]
  then
    echo "$sdfPr1 exist"
  else
    echo "Please provide the name/path for the MEL sdf using -f list_1.sdf"
    exit
fi

if [ -e "$sdfPr2" ]
  then
    echo "$sdfPr2 exist"
  else
    echo "Please provide the name/path for the MEL sdf using -f list_2.sdf"
    exit
fi

if [ -z "$ICM_BIN" ] || [ ! -x "$ICM_BIN" ]; then
  echo "ERROR: ICM executable not found. Set ICM_BIN to the full path of icm64, for example:"
  echo "  ICM_BIN=/path/to/icm64 bash RMSD.sh -P list_1.sdf -p list_2.sdf -c 4"
  exit 1
fi

if ! [[ "$nproc" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: CPU count must be a positive integer."
  exit 1
fi

##In case fragments.sdf was generate using Windows
#sed -i -e 's/\r$//' $sdfFrag

nof_frags=$(grep -c \$\$\$\$ $sdfPr2)
chunksize=$(((($nof_frags+(($nproc-1))))/$nproc))

echo "Number of Processors: $nproc"
echo "Number of Fragments: $nof_frags"
echo "Chunk Size: $chunksize"

echo "Splitting fragments into $nproc Chunks ..."
## splits fragment.sdf into chunks 
awk -v var=$chunksize '/^\$\$\$\$$/ { if(++delim % var == 0) { next } } { file = sprintf("RMSD/list_2_%s.sdf", int(delim / var )); print > file; }' < $sdfPr2
echo "Sucessfully created  $nproc chunks."

mkdir -p RMSD/tmp
for ((i=0; i<$nproc; i++));
do
  chmod +x RMSD/list_2_${i}.sdf
  if (($i < $(($nproc-1)) ))
    then
      printf "\$\$\$\$\n" >> RMSD/list_2_${i}.sdf
  fi
  mkdir -p RMSD/tmp/RMSD_${i}
  rm -f RMSD/tmp/RMSD_${i}/list_2_fin.sdf
  
  mv RMSD/list_2_${i}.sdf RMSD/tmp/RMSD_${i}/list_2.sdf
  cp $sdfPr1 RMSD/tmp/RMSD_${i}/list_1.sdf
  

  cp RMSD_V_SYNTHES.icm RMSD/tmp/RMSD_${i}/
  cd RMSD/tmp/RMSD_${i}
  #./CapSelect  > output_${i}.log &
  #/usr/icm-3.9-2b/icm64 RMSD_V_SYNTHES.icm list_1.sdf list_2.sdf > output_RMSD_V_SYNTHES.log &
  "$ICM_BIN" RMSD_V_SYNTHES.icm list_1.sdf list_2.sdf | gzip > output_RMSD_V_SYNTHES.log.gz &
  #/usr/icm-3.9-2b/icm64 RMSD_V_SYNTHES.icm list_1.sdf list_2.sdf
  #let PID${i}=${!}
  #let crtpid=PID${i}
  cd ../../../
done

echo "Started $nproc RMSD instances."
echo "Please wait ..."
wait
echo "NOTE: All threads are completed"

output_count=$(find RMSD/tmp -name list_2_fin.sdf | wc -l | tr -d ' ')
if [ "$output_count" -ne "$nproc" ]; then
  echo "ERROR: Expected $nproc completed chunk outputs, found $output_count. Check the ICM logs under RMSD/tmp/."
  exit 1
fi
find RMSD/tmp -name list_2_fin.sdf | sort -t_ -k2,2n | xargs cat > RMSD/RMSD_MP.sdf

echo "Clean up temporary files..."
#rm -r CapSelect_files/tmp

#/usr/icm-3.9-2b/icm64 icm_CapSelect_to_frags_for_enum.icm > output_icm_CapSelect_to_frags_for_enum.log

echo "DONE"


