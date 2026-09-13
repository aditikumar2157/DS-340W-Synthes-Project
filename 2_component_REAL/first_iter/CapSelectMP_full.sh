#!/bin/bash

########################################################################
##                                                                    ##
## This script full automates CapSelect ranging from the generation   ##
## of the necessary input files (ICM), the execution of the           ##
## algorithm and generation of frags_for_enum.sdf for enumeration     ##
##                                                                    ##
## Version 02082022  -  Antonina L. Nazarova   nazarova@usc.edu       ## 
##                                                                    ##
########################################################################



## Make CapSelect files directory - All input and output files of CapSelect are stored here
mkdir CapSelect_files

## Execute icm_generate_CapSelect_files.icm using ICM (icm-3.9-2b) - ICMHOME folder might eventually be updated down the road
/home/nazarova/icm-3.9-4/icm64 icm_generate_CapSelect_files.icm > output_icm_generate_CapSelect_files.log

##Fallback options if no inputs are passed along with the bash script
nproc=16				#Default is 16 CPU cores
sdfPr1=CapSelect_files/protein1.sdf	#Default names & path of input files 
sdfPr2=CapSelect_files/protein2.sdf
sdfFrag=CapSelect_files/fragments.sdf

while getopts f:P:p:c: flag
do
	case "${flag}" in
	    f) sdfFrag=${OPTARG};;
	    P) sdfPr1=${OPTARG};;
	    p) sdfPr2=${OPTARG};;
	    c) nproc=${OPTARG};;
  esac
done

echo "NOTE: Starting CapSelectMP ..."
##Check if input files exist
if [ -e "$sdfFrag" ]
  then
    echo "$sdfFrag exist"
  else
    echo "Please provide the name/path for the fragment sdf using -f FRAGMENTS.sdf"
    exit
fi

if [ -e "$sdfPr1" ]
  then
    echo "$sdfPr1 exist"
  else
    echo "No protein1.sdf file"
    exit
fi

if [ -e "$sdfPr2" ]
  then
    echo "$sdfPr2 exist"
  else
    echo "No protein2.sdf file, running CapSelect for 3D Model."
    #read -p "Press any key to proceed if not a 4D model read " -n1 -s
fi

##In case fragments.sdf was generate using Windows
sed -i -e 's/\r$//' $sdfFrag

nof_frags=$(grep -c \$\$\$\$ $sdfFrag)
chunksize=$(((($nof_frags+(($nproc-1))))/$nproc))

echo "Number of Processors: $nproc"
echo "Number of Fragments: $nof_frags"
echo "Chunk Size: $chunksize"

echo "Splitting fragments into $nproc Chunks ..."
## splits fragment.sdf into chunks 
awk -v var=$chunksize '/^\$\$\$\$$/ { if(++delim % var == 0) { next } } { file = sprintf("CapSelect_files/fragment_%s.sdf", int(delim / var )); print > file; }' < $sdfFrag
echo "Sucessfully created  $nproc chunks."

##renames fragments
#mv $sdfFrag orig_$sdfFrag
#mv $sdfPr1 protein1.sdf
#mv $sdfPr2 protein2.sdf

echo "$sdfFrag"
echo "$sdfPr1"
if [ -e "$sdfPr2" ]
  then
    echo "$sdfPr2"
fi

mkdir -p CapSelect_files/tmp
for ((i=0; i<$nproc; i++));
do
  chmod +x CapSelect_files/fragment_${i}.sdf
  if (($i < $(($nproc-1)) ))
    then
      printf "\$\$\$\$\n" >> CapSelect_files/fragment_${i}.sdf
  fi
  mkdir -p CapSelect_files/tmp/CapSelect_${i}
  
  mv CapSelect_files/fragment_${i}.sdf CapSelect_files/tmp/CapSelect_${i}/fragments.sdf
  cp $sdfPr1 CapSelect_files/tmp/CapSelect_${i}/protein1.sdf
  if [ -e "$sdfPr2" ]
    then
      cp $sdfPr2 CapSelect_files/tmp/CapSelect_${i}/protein2.sdf
  fi

  cp CapSelect CapSelect_files/tmp/CapSelect_${i}/
  cd CapSelect_files/tmp/CapSelect_${i}
  ./CapSelect  > output_${i}.log &
  let PID${i}=${!}
  let crtpid=PID${i}
  cd ../../../
done

echo "Started $nproc CapSelect instances."
echo "Please wait ..."
wait
echo "NOTE: All threads are completed"

find . -name CapSelect.sdf | sort -t_ -k2,2n | xargs cat > CapSelect_files/CapSelectMP.sdf

echo "Clean up temporary files..."
#rm -r CapSelect_files/tmp

/home/nazarova/icm-3.9-4/icm64 icm_CapSelect_to_frags_for_enum.icm > output_icm_CapSelect_to_frags_for_enum.log

echo "DONE"


