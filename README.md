V-SYNTHES2 Pipeline (Prepared by Dr. Antonina Nazarova, Dr. Anastasiia Sadybekov, Dr. Arman Sadybekov)
This directory contains scripts necessary for executing the V-SYNTHES2 workflow (located in the scripts/ folder), along with example files derived from the Enamine REAL Space library for demonstration purposes. The receptor model used here is the Rho receptor, thoroughly characterized within the manuscript. The included examples represent two- and three-component reaction subsets of Enamine REAL Space. The current REAL Space library contains approximately 36 billion fully enumerated molecules across more than 164 well-established parallel synthesis reactions (for comprehensive information for your specific REAL library implemented in screening, please contact Enamine).

Minimum Requirements:

Execution of V-SYNTHES scripts and subsequent docking analyses require an ICM-Pro+VLS license, available from MolSoft (https://www.molsoft.com/products.html). Scripts included in this directory were validated using ICM-Pro+VLS version 3.9-4a running on Ubuntu Linux version 18.04.

Directory structure and contents:

Example MEL fragment files for 2-component (r_11__const_frag_1.sdf and r_11__const_frag_2.sdf) and 3-component (r_275592__const_frag_12.sdf, r_275592__const_frag_13.sdf) projects are provided within subdirectories under 2_component_REAL/first_iter/files/ and 3_component_REAL/first_iter/files/, specifically. Complete versions of these datasets can be requested directly from Enamine LLC. Reaction libraries essential for both 2- and 3-component reactions are included as 2_component_REAL/SYNTHONS_REACTIONS_REAL.icb and 3_component_REAL/SYNTHONS_REACTIONS_REAL.icb. Synthons datasets for both 2- and 3-component reactions are included as 2_component_REAL/first_iter/synthons/ and 3_component_REAL/first_iter/synthons/.

Corresponding procedures:

To apply the V-SYNTHES 2.0 workflow using the provided examples, follow the detailed instructions outlined in Figure 1 of the manuscript, specifically described herein in Procedure 1 (V-SYNTHES2 Automated Run). For statistical evaluations of docking poses and conformational analyses of enumerated hits, refer to Procedure 2 (RMSD Assessment between MEL fragments and Corresponding Enumerated Molecules) discussed here as well. For generation of the random REAL subset for benchmarking analysis, refer to Procedure 1, stage VIII (Random Sampling of the REAL Library for Statistical Analysis of Enrichment values).

What's newly added:

In addition to revised scripts from the V-SYNTHES 1.0 version, V-SYNTHES2 includes 1) a newly developed automated algorithm, CapSelect. CapSelect is specifically designed to optimize and automate the step (2b) that was performed manually in the original V-SYNTHES 1.0 workflow; 2) complete RMSD assessments between MEL fragments and corresponding enumerated molecules for different protein targets and REAL reaction-based subspaces.


PROCEDURE 1: V-SYNTHES2 AUTOMATED RUN

Preliminary stage: V-SYNTHES2 Project Repository Preparation/Installation of Docking Software/ Timing: Uncertain
1.	Install ICM-Pro Desktop Modeling Software on your Linux workstation and obtain the necessary ICM licenses for your HPC cluster and Linux workstation (installation info here: https://www.molsoft.com/icmpro/faq-installation.html).
2.	Request the current REAL Space from Enamine at libraries@enamine.net,  including MEL fragment library (.sdf) and corresponding synthons (.sdf) from Enamine LLC . Upload these to a secure directory on your cluster.
3.	Create a directory named V-SYNTHES2_project on your Linux workstation. Copy the scripts and the run/ folder from the V-SYNTHES2 workflow repository on GitHub into this directory.
Stage I: Docking Project Setup for V-SYNTHES2 / Timing: 10 min
4.	Obtain a high-resolution structure of a protein target.
5.	Open the file containing the structure in ICM-Pro and prepare it for virtual screening following the procedure in the ICM manual (https://www.molsoft.com/gui/start-dock.html). The conformational model can be validated by docking of known ligands and/or optimized by ligand-guided procedures (e.g. LiBERO49 or MD-based refinement50).
6.	Follow the step-by-step workflow in Molsoft ICM for generating 3D potential grids51.
Upon completion, the docking grids will be generated in the selected project folder. This should include six .map files, one _probe.ob file, and one _rec.ob file, a .dtb file containing customized docking parameters, and a LOG.htm file.
Box 1 | Example of V-SYNTHES2_project Input Directory on Your Linux Workstation
This example illustrates the setup for a screening campaign targeting the cPLA2 receptor (PDB ID: ).

V-SYNTHES2_project/
├── run/
│   ├── RHO_1u19_reg.dtb
│   ├── RHO_1u19_reg_gb.map
│   ├── RHO_1u19_reg_gc.map
│   ├── RHO_1u19_reg_ge.map
│   ├── RHO_1u19_reg_gh.map
│   ├── RHO_1u19_reg_gl.map
│   ├── RHO_1u19_reg_gs.map
│   ├── RHO_1u19_reg_LOG.htm
│   ├── RHO_1u19_reg_probe.ob
│   ├── RHO_1u19_reg_rec.ob
├──input_VSYNTHES.in
├──icm_docking_launcher.icm
├──icm_load_hits_Enamine.icm
├──CapSelect (executable)
├──CapSelectMP_full.sh
├──icm_generate_CapSelect_files.icm
├──icm_CapSelect_to_frags_for_enum.icm
├──icm_enumerate_REAL_frags_Enamine.icm
└──icm_postprocessing.icm
7.	Move the generated docking project grids to the run/ folder within the V-SYNTHES2_project/ directory on your Linux workstation.
Stage II: Molecular Docking of Minimal Enumeration Library (V-SYNTHES2, Figure 1, Step 2a)/ Timing: 5-50 hours (~10 sec * #MEL fragments/#CPU cores on average,  total time depending on your setup)
Docking in V-SYNTHES2 is performed using Molsoft ICM docking program (https://www.molsoft.com/docking.html). The docking_launcher.icm script initiates docking of prepared in advance MEL fragments in .sdf format into the receptor docking maps. The docking output in .ob format will be saved in the run/ in the current working directory
8.	Modify input parameters for docking_launcher.icm script in  input_VSYNTHES.in:  
DOCKING_SCORE_THRESHOLD is used in docking algorithm to save only molecules with predicted scores lower than the specified value. Selection of correct score threshold allows to save sufficient number of hits for the next step, while keeping the required hard drive space usage low.
ICM_HOME – path to ICM-Pro home directory.
EFFORT_DOCKING – docking effort. Higher effort provides more extensive sampling and takes more time to finish. Docking effort is recommended to be set to 2, as a higher effort would make screening of MEL fragments more time and resource-consuming without significant improvement in pose predictions.
NCPU_DOCKING – number of  threads to use in docking.
PROJECT_PATH – docking project directory where receptor docking maps are stored. Default is run/ in the current working directory.
SDF_FOR_DOCKING_PATH – path to library of small molecules for docking in .sdf format. Default is files/ in the current working directory.
DOCKING_SCORE_THRESHOLD 	"-12."  
ICM_HOME       		"/usr/icm-3.9-3b"       
EFFORT_DOCKING 		"2."    
NCPU_DOCKING   		"31"    
PROJECT_PATH    		""      
SDF_FOR_DOCKING_PATH    	""   
Setting an appropriate docking score threshold for MEL docking is crucial for identifying a sufficient number of high-scoring fragments for enumeration. Depending on the reaction-based REAL space, this number can range from approximately 1K to 15K fragments (Supporting Information, Table SI2).
9.	Perform MEL docking: 
/path_to_ICM/icm64 icm_docking_launcher.icm
It is reccomended to perfom the docking step of full V-SYNTHES pipeline on a Linux cluster with multi-core CPU nodes. The provided script is designed to run on a single Linux workstation. The user should modify the script to work with a specific cluster architecture and workload manager.
CRITICAL STEP It is important to verify that all fragments in each .sdf/.inx file have been docked. A significant discrepancy between the number of docked molecules reported in .log files and the number of ligands provided as input files could mean that docking jobs were terminated prematurely. To count the numbr of docked ligands in the .log files, count the number of  "SCORE" entries reported in the .log outputs and that it matches the number of molecules in the corresponding .sdf files. As shown in Supporting Information Table SI5, the total number of docked compounds differs on average by no more than 0.2% from the original MEL datasets. The difference comes from molecules for which 3D pose was not generated, which could happen due to failure of conformers generation or some other failure during the docking algorithm, and >1% failure rate could indicate issues during docking.
Stage III: Generation of the Hitlist from the docked MEL (V-SYNTHES2, Figure 1, Step 2a continuation)/ Timing: 5 min – 3 hours (depending on the REAL Space reaction-based subset and on specified score threshold)
Hitlist generation step takes raw .ob output files from docking procedure and receptor docking maps and generates interactive table with docking scores and poses for best hit molecules ready for the next steps of V-SYNTHES pipeline and suitable for manual inspection by user.
10.	Modify input parameters for icm_load_hits_Enamine.icm script in input_VSYNTHES.in:  
PATH_INX_FILES – path to indexed .sdf files which were used for docking, default in files/ in the current working directory.
LOAD_SCORE_THRESHOLD – specifies score threshold used to save molecules for next step. Only hits with scores lower than the threshold value will be considered for further analysis.
PATH_INX_FILES 		""      
LOAD_SCORE_THRESHOLD   	"-12."
11.	Once the docking jobs are finished, navigate to the root folder V-SYNTHES2_project/ and generate hitlist of best scoring molecules: 
/path_to_ICM/icm64 icm_load_hits_Enamine.icm
	CRITICAL STEP It is crucial to set the docking threshold value to ensure that enough MEL fragments are saved for the next selection step, usually at least 1% of the MEL library. In general, the default docking threshold  -12 saves about 5% of the MEL 2-component subspace and is sufficient in most cases. If the number of saved MEL fragments is too low, the LOAD_SCORE_THRESHOLD value can be increased.  Additional filters like LogP, LogS, or MW can be applied by modifying icm_load_hits_Enamine.icm. 
The number of MEL fragments saved can be found in output:
total number of molecules provided for docking:1710727
number of hits after applying score threshold:45537 
Once the hitlist is generated the following files will be saved in VSYNTHES2_project/run/processing_files/ folder:
-	project_name_root_folder_name.icb contains interactive table with the best scoring compounds (30,000 molecules maximum). ICM graphical user interface (/path_to_ICM/icm64 -g in the terminal) can be used to inspect docking poses and scores of saved hits in .icb file.  
-	frags_for_enum.sdf contains best scoring molecules that will be used in the next step.
-	three .tab files contain MW, logP, and docking score distribution histograms of the hits. 
Stage IV: Selection of Productive MEL Fragments (V-SYNTHES2, Figure 1, Step 2b)/ Timing: 30min (if running in parallel on 16+ CPUs)
12.	 To run the CapSelect algorithm on a local Linux workstation, the user should execute the CapSelectMP_full.sh bash script from the base directory, where all necessary script files are located. This script will perform the following: 
o	Run icm_generate_CapSelect_files.icm to create protein.sdf and fragments.sdf input files from the project_name_+root_folder_name.icb file.
o	Calculate parameters for each docked MEL fragment, as detailed in Table 1.
o	Run icm_CapSelect_to_frags_for_enum.icm to generate the final output frags_for_enum.sdf file.
To execute CapSelectMP_full.sh from the root folder:
./CapSelectMP_full.sh -c <number of cores> 
Example of running on Linux workstation:
./CapSelectMP_full.sh -c 30 
CRITICAL STEP1 The default parallel processing core count is 16. If your workstation supports more, specify the higher core count, considering RAM availability.
CRITICAL STEP2 For a 4D receptor model with two receptor structures, the CapSelect algorithm will automatically generate protein1.sdf and protein2.sdf input files. It is recommended not to use more than two receptor structures in a 4D model for V-SYNTHES2 screening.
13.	Upon execution of the CapSelect algorithm, the frags_for_enum.sdf output file will be automatically created in the V-SYNTHES2_project/run/processing_files/ folder. This file includes all MEL fragments from the project_name_+root_folder_name.icb file, plus five additional columns for newly calculated parameters: <Score2>, <CapScore>, <Spheres>, <Comment>, and <Distance>. The MEL fragments are sorted in descending order based on <Score2> (MergedScore), which is calculated according to equation (3).
?TROUBLESHOOTING
The CapSelect algorithm actively writes output files containing the coordinates of calculated equidistant spheres that represent available space within the binding pocket. These files are generated simultaneously with the execution of the CapSelect_MP.sh script and saved in the V-SYNTHES2_project/CapSelect_files/tmp/CapSelect_*/output_*.log file. Users can utilize any molecular modeling software to visualize these spheres, representing potential "methyl groups." (Figure 3).
?TROUBLESHOOTING
The provided CapSelect executable is compiled for Ubuntu 20.04. If you have an older version or a different Linux distribution, compile it on your terminal using the following command and the CapSelect.cpp program file before running CapSelect_MP.sh:
g++ -o CapSelect CapSelect.cpp
Stage V: Enumeration of Selected MEL Fragments (V-SYNTHES2, Figure 1, Step 3)/ Timing: 1 hour – 8 hours (1Mln Enumerated Subset)
The enumeration protocol generates enumerated compounds from the REAL space by sequentially enumerating selected MEL fragments from frags_for_enum.sdf file. The number of MEL fragments required depends on the reaction subset (2-component or 3-component) and the desired number of enumerated molecules. Partially or fully enumerated molecules are saved in .sdf format with corresponding index .inx files ready for next step docking.
14.	Modify input parameters for icm_enumerate_REAL_frags_Enamine.icm script in input_VSYNTHE.in:  
MAX_MOLS – specifies maximum number of enumerated molecules in the output. It is recommended to set MAX_MOLS to between "1000000" and "2000000".
CHUNKS_COUNT – controls the number of enumerated molecules in each output .sdf file. The number of molecules in individual .sdf file is calculated as ratio MAX_MOLS/CHUNKS_COUNT.
FLT_MW – Molecular weight filter applied on the final enumeration step. Only molecules with MW below specified value will be generated. Default value is 500.
FLT_3COMP – Molecular weight filter applied on the first enumeration step for 3-component reaction. Only molecules with MW below specified value will be generated. Default value is 400.
ADDITIONAL_FILTERS – additional filters applied to generate enumerated molecules, including, but not limited to logP, logS, number of rotatable bonds, number of hydrogen bond donors and accepters. 
OUTPUT_SUFFIX – suffix used to generate names of output .sdf files
MAX_MOLS        	"1000000"       
CHUNKS_COUNT    	"20"    
FLT_MW 		 	"500."
FLT_3COMP       	"400."
ADDITIONAL_FILTERS	"MolLogP(mol)<5. & Nof_RotB(mol)<11 &  Nof_HBD(mol)<6 & Nof_HBA(mol)<11"
OUTPUT_SUFFIX   	"2comp_final"     
15.	To start the enumeration in the V-SYNTHES2_project/ folder:
/path_to_ICM/icm64 icm_enumerate_REAL_frags_Enamine.icm 
During enumeration, MEL fragments from frags_for_enum.sdf will be connected with their corresponding counter-synthons based on synthon_id_X and rxn_ID until either all selected MEL fragments are used or specified maximum number of enumerated molecules is reached, whichever happens first. The output will be saved in /run/processing_files/:
-	enumerated_best_frags_PROJECT_NAME_OUTPUT_SUFFIX_MAX_MOLS contains .sdf files with enumerated molecules for each individual MEL fragment.
-	enumerated_PROJECT_NAME_OUTPUT_SUFFIX_chunks_chunk_size contains indexed .sdf files with enumerated molecules split into chunks ready for docking.
-	enumeration_results.tab file contains the number of molecules enumerated from each MEL fragment.
Stage VI: Molecular Docking of The Enumerated Dataset (V-SYNTHES2, Figure 1, Step 4)/ Timing: 10 hours – 50 hours (~10 sec/CPU core/MEL fragment on average,  total time depending on your setup)
16.	Modify input parameters for docking_launcher.icm script in  input_VSYNTHES.in as described in step 8. Use .sdf/.inx files in enumerated_PROJECT_NAME_OUTPUT_SUFFIX_chunks_chunk_size generated on the previous step as input files for docking.
17.	Start docking of enumerated molecules as described in step 9.
Stage VII: Generation of interactive hitlist from the docking of enumerated molecules  (Figure 1, Step 4)/ Timing: 5 min – 3 hours (depending on the size of chunks in step 14 and specified LOAD_SCORE_THRESHOLD)
18.	Modify input parameters for icm_load_hits_Enamine.icm script in input_VSYNTHES.in as described in step 10.
19.	Once the docking of enumerated molecules is finished, generate the hitlist of best-scoring molecules as described in step 11.
CRITICAL STEP. To generate fully enumerated molecules from 3-component reaction REAL subset repeat Stages IV-VII. 
Stage VIII: Postprocessing (optional, custom-based)/ Timing: 1 hour – 3 hours
Postprocessing includes removing compounds with sub-optimal predicted properties for drug discovery as calculated in ICM-Pro software. Default filters remove compounds with the following properties, as specified in input_VSYNTHES.icm: 
-	LogS < -5
-	Predicted Pan Assay Interference Compound (PAINS) with value > 0.5 (that indicates high probability of being a PAINS Compound)
-	Tox_Score >1 (indicates substructure/substituent that is flagged as unfavorable) and Tox_Class > 0.5 (indicates toxic compound). The Tox Score gives a prediction for how reactive or toxic a chemical is.
-	Bad groups – unwanted or reactive chemical functionality.
20.	Begin postprocessing: 
/path_to_ICM/icm64 icm_postprocessing.icm
The output will be two files:
-	project_name_root_folder_name_properties.icb – contains interactive hitlist table with all molecules from the hitlist and their calculated properties.
-	project_name_root_folder_name_clean.icb – contains interactive hitlist with only the hits that satisfy the filters above.
Supporting Information, Table SI6 shows a percentage of compounds that do not satisfy each of the above filters from the top 10,000 hits received from V-SYNHTES2 screenings for AT2, cPLA2, Rho proteins.
CRITICAL STEP It is recommended to ensure novelty and diversity of the selected for synthesis compounds by removing all hits with Tanimoto distance <0.3 to known high affinity ligands, performing chemical clustering of the remaining hits and selecting representative molecules from the best clusters. Table SI6 shows the number of clusters and number of singletons in the top 10,000 hits received from V-SYNHTES2 screenings for AT2, cPLA2, Rho proteins with Tanimoto distance 0.3 using Molsoft’s and ECFP fingerprints. 
Stage VIII: Random Sampling of the REAL Library for Statistical Analysis of Enrichment values/ Timing: 30 min 
This stage is performed only for benchmarking purposes of V-SYNTHES2 results and is not required for users running V-SYNTHES2 to identify hits.
For the fast generation of random datasets, all synthons should be saved to a Molcart database (.molt format). 
1.	Modify input parameters in icm_generate_molt.icm script: 
path_to_files – path to .sdf files containing all synthons
molt_dir – path to the output folder
molt_name – name of the output .molt file
2.	Generate molcart database:
/path_to_ICM/icm64 icm_generate_molt.icm
3.	Modify input variables in icm_generate_rnd_lib_new_fmt.icm script:
MAX_MOLS – maximum number of enumerated molecules in the output
FLT_MW – molecular weight filter applied during enumeration of random subset
PERCENT_DB_STR – percentage of selected for enumeration synthons from each reaction divided by 100. 
comp_str – variable that specifies from which space the random subset will be generated “2comp” or “3comp”
ADDITIONAL_FILTERS – additional filters applied during enumeration. 
path_to_reactions – path to .icb file with the definition of the reactions
molt_synthons_file – path to .molt file with synthons 
output_folder – path to the folder where output files will be saved. Can be empty, then output will be written to the current folder.
4.	Generate random datasets from REAL space:
/usr/icm-3.9-2b/icm64 icm_generate_rnd_lib_new_fmt.icm
The random subsets generated will be saved in .molt format. 
CRITICAL STEP The PERCENT_DB_STR parameter determines percent of synthons selected from each reaction at each attachment point for enumeration. To generate one million properties matched (MW<500 and logP<5) enumerated 2-component subset, 2.5% synthons at each attachment point from each reaction were selected for enumeration. To generate one million properties matched (MW<500 and logP<5) enumerated compounds from 3-comp subset, 3.7% synthons at each attachment point from each reaction were selected for enumeration. Adjustments are required based on the desired dataset size. 


PROCEDURE 2: RMSD ASSESSMENT BETWEEN MEL FRAGMENTS AND THEIR CORRESPONDING ENUMERATED MOLECULES 

Preliminary stage: RMSD assessment for V-SYNTHES2 Project Repository Preparation/Installation of Docking Software/ Timing: 0.5-1 hours
First 2 steps of Procedure 2 are performed the same as for Procedure 1, which follows by steps:
3. Create a directory named V-SYNTHES2_RMSD_project on your Linux workstation. Copy the scripts from the V-SYNTHES2_RMSD_project repository on GitHub into this directory.
4. Activate the Python 3 environment for data analysis of output .sdf files..
Box 3 | Example of V-SYNTHES2_RMSD_project Input Directory on Your Linux Workstation
Below are the scripts for running RMSD assessments between MEL fragments and corresponding enumerated molecules. Files list_1.sdf and list_2.sdf are input files generated as described in Step 5 below.
V-SYNTHES2_RMSD_project/
├──list_1.sdf
├──list_2.sdf
├──RMSD.sh
├──RMSD_V_SYNTHES.icm
└──SMILES_generator_REAL.icm
Stage I: Generation of Input Files for RMSD Accessment in V-SYNTHES2 / Timing: 10 min
1.	Calculation Calculate RMSD between MEL fragments and their corresponding enumerated molecules after V-SYNTHES 2.0 screening. Save standalone tables in .sdf format from the project_name_+root_folder_name.icb file obtained after MEL fragment docking (as list_1.sdf) and from the file obtained after docking of fully enumerated molecules from REAL Space  (as list_2.sdf). Both files should be saved in the V-SYNTHES2_RMSD_project folder.
Stage II: RMSD Accessment in V-SYNTHES2 / Timing: 1-6 hours (depending on the REAL Space reaction subset)
2.	Run SMILES_generator_REAL.icm from the root folder, specifying the directory for the REAL synthons .sdf files library:
/usr/icm-3.9-2b/icm64 SMILES_generator_REAL.icm > output_SMILES.log &
This script adds a SMILES column to each synthon .sdf file in the REAL Space dataset to ensure accurate substructure comparison for RMSD calculation.
3.	To run RMSD calculations between MEL fragments and their corresponding enumerated molecules, use the RMSDMP_full.sh script to run RMSD_V_SYNTHES.icm in parallel on several CPUs. The default setting is 16 CPUs, but users should adjust this number based on their workstation's RAM capacity; in this instance, the adjustment was made to use 4 CPUs: 
./RMSDMP_full.sh -c 4
4.	The calculated RMSD will be added as an RMSD column to list_2.sdf, reflecting the RMSD between the compound binding pose and the corresponding MEL fragment substructure. This output file, now named RMSD_MP.sdf, will be saved in the V-SYNTHES2_RMSD_project/RMSD folder.
CRITICAL STEP1 When calculating RMSD between MEL fragment binding poses and their corresponding enumerated molecules—either after the initial enumeration or between the first and final enumerated subsets for 3-component REAL Space reaction subsets—the subsequently higher enumerated structures should be designated as list_2.sdf.
5.	Once the run is finished, visualize the obtained RMSD values using the provided RMSD_visualization.py script, which will generate a violin plot with median, interquartile range, and deviation values.

