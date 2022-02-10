#!/bin/bash
yyyymmddhh=$1
cat <<EOS | qsub -V -N $yyyymmddhh -A NMMM0021 -q htc -l select=1:ncpus=5:mem=4GB,walltime=00:05:00
#!/bin/csh
module load ncarenv conda
conda activate
/glade/work/ahijevyc/NSC_objects/upscale_HRRR-ZARR.py $*
EOS
