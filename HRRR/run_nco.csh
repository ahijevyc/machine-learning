#!/bin/csh
foreach h (`seq -w 5 24`)
cat <<EOS | qsub -V -N $h -A NMMM0021 -q htc -l select=1:ncpus=1:mem=10GB,walltime=02:00:00
#!/bin/csh
module load nco
cd /glade/work/ahijevyc/NSC_objects/HRRR
# Exclude time. Time ends up being a single value associated with first file in the list
# decimal after $h. Makes nco interpret $h. as coordinate value, not index. 
# They are different because forecast_period starts with 1, not 0. So forecast_period = index+1 
ncrcat --no_crd -x -v time -O -d forecast_period,$h.,$h. 20????????_HRRR-ZARR_upscaled.nc $h.nc 
EOS
end
