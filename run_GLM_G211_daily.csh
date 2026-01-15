#!/bin/csh

if ("$1" != "") set yyyymmdd=$1

if (! $?yyyymmdd) then
    echo define environmental variable yyyymmdd
    exit
endif

if ($yyyymmdd !~ 20[012][0-9][01][0-9][0-3][0-9]) then
   echo yyyyymmdd must be initialization date 
   echo got \"$yyyymmdd\"
   exit
endif

if (! $?TMPDIR) setenv TMPDIR /glade/derecho/scratch/ahijevyc/tmp
set tmp=$TMPDIR/GLM_G211.$yyyymmdd.pbs

cat <<EOS > $tmp
#!/bin/csh
#PBS -N $yyyymmdd
#PBS -A NMMM0021
#PBS -q casper@casper-pbs
#PBS -j oe
#PBS -k eod
#PBS -l select=1:ncpus=20:mem=8GB,walltime=01:30:00

module load conda
conda activate glmval

foreach h (\`seq -w 0 23\`)
    foreach twin (1 2 4)
        python /glade/work/ahijevyc/NSC_objects/GLM_G211.py $yyyymmdd-\$h \$twin --maxbad \$twin
    end
end
EOS

/opt/pbs/bin/qsub $tmp
