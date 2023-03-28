#!/bin/csh
# not necessary, but example of git ls-files
git ls-files '**/*.2layer_[0-9]/*' '**/*.2layer_[012][0-9]/*' | tr '\n' ' ' > crudnames

set dirs=''
set olddirs=''
foreach f (`seq 0 29`)
    foreach d (default with_storm_mode)
        #git rm -r NSC3km-12sec.$d.rpt_40km_2hr.16n.ep30.f12-f35.bs512.2layer_$f
        set dirs="$dirs NSC3km-12sec.$d.rpt_40km_2hr.16n.ep30.f12-f35.bs512.2layer_$f"
        set olddirs="$olddirs nn_NSC3km-12sec.$d.rpt_40km_2hr.16n.ep30.f01-f48.bs512.2layer_$f"
    end
end
#git commit -m 'bad dirname' $dirs

#git checkout 439e061eeace5122d7830b343e8c57c80c488f3f -- $olddirs


foreach f (`seq 0 29`)
    foreach d (default with_storm_mode)
        # Had to remove nn/ from .gitignore, so the source dir would not be moved into a new subdirectory.
        git mv nn_NSC3km-12sec.$d.rpt_40km_2hr.16n.ep30.f01-f48.bs512.2layer_$f NSC3km-12sec.$d.rpt_40km_2hr.16n.ep30.f12-f35.bs512.2layer_$f
    end
end

