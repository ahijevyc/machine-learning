#!/bin/csh

foreach f (`seq 0 29`)
    foreach s (with_storm_mode default)
        #git mv -f trainend20160701.NSC/NSC3km-12sec.$s.rpt_40km_2hr.16n.ep30.f12-f35.bs512.2layer_$f trainend20160701.NSC/NSC3km-12sec.$s.rpt_40km_2hr.16n16n.ep30.f12-f35.bs512_$f 
    end
end

foreach n (15 16 64 256 512 1024)
    find -name "*.${n}n.*.2layer*" -exec rename -v .${n}n. .${n}n${n}n. {} \;
end
find -name "*.2layer*" -exec rename -v .2layer '' {} \;
