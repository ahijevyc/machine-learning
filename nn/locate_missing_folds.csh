#!/bin/csh
# v is list of all 5fold? directories
foreach f (`cat v`)
    printf "$f "
    find -type d -name "$f?" | wc
end
