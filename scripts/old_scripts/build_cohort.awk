BEGIN {
    FS = ","
    OFS = ","

    psa_itemid = 50974
}

FNR==NR {
    if (FNR == 1 && $1 ~ /subject_id/) next
    patient[$1] = 1
    next
}

FNR!=NR {

    if (FNR == 1) next

    sid  = $2
    item = $5

    if ((sid in patient) && item == psa_itemid) {

        charttime = $7
        valuenum  = $10
        valueuom  = $11

        print sid, charttime, valuenum, valueuom
    }
}