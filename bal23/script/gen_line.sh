#!/bin/bash
if [[ $# -lt 6 ]]; then
    echo "Usage: $0 <chr> <pos> <bam> <ref> <snapshot> <part>"
    echo "Passed into draw.py: <pos> <alt allele> <genotype>"
    exit 1
fi
name="HG002_$5_$1_$2_gen"
samtools view $3 $1:$2-$2 > $name.sam
#echo -n $name
#echo -n " "
#echo -n $2
#echo -n " "
#echo -n $3
#echo -n " "
#echo $6
python draw.py $name $2 > /dev/shm/csv-result/"result0$5"/"s$6"/$name.csv
rm $name.sam
