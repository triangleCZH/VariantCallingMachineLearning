#from_to snapshot_num part_num
if [ $# != 4 ]
then
  echo "Usage: ./run_line.sh [from_line] [to_line] [snapshot_number] [part_number]"
  exit 1
fi
sed -n "$1,$2p" ../data/snapshot_0$3.filter.label.csv | cut -d'	' -f17,18 > tmp_$3_$1_$2.txt
sed -i 's/      / /g' tmp_$3_$1_$2.txt
count=0
begin=$1
end=$1
while read line
do
  ((count=count+1))
  #((end=end+1))
  #echo $line
  #echo $count
  chr=`echo $line | cut -d' ' -f1`
  pos=`echo $line | cut -d' ' -f2`
  echo -n $chr >> /dev/shm/csv-result/log/s$3p$4.log
  echo -n " " >> /dev/shm/csv-result/log/s$3p$4.log
  echo -n $pos >> /dev/shm/csv-result/log/s$3p$4.log
  echo -n " " >> /dev/shm/csv-result/log/s$3p$4.log
  echo $count >> /dev/shm/csv-result/log/s$3p$4.log
  #echo -n "from "
  #echo -n $begin
  #echo -n "to "
  #echo $end

  ./gen_line.sh $chr $pos ../data/HG002.bam  ../data/ucsc.hg19.fasta $3 $4 #this is the part number
  if [ $count -eq 400 ]
  then
    sed -n "${begin},${end}p" ../data/snapshot_0$3.filter.data.csv > /dev/shm/csv-result/result0$3/s$4/$3_$1_$2.data.csv
    sed -n "${begin},${end}p" ../data/snapshot_0$3.filter.label.csv > /dev/shm/csv-result/result0$3/s$4/$3_$1_$2.label.csv
    echo "extract starts"
    python3 /dev/shm/csv-result/result0$3/s$4/extract$3$4.py >> /dev/shm/ml-result/log/s$3p$4.log
    find /dev/shm/csv-result/result0$3/s$4/ -name '*.csv' | xargs rm -f
    echo "$count Reset to 0"
    count=0
    ((begin=end+1))
  elif [ $end -eq $2  ]
  then
    sed -n "${begin},${end}p" ../data/snapshot_0$3.filter.data.csv > /dev/shm/csv-result/result0$3/s$4/$3_$1_$2.data.csv
    sed -n "${begin},${end}p" ../data/snapshot_0$3.filter.label.csv > /dev/shm/csv-result/result0$3/s$4/$3_$1_$2.label.csv
    echo "extract starts"
    python3 /dev/shm/csv-result/result0$3/s$4/extract$3$4.py >> /dev/shm/ml-result/log/s$3p$4.log
    find /dev/shm/csv-result/result0$3/s$4/ -name '*.csv' | xargs rm -f
    echo "$count Reset to 0"
    count=0
  fi
  ((end=end+1))
done < tmp_$3_$1_$2.txt
rm tmp_$3_$1_$2.txt
