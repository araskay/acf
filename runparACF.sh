\cd /home/achemparathy/PHANTOM_DATA/PHA_detrendedAFNIsmoothed7mm/
scans=(detrended*.nii)
for scan in ${scans[@]}
do
 qsub /home/achemparathy/PHANTOM_DATA/detrendedAFNIsmoothed7mm.sh $scan
done
