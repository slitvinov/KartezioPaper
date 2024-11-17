#/bin/sh

for i in *.gv
do dot $i -T png -o "${i%%.gv}.png"
done
