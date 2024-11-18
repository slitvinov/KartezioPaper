#/bin/sh

set -eu

m=5
g=0
for d in [0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]/
do d=${d%%/}
   cnt=0
   pp= ii=
   for i in ${d}/[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9].gv
   do p=${i%%.gv}.ppm
      dot "$i" -o /tmp/visual.png  -T png -Gsize=4,7\! -Gdpi=100
      convert /tmp/visual.png -gravity center -background white -extent 400x700 $p
      if test -f "$pp"
      then if cmp >/dev/null $ii $i || cmp >/dev/null $pp $p
	   then rm $p
		continue
	   fi
      fi
      pp=$p ii=$i
      cnt=$((cnt+1))
      if test $cnt -eq $m
      then break
      fi
   done
   g=$((g+1))
   convert $d/00000000.ppm -bordercolor red -border 10x10 \
	   -gravity northeast -fill black -pointsize 50 -annotate +10+10 `printf %02d $g` \
	   $d/00000000.ppm
   montage $d/[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9].ppm -tile ${m}x1 -geometry +0+0 $d.png
done
