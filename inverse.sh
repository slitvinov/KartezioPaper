#/bin/sh

set -eu

one () {
    dot "$1" -o /tmp/visual.png  -T png -Gsize=4,4\! -Gdpi=100
    convert /tmp/visual.png -gravity center -background white -extent 400x400 "$2"
}

m=5
g=0
for d in [0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]/
do d=${d%%/}
   cnt=0
   pp= ii0= ii1=
   for i0 in $d/forward.[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9].gv
   do i1=`basename $i0`
      i1=${i1##forward.}
      i1=$d/inverse.$i1
      p0=${i0%%.gv}.ppm
      p1=${i1%%.gv}.ppm

      one $i0 $p0
      one $i1 $p1
      convert $p1 -background black -gravity north -splice 0x4 $p1

      montage -tile 1x2 $p0 $p1 -geometry +0+0 $p0
      if test -f "$pp"
      then if cmp >/dev/null $pp $p0
	   then rm $p0
		continue
	   fi
      fi
      pp=$p0
      cnt=$((cnt+1))
      if test $cnt -eq $m
      then break
      fi
   done
   g=$((g+1))
   convert $d/forward.00000000.ppm -bordercolor red -border 10x10 \
	   -gravity northeast -fill black -pointsize 50 -annotate +10+10 `printf %02d $g` \
	   $d/forward.00000000.ppm
   montage $d/forward.[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9].ppm -tile ${m}x1 -geometry +0+0 $d.png
done
./ffmpeg.sh -o o.mp4 [0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9].png
