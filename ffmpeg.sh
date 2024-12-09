#!/bin/sh

: "${FFMPEG=ffmpeg}"
prog=co.ffmpeg
msg () { echo >&2 "$@" ; }
err () { msg "$prog: $@"; exit 2 ; }
e   () {
    if test "$LOG" != 0
    then msg ": $@"
    fi
    eval "$@"
}

filep() { if test ! -f "$1"; then err "not a file '$1'"; fi; }
intp()  { echo "$1" | grep -q '^[0-9][0-9]*$'; }
if test $# -ne 0 && test "$1" = -h
then
    cat <<'!'
Usage: $prog [-r INT] -o OUTPUT.mp4 IMAGES...

Options:
  -r rate    Set frame rate (default: 5)
!
    exit 1
fi

if ! $FFMPEG -version >/dev/null 2>/dev/null
then err "$FFMPEG command not found"
fi

Output= R=5
while test $# -ne 0
do case "$1" in
       -o) shift
	   if test $# -eq 0; then err '-o needs a file'; fi
	   Output=$1; shift
	   ;;
       -r) shift
	   if test $# -eq 0; then err '-r needs an argument'; fi
	   R=$1; shift
	   ;;
       -*) err "unknown option '$1'" ;;
       *) break ;;
   esac
done

if test -z "$Output"; then err "OUTPUT.mp4 is not given"; fi
if test $# -eq 0; then err 'no IMAGES..'; fi
for i; do filep "$i"; done
if ! intp "$R"; then err "not an integer '$R'"; fi
t=/tmp/co.ffmpeg.$$
trap 'rm -f $t; exit 2' 1 2 3 15
for i; do printf 'file %s\n' "$i"; done > $t
ffmpeg \
  -fflags +genpts \
   -r $R \
  -safe 0 -f concat -i $t \
  -y \
  -loglevel error \
  -vcodec libx264 -crf 1 -pix_fmt yuv420p "$Output"
code=$?
rm -f $t
exit $code
