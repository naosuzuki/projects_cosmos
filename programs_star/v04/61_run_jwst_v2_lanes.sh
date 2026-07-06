#!/bin/bash
# JWST v2.1 campaign: f444w pilot verify, then 3 lanes over all tile-bands.
cd /Users/suzuki/github/projects_cosmos/programs_star/v04
PY=/opt/miniconda3/bin/python
W=/Volumes/exdisk1/data/photometry_v04
LOG=/tmp/overnight
mkdir -p $LOG

site_refresh() {
  if mkdir /tmp/site.lock 2>/dev/null; then
    $PY 55_make_psf_qa_site.py >/dev/null 2>&1
    rmdir /tmp/site.lock
  fi
}

jwst_one() {   # skip build if meta already v2; skip QA if thumb exists
  local t=$1 b=$2 d=$W/jwst_nircam_$b/$t/psf
  if ! grep -q "hostgalxy-consistent" "$d/psf_$b.meta.json" 2>/dev/null; then
    $PY 54_step3a_build_psf_model.py --instrument jwst_nircam_$b --tile $t --filter $b --reuse-pass1 \
        >> $LOG/jwst_build.log 2>&1 || { echo "FAILBUILD $t $b" >> $LOG/jwst.log; return 1; }
  fi
  if [ ! -e "$d/psf_samples_thumb.png" ] || [ "$d/psf_$b.meta.json" -nt "$d/psf_samples_thumb.png" ]; then
    $PY 56_make_psf_qa_plots_single.py --instrument jwst_nircam_$b --tile $t --reuse-outcat \
        >> $LOG/jwst_qa.log 2>&1 || { echo "FAILQA $t $b" >> $LOG/jwst.log; return 1; }
  fi
  rm -f $d/pass1_$b.fits $d/samp*.fits $d/resi*.fits $d/snap*.fits $d/proto*.fits
  echo "DONE $t $b $(date +%H:%M)" >> $LOG/jwst.log
  site_refresh
}

# pilot completion: f444w (f115w already v2.1)
jwst_one A4 f444w
n=$($PY -c "import json;print(json.load(open('$W/jwst_nircam_f444w/A4/psf/psf_f444w.meta.json')).get('n_model_stars') or 0)" 2>/dev/null)
echo "PILOT2 f444w n_model_stars=$n" >> $LOG/jwst.log
if [ "${n:-0}" -lt 30 ]; then
  echo "PILOT2 FAILED — lanes NOT started" >> $LOG/jwst.log
  exit 1
fi
lane() { for t in $2; do for b in f115w f150w f277w f444w; do jwst_one $t $b; done; done; echo "LANE$1 DONE $(date +%H:%M)" >> $LOG/jwst.log; }
lane 1 "A1 A2 A3 A4 A5 A6 A7"   & L1=$!
lane 2 "A8 A9 A10 B1 B2 B3 B4"  & L2=$!
lane 3 "B5 B6 B7 B8 B9 B10"     & L3=$!
wait $L1 $L2 $L3
site_refresh
echo "JWST CAMPAIGN COMPLETE $(date '+%Y-%m-%d %H:%M')" >> $LOG/jwst.log
