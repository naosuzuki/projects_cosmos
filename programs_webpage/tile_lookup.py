"""Fast coordinate→tile resolver backed by tile_footprints.parquet.

No FITS opened for resolution — pure in-memory bbox test against precomputed
WCS footprints. Use this to (a) find which tile-file contains a coordinate,
and (b) group a set of positions by tile so each FITS is opened exactly once.

Scales to millions of positions: resolution is vectorized numpy, IO is one
open per (survey,band,tile) that has ≥1 requested point.
"""
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path

FOOTPRINTS = Path("/Users/suzuki/github/projects_cosmos/csvfiles_sn/tile_footprints.parquet")

# band aliases so callers can pass either canonical or file-style band names
_BAND_ALIAS = {"F814W":"F814W",
               "F115W":"F115W","F150W":"F150W","F277W":"F277W","F444W":"F444W",
               "VIS":"VIS","Y":"NIR-Y","J":"NIR-J","H":"NIR-H",
               "NIR-Y":"NIR-Y","NIR-J":"NIR-J","NIR-H":"NIR-H"}


class TileResolver:
    def __init__(self, path=FOOTPRINTS):
        t = pq.read_table(path).to_pandas()
        self.t = t
        # pre-split by (survey, band) for fast vectorized membership
        self._idx = {}
        for (sv, bd), grp in t.groupby(["survey","band"]):
            self._idx[(sv, bd)] = grp.reset_index(drop=True)

    def resolve(self, survey, band, ra, dec, margin_arcsec=2.0):
        """Return (tile, path) whose footprint contains (ra,dec), else (None,None).
        `margin` expands the bbox so points near an edge still resolve (cutout
        will be partial-filled). Vectorizable but here scalar for clarity."""
        bd = _BAND_ALIAS.get(band, band)
        grp = self._idx.get((survey, bd))
        if grp is None or len(grp)==0: return (None, None)
        m = margin_arcsec/3600.0
        cosd = np.cos(np.deg2rad(dec))
        mr = m/max(1e-6, cosd)
        hit = ((ra >= grp["ra_min"]-mr) & (ra <= grp["ra_max"]+mr) &
               (dec >= grp["dec_min"]-m) & (dec <= grp["dec_max"]+m))
        idx = np.where(hit.values)[0]
        if len(idx)==0: return (None, None)
        # if multiple tiles overlap, pick the one whose center is closest
        if len(idx)>1:
            cra = 0.5*(grp["ra_min"].values+grp["ra_max"].values)[idx]
            cdec= 0.5*(grp["dec_min"].values+grp["dec_max"].values)[idx]
            d = ((cra-ra)*cosd)**2 + (cdec-dec)**2
            idx = [idx[int(np.argmin(d))]]
        r = grp.iloc[int(idx[0])]
        return (str(r["tile"]), str(r["path"]))

    def group_by_tile(self, survey, band, positions):
        """positions: list of (key, ra, dec). Returns {path: [(key,ra,dec),...]}
        so each FITS is opened exactly once. Points off all footprints are
        dropped (caller can detect missing keys)."""
        groups = {}
        bd = _BAND_ALIAS.get(band, band)
        grp = self._idx.get((survey, bd))
        if grp is None: return groups
        ramin=grp["ra_min"].values; ramax=grp["ra_max"].values
        decmin=grp["dec_min"].values; decmax=grp["dec_max"].values
        paths=grp["path"].values
        for key, ra, dec in positions:
            cosd=np.cos(np.deg2rad(dec)); mr=2.0/3600/max(1e-6,cosd); m=2.0/3600
            hit=np.where((ra>=ramin-mr)&(ra<=ramax+mr)&(dec>=decmin-m)&(dec<=decmax+m))[0]
            if len(hit)==0: continue
            if len(hit)>1:
                cra=0.5*(ramin+ramax)[hit]; cdec=0.5*(decmin+decmax)[hit]
                hit=[hit[int(np.argmin(((cra-ra)*cosd)**2+(cdec-dec)**2))]]
            p=str(paths[int(hit[0])])
            groups.setdefault(p, []).append((key, ra, dec))
        return groups


if __name__ == "__main__":
    # self-test: resolve the 56 master SNe and report unique FITS opens
    import csv, time
    r = TileResolver()
    t0=time.time()
    rows=list(csv.DictReader(open("/Users/suzuki/github/projects_cosmos/csvfiles_sn/lookup_master56.csv")))
    bands=[("hst","F814W"),("jwst","F115W"),("jwst","F150W"),("jwst","F277W"),("jwst","F444W")]
    opens=set(); miss=0
    for row in rows:
        ra,dec=float(row["sn_ra"]),float(row["sn_dec"])
        for sv,bd in bands:
            tile,path=r.resolve(sv,bd,ra,dec)
            if path: opens.add(path)
            else: miss+=1
    print(f"56 SNe × 5 bands: {len(opens)} unique FITS files to open, {miss} off-footprint")
    print(f"resolve time: {1000*(time.time()-t0):.1f} ms (pure header-geometry, no pixel I/O)")
