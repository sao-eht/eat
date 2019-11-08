#! /usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Jun LIU'
__copyright__ = 'Copyright (c) 2019 Jun Liu <jliu@mpifr-bonn.mpg.de>'
__license__ = 'GPL v3'
__version__ = '1.3'

###############################################################################
# UPDATE NOV. 08, 2019:
# NEW CAPABILITY OF MERGING ARBITRARY UVFITS FILES (I.E. UVFITS WITH DIFFERENT
# AN TABLES, FREQUENCIES AND TIMESTAMPS).
# NOTE THAT MULTIPLE-SOURCE MERGING IS CURRENTLY NOT SUPPORTED !!!
###############################################################################

import os
import numpy as np
import astropy.io.fits as pyfits
from astropy.time import Time
import argparse

def bsl_relabel(bsl_id, ant_id_old, ant_id_new):

  bsl_id_old = np.int32(bsl_id)
  ant2_old = bsl_id_old%256
  ant1_old = (bsl_id_old - ant2_old)/256

  ant2_new = [ant_id_new[ant_id_old.index(a)] for a in ant2_old]
  ant1_new = [ant_id_new[ant_id_old.index(a)] for a in ant1_old]

  ant2_new = np.array(ant2_new, dtype=np.float32)
  ant1_new = np.array(ant1_new, dtype=np.float32)

  return 256*ant1_new + ant2_new


def check_uvf_antab(uvf_list):
# DATA SANITY CHECK, CHECK IF ALL UVFITS HAVE IDENTICAL AN TABLE
# IF NOT, UPDATE THE AN TABLES AND BASELINE IDS
  print('Checking antenna tables ...')
  an_data = []
  for uvf in uvf_list:
    with pyfits.open(uvf) as f:
      # DUMP THE AN DATA TO A DICTIONARY, IN CASE OF EMPTY FIELDS
      # (E.G. ORBPARM, POLCALA, POLCALB)
      data = f['AIPS AN'].data
      _data = {}
      for key in data.names:
        _data[key] = data[key]
      an_data.append(_data)
      del _data
      if not len(an_data[-1]['ORBPARM']):
        nant = len(an_data[-1]['ANNAME'])
        # FILL THE EMPTY FIELDS WITH ZEROS
        an_data[-1]['ORBPARM'] = np.zeros(nant)
        an_data[-1]['POLCALA'] = np.zeros(nant)
        an_data[-1]['POLCALB'] = np.zeros(nant)
      del data

  an_keys = ['ANNAME', 'STABXYZ', 'ORBPARM', 'NOSTA', 'STAXOF', 'MNTSTA', \
             'POLTYA', 'POLAA', 'POLCALA', 'POLTYB', 'POLAB', 'POLCALB']
  an_data_new = [an_data[0][key] for key in an_keys]
  # STRUCTURE IS IN [COLUMN, ANTENNA]

  for i in range(1, len(an_data)):
    # I -- CURRENT UVFITS
    ad = [an_data[i][key] for key in an_keys]
    for (j, xyz) in enumerate(ad[1]):
      # J -- CURRENT ANTENNA
      irow = np.where(an_data_new[1]==xyz)[0]
      if len(irow) == 0:
        # ADD NEW ANTENNA TO THE TABLE
        print('  New antenna (%s) found in %s !!!' %(ad[0][j], uvf_list[i]))
        if ad[0][j] in an_data_new[0]:
          # IF THE ANTENNA NAME ALREADY EXISTS, RENAME IT
          ad[0][j] = '%s%d' %(ad[0][j], j)
        for k in range(len(ad)):
          # K -- CURRENT COLUMN OF AN TABLE
          print k, j
          an_data_new[k] = np.append(an_data_new[k], [ad[k][j]], axis=0)
  print('AN table checking done !!!')

  # AN-DATA GATHERING READY, NOW RE-LABELING THE BASELINES FOR EACH UVFTIS
  print('Re-labeling baseline num. and AN tables ...')
  idx_sorted =  np.argsort(an_data_new[0])
  for i in range(len(an_data)):
    an_data_new[i] = an_data_new[i][idx_sorted]
  an_data_new[3] = np.arange(len(an_data_new[0]), dtype=np.int32) + 1

  uvf_list_new = []
  for (i, uvf) in enumerate(uvf_list):
    with pyfits.open(uvf) as f:
      print('  Re-labeling for %s ...' %uvf)
      ad = f['AIPS AN'].data
      ant_id_new = []
      for (j, xyz) in enumerate(ad['STABXYZ']):
        idxs = np.asarray(list(set(np.where(an_data_new[1] == xyz)[0])))
        if len(idxs) > 1:
          # SAME POSITION, DIFFERENT NAMES
          for idx in idxs:
            if ad['ANNAME'][j] in an_data_new[0][idx]:
              break
          idxs = np.array([idx])
        ant_id_new.append(int(an_data_new[3][idxs][0]))
      ant_id_old = list(np.int32(ad['NOSTA']))

      # UPDATE THE BASELINE ENTRY
      f[0].data['BASELINE'] = bsl_relabel(f[0].data['BASELINE'],
                                          ant_id_old, ant_id_new)

      # REWRITE THE AN TABLE
      print('  Updating AN table for %s ...' %uvf)
      ah_old = f['AIPS AN'].header
      nant = len(an_data_new[0])
      an_formats = ['%dA' %nant, '3D', '1E', '1J', '1J', '1E', \
                    '1A', '1E', '3E', '1A', '1E', '3E']
      cols = [pyfits.Column(name=an_keys[k], format=an_formats[k], \
              array=an_data_new[k]) for k in range(len(an_keys))]
      antab = pyfits.BinTableHDU.from_columns(cols)
      ah = antab.header
      for ahk in ah_old:
        if ahk not in ah:
          ah[ahk] = ah_old[ahk]

      del f['AIPS AN']
      f.append(antab)
      print('  done !!!')

      f.writeto('tmp_AN_update_%d.uvfits' %i, overwrite=True)
      uvf_list_new.append('tmp_AN_update_%d.uvfits' %i)
  print('Baseline re-labeling and AN table updating done !!!')
  return uvf_list_new


def check_uvf_ifs(uvf_list):
  # RE-ORDER THE UVF_LIST
  # PICK THE LOWEST FREQ IF AS THE REFERENCE IF
  uvf_bands = []
  for (i, uvf) in enumerate(uvf_list):
    with pyfits.open(uvf) as f:
      f0 = f[0].header['CRVAL4']
      fqdata = f['AIPS FQ'].data
      fgap = fqdata['IF FREQ'].flatten()
      bw = fqdata['CH WIDTH'].flatten()
      flo = f0 + fgap - bw*0.5
      fhi = f0 + fgap + bw*0.5
      band = flo + 1j*fhi
      uvf_bands.append(band)

  r_bands = [rb for ub in uvf_bands for rb in ub]
  r_bands = list(set(r_bands))
  r_bands.sort()
  r_bands = np.array(r_bands)

  # UNION OF ALL FREQS AND IFS, GET THE FREQ RANGE FOR EACH IF
  # FIND THE IF STAMPS FOR EACH UVFITS
  if_pos = []
  for (i, band) in enumerate(uvf_bands):
    iband = np.where(r_bands==band)[0]
    if_pos.append(iband)

  return r_bands, if_pos


def index_uvf_keys(keys, exp_keys):
  idx = []
  iek = []
  for ek in exp_keys:
    ekk = [k for k in keys if ek in k]
    if len(ekk) > 0:
      idx.append(keys.index(ekk[0]))
      iek.append(ekk[0])
      if ekk[0] == 'DATE':
        idx.append(keys.index(ekk[0])+1)
        iek.append(ekk[0])
    else:
      print('Keyword does not match!')
      return None
  idx.append(len(keys))
  iek.append('DATA')
  return idx, iek


def uvf_if_combine(uvf_list, outp='merged.uvfits'):

  uvf_list = check_uvf_antab(uvf_list)
#  uvf_list = ['tmp_AN_update_%d.uvfits' %i for i in range(len(uvf_list))]
  r_bands, if_pos = check_uvf_ifs(uvf_list)
  Nif = len(r_bands)

  Nuvf = len(uvf_list)
  # UU, VV and WW, BASELINE, DATE, DATE, INTTIM AND DATA
  exp_keys = ['UU', 'VV', 'WW',  'BASELINE', 'DATE', 'INTTIM']
  comb_data = [[] for i in range(8)]
  uvf_hdul = [0 for i in range(Nuvf)]
  for (i, uvf) in enumerate(uvf_list):
    uvf_hdul[i] = pyfits.open(uvf)

  # GET THE TIMESTAMPS
  print('Computing time and baseline stamps ...')
  all_rjds = []
  for i in range(Nuvf):
    data = uvf_hdul[i][0].data
    keys = data.parnames
    idx, iek = index_uvf_keys(keys, exp_keys)
    jds = np.float64(data.par(idx[4])) + np.float64(data.par(idx[5]))
    all_rjds += list(set(jds))

  all_rjds = list(set(all_rjds))
  all_rjds.sort()
  Ntime = len(all_rjds)
  print('T-B stamps done!')

  # GET THE BASELINES AT ALL TIMESTAMPS
  print('Segmenting datasets ...')
  all_bsls = [0 for i in range(Ntime)]
  for (i, rjd) in enumerate(all_rjds):
    # BASELINES AT A SINGLE TIMESTAMP (FOR EACH UVFIT)
    # THIS LIST WILL BE USED TO GET THE UVW AND INTTIM
    st_bsls = []
    st_intts = []
    st_uus, st_vvs, st_wws = [], [], []
    st_data = []
    # BASELINES AT A SINGLE TIMESTAMP (FOR ALL UVFITS)
    t_bsls = []
    for j in range(Nuvf):
      data = uvf_hdul[j][0].data
      keys = data.parnames
      idx, iek = index_uvf_keys(keys, exp_keys)
      jds = np.float64(data.par(idx[4])) + np.float64(data.par(idx[5]))
      flt = jds == rjd

      bsl = list(data.par(idx[3])[flt])
      t_bsls += bsl
      st_bsls.append(bsl)
      st_uus.append(list(data.par(idx[0])[flt]))
      st_vvs.append(list(data.par(idx[1])[flt]))
      st_wws.append(list(data.par(idx[2])[flt]))
      st_intts.append(list(data.par(idx[6])[flt]))
      st_data.append(list(data.par(idx[7])[flt]))

    t_bsls = list(set(t_bsls))
    t_bsls.sort() # UNION BASELINES AT A GIVEN TIME

    # PREPARE THE UVW AND INTTIM
    for t_bsl in t_bsls:
      # ITERATE ON UVFITS
      for (j, st_bsl) in enumerate(st_bsls):
        if t_bsl in st_bsl:
          k = st_bsl.index(t_bsl)
          comb_data[0].append(st_uus[j][k])
          comb_data[1].append(st_vvs[j][k])
          comb_data[2].append(st_wws[j][k])
          comb_data[6].append(st_intts[j][k])
          break

      # NOW IT IS TIME TO WORK ON IFS !!!
      # THIS IS GOING TO BE COMPLICATE BECAUSE WE ARE DEALING WITH TIME AND IF
      # AT THE SAME TIME
#      _arr_st_data = np.empty((0, 1, 0, 1, 4, 3))
#      for (j, st_bsl) in enumerate(st_bsls):
#        if t_bsl in st_bsl:
#          k = st_bsl.index(t_bsl)
#
      _arr_st_data = np.zeros((1, 1, Nif, 1, 4, 3))
      # ITERATION ON UVFITS
      for (j, st_bsl) in enumerate(st_bsls):
        if t_bsl in st_bsl:
          k = st_bsl.index(t_bsl)
          for (l, ip) in enumerate(if_pos[j]):
            _arr_st_data[:,:,ip,:,:,:] = st_data[j][k][:,:,l,:,:,:]
      comb_data[7].append(_arr_st_data)
      del _arr_st_data

    all_bsls[i] = t_bsls
    del data, jds, bsl, flt, st_bsls, st_intts, t_bsls,
    del t_bsl, st_bsl, st_uus, st_vvs, st_wws
  print('Segmenting done!')

  print('Merging datasets ...')
  Nvis = len(comb_data[0])
  # WHEN TIMESTAMPS, BASELINES AND INTTIM ARE READY
  for (i, rjd) in enumerate(all_rjds):
    t_nbsl = len(all_bsls[i])
    comb_data[3] += all_bsls[i]
    comb_data[4] += [int(rjd)]*t_nbsl
    comb_data[5] += [rjd-int(rjd)]*t_nbsl

  # DATA IS READY
  for i in range(len(iek)):
    comb_data[i] = np.array(comb_data[i], dtype=np.float64)

  gdata = pyfits.GroupData(
      input = comb_data[-1],
      parnames = iek[:-1],
      pardata = comb_data[:-1],
      bscale = 1.0, bzero = 0.0, bitpix = -32)
  ghdu = pyfits.GroupsHDU(gdata)

  # FORMATTING THE FITS HEADER. FOR SOME CASES THE KEY-VALUES FROM ORIGINAL
  # HEADER ARE BORROWED. NORMALLY THIS SHOULD BE FINE, IF NOT, PLEASE JUST LOAD
  # THE ORIGINAL DATASET INTO DIFMAP AND THEN SAVE IT OUT. SO THAT THE HEADERS
  # ARE AUTOMATICALLY FORMATTED.
  print('Formatting FITS header ...')
  hdr0 = uvf_hdul[0][0].header
  cards = []
  # Complex
  cards.append(('CTYPE2', 'COMPLEX', ''))
  cards.append(('CRPIX2', 1.0, ''))
  cards.append(('CRVAL2', 1.0, ''))
  cards.append(('CDELT2', 1.0, ''))
  cards.append(('CROTA2', 0.0, ''))
  # Stokes
  cards.append(('CTYPE3','STOKES', ''))
  cards.append(('CRPIX3', 1.0, ''))
  cards.append(('CRVAL3', hdr0['CRVAL3'], ''))
  cards.append(('CDELT3', hdr0['CDELT3'], ''))
  cards.append(('CROTA3', 0.0, ''))
  # FREQ
  if_freq = (r_bands.real + r_bands.imag)*0.5
  ch_width = (r_bands.imag - r_bands.real)
  cards.append(('CTYPE4', 'FREQ', ''))
  cards.append(('CRPIX4', 1.0, ''))
  cards.append(('CRVAL4', if_freq[0], ''))
  cards.append(('CDELT4', ch_width[0], ''))
  cards.append(('CROTA4', 0.0, ''))
  del if_freq, ch_width
  # IF
  cards.append(('CTYPE5','IF',''))
  cards.append(('CRPIX5',1.0,''))
  cards.append(('CRVAL5',1.0,''))
  cards.append(('CDELT5',1.0,''))
  cards.append(('CROTA5',0.0,''))
  # RA & Dec
  cards.append(('CTYPE6','RA',''))
  cards.append(('CRPIX6',1.0,''))
  cards.append(('CRVAL6', hdr0['CRVAL6'], ''))
  cards.append(('CDELT6',1.0,''))
  cards.append(('CROTA6',0.0,''))
  cards.append(('CTYPE7','DEC',''))
  cards.append(('CRPIX7',1.0,''))
  cards.append(('CRVAL7', hdr0['CRVAL7'], ''))
  cards.append(('CDELT7',1.0,''))
  cards.append(('CROTA7',0.0,''))
  for card in cards:
    ghdu.header.append(card)

  # PTYPE HEADER
  cards = []
  pcount = 7 # DIFMAP DEFAULTS, SHOULD NOT BE A PROBLEM FOR MOST CASES
  for i in range(1, pcount+1):
    pk = 'PTYPE%d' %i
    cards.append((pk, hdr0[pk]))
    cards.append(('PSCAL%d' %i, 1.0))
    cards.append(('PZERO%d' %i, 0.0))
  for card in cards:
    ghdu.header.append(card)

  # Other Header
  utc = Time(all_rjds[0], scale='utc', format='jd')
  cards = []
  cards.append(('DATE-OBS', utc.isot[:10]))
  cards.append(('TELESCOP', hdr0['TELESCOP']))
  cards.append(('INSTRUME', hdr0['INSTRUME']))
  cards.append(('OBSERVER', hdr0['OBSERVER']))
  cards.append(('OBJECT', hdr0['OBJECT']))
  if 'EQUINOX' in hdr0:
    cards.append(('EPOCH', int(hdr0['EQUINOX'][1:])))
  elif 'EPOCH' in hdr0:
    cards.append(('EPOCH', hdr0['EPOCH']))
  else:
    cards.append(('EPOCH', '2000.0'))
  cards.append(('BSCALE', 1.0))
  cards.append(('BSZERO', 0.0))
  cards.append(('BUNIT', hdr0['BUNIT']))
  cards.append(('VELREF', 3))
  cards.append(('ALTRVAL', 0.0))
  cards.append(('ALTRPIX', 1.0))
  cards.append(('OBSRA', hdr0['CRVAL6'], ''))
  cards.append(('OBSDEC', hdr0['CRVAL7'], ''))
  cards.append(('RESTFREQ', 0))
  for card in cards:
      ghdu.header.append(card)

  # REFORMAT AN AND FQ TABLE
  # IN THE CASE OF SINGLE IF, THE ARRAY HAS TO BE TRANSPOSED
  print('Appending AN and FQ Tables ...')
  antab = uvf_hdul[0]['AIPS AN']

  if_freq = (r_bands.real + r_bands.imag)*0.5
  if_freq = if_freq[1:] - if_freq[:-1]
  if_freq = np.insert(if_freq, 0, 0)[np.newaxis]
  ch_width = (r_bands.imag - r_bands.real)[np.newaxis]
  tot_bw = ch_width * 1.0
  sb = np.ones_like(tot_bw)

  fq_keys = ['FRQSEL', 'IF FREQ', 'CH WIDTH', 'TOTAL BANDWIDTH', 'SIDEBAND']
  fq_formats = ['1J', '%dD' %Nif, '%dE' %Nif, '%dE' %Nif, '%dJ' %Nif]
  fq_data = [np.array([1]), if_freq, ch_width, tot_bw, sb]

  cols = [pyfits.Column(name=fq_keys[k], format=fq_formats[k], \
          array=fq_data[k]) for k in range(len(fq_keys))]
  fqtab = pyfits.BinTableHDU.from_columns(cols)
  fh = fqtab.header
  fh['EXTNAME'] = 'AIPS FQ'
  fh['NO_IF'] = Nif

  comb_hdu = pyfits.HDUList([ghdu, antab, fqtab])
  comb_hdu.writeto(outp, overwrite=True)

  for uvf in uvf_list:
    os.system('rm -rf %s' %uvf)
  print('Done!!! \nMerged visibility data: %s' %outp)


def main():
#  parser = argparse.ArgumentParser(description=__doc__,
  parser = argparse.ArgumentParser(
      description ='  UV combine version 1.3 \n'
      '  jliu@mpifr-bonn.mpg.de \n'
      '  date: Nov. 08, 2019 \n\n'
      '  uv_comb.py is designed to merge arbitrary uvfits files, i.e., \n'
      '  uvfits files with different antenna tables, IFs and timestamps. \n'
      '  Currently multiple-source merging is not supported. \n\n'
      '  EXAMPLE: ./uv_comb.py a.uvfits b.uvfits c.uvfits -o abc.uvfits',
      formatter_class = argparse.RawDescriptionHelpFormatter)

  parser.add_argument('uvfits', help='input uvfits files to be merged',
                      metavar="FILE", nargs='*', default=None)
  parser.add_argument('-o','--output', metavar='FILE', type=str, required=False,
                    default='merged.uvfits', help='output merged uvfits file')
  args = parser.parse_args()

  outp = args.output
  uvf_list = args.uvfits
  if len(uvf_list) < 2:
    print('Error: Please give at least two input UVFITS files!')
    exit(0)

  uvf_if_combine(uvf_list, outp)


if __name__ == '__main__':
  main()
