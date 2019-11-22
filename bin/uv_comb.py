#! /usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Jun LIU'
__copyright__ = 'Copyright (c) 2019 Jun Liu <jliu@mpifr-bonn.mpg.de>'
__license__ = 'GPL v3'
__version__ = '1.5'

###############################################################################
# UPDATE NOV. 22, 2019:
# NEW MERGING MODE "IF_FLATTEN" IS ADDED. USING '-M IF_FLATTEN' ENABLE ONE TO
# MERGE THE MULTIPLE IF UVFITS INTO A SINGLE ONE. THE PURPOSE OF THIS NEW
# IS TO PROVIDE DATA FOR STATISTICS ON CLOSURE QUANTITIES IN EHTIM.
# FOR MORE DETAILS, SEE ./UV_COMB.PY -H
###############################################################################
# UPDATE NOV. 11, 2019:
# NEW COMMAND LINE ARGUMENT 'MODE' IS ADDED, USING '-M CHECK_SCAN' ENABLE
# ONE TO CHECK THE SCANS BETWEEN DIFFERENT SOURCES.
# FOR MORE DETAILS, SEE ./UV_COMB.PY -H
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
      # DUMP THE AN DATA TO A DICTIONARY, SET ORBPARM, POLCALA, POLCALB TO
      # EMPTY, THIS SHOULD BE FINE WITH MOST CASES.
      data = f['AIPS AN'].data
      nant = len(data['ANNAME'])
      _data = {}
      for key in data.names:
        if key in ['ORBPARM', 'POLCALA', 'POLCALB']:
          _data[key] = np.empty((nant, ))
        else:
          _data[key] = data[key]
      an_data.append(_data)
      del _data, data

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
      an_formats = ['%dA' %nant, '3D', '0E', '1J', '1E', '1J', \
                    '1A', '1E', '0E', '1A', '1E', '0E']
      cols = [pyfits.Column(name=an_keys[k], format=an_formats[k], \
              array=an_data_new[k]) for k in range(len(an_keys))]
      antab = pyfits.BinTableHDU.from_columns(cols, name='AIPS AN')
      ah = antab.header
      for ahk in ah_old:
        if ahk not in ah:
          ah[ahk] = ah_old[ahk]

      del f['AIPS AN']
      f.append(antab)
      print('  done !!!')

      f.writeto('tmp_AN_update_%d.uvfits' %i,
                output_verify = 'silentfix',
                overwrite = True)
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
  for band in uvf_bands:
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

def borrow_cards(hdr, key):
  return hdr[key] if key in hdr else key

def uvf_if_flatten(uvf, outp='merged.uvfits'):

  # THIS IS TO ADD NEW MERGING MODE: IF_FLATTEN
  # ASSUMING THE SINGLE UVF IS WELL BEHAVED
  uvf_hdul = pyfits.open(uvf)
  f0 = uvf_hdul[0].header['CRVAL4']
  freqs = uvf_hdul['AIPS FQ'].data['IF FREQ'].flatten() + f0
  bw = freqs[-1] - freqs[0]
  Nif = len(freqs)
  freq0 = freqs.mean()

  data = uvf_hdul[0].data
  exp_keys = ['UU', 'VV', 'WW',  'BASELINE', 'DATE', 'INTTIM']
  keys = data.parnames
  idx, iek = index_uvf_keys(keys, exp_keys)
  comb_data = [[] for i in range(7)]
  comb_data.append(np.empty((0, 1, 1, 1, 1, 4, 3)))
  for i in range(Nif):
    for j in range(7):
    # FILLINF UU, VV and WW, BASELINE, DATE, DATE, INTTIM AND DATA
      if j == 5: # ADD A ONE-SEC OFFSET TO TIMESTAMPS
        comb_data[j].append(data.par(idx[j]) + i*1.0/86400)
      else:
        comb_data[j].append(data.par(idx[j]))
    comb_data[7] = np.append(comb_data[7], data.data[:,:,:,[i],:,:,:], axis=0)

  # DATA IS READY
  for i in range(len(iek)-1):
    comb_data[i] = np.array(comb_data[i], dtype=np.float64).flatten()
  mjd = comb_data[4] + comb_data[5]
  idx_tsort = mjd.argsort()
  for i in range(len(iek)):
    comb_data[i] = comb_data[i][idx_tsort]

  print(comb_data[-1].shape)
  gdata = pyfits.GroupData(
      input = comb_data[-1],
      parnames = iek[:-1],
      pardata = comb_data[:-1],
      bscale = 1.0, bzero = 0.0, bitpix = -32)
  ghdu = pyfits.GroupsHDU(gdata)
  ghdu.header = uvf_hdul[0].header
  ghdu.header['CRVAL4'] = freq0
  ghdu.header['CDELT4'] = bw

  antab = uvf_hdul['AIPS AN']
  fq_keys = ['FRQSEL', 'IF FREQ', 'CH WIDTH', 'TOTAL BANDWIDTH', 'SIDEBAND']
  fq_formats = ['1J', '1D', '1E', '1E', '1J']
  fq_data = [np.array([1]), [0], [bw], [bw], [1]]

  cols = [pyfits.Column(name=fq_keys[k], format=fq_formats[k], \
          array=fq_data[k]) for k in range(len(fq_keys))]
  fqtab = pyfits.BinTableHDU.from_columns(cols, name='AIPS FQ')
  fh = fqtab.header
  fh['NO_IF'] = 1

  comb_hdu = pyfits.HDUList([ghdu, antab, fqtab])
  comb_hdu.writeto(outp,
      output_verify = 'silentfix',
      overwrite = True)

def uvf_combine(uvf_list, outp='merged.uvfits', mode='normal'):

  uvf_list = check_uvf_antab(uvf_list)
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
  all_rjds.sort() # SORTED UNION OF TIMESTAMPS
  Ntime = len(all_rjds)
  print('T-B stamps done!')

  # GET THE BASELINES AT ALL TIMESTAMPS
  print('Segmenting datasets ...')
  all_bsls = [0 for i in range(Ntime)]
  for (i, rjd) in enumerate(all_rjds):
    # ST_BSLS: BASELINES AT A SINGLE TIMESTAMP (FOR EACH UVFIT, LIST OF LIST).
    # THIS LIST WILL BE USED TO GET THE UVW AND INTTIM
    st_bsls = []
    # T_BSLS: BASELINES AT A SINGLE TIMESTAMP (FOR ALL UVFITS)
    t_bsls = []
    st_intts = []
    st_uus, st_vvs, st_wws = [], [], []
    st_data = []

    for j in range(Nuvf):
      data = uvf_hdul[j][0].data
      keys = data.parnames
      idx, iek = index_uvf_keys(keys, exp_keys)
      jds = np.float64(data.par(idx[4])) + np.float64(data.par(idx[5]))
      flt = jds == rjd

      bsl = list(data.par(idx[3])[flt])
      t_bsls += bsl
      st_bsls.append(bsl)
      # THE ORDER OF BSL DOES NOT EFFECT THE ORDER OF U, V, W AND INTTS
      st_uus.append(list(data.par(idx[0])[flt]))
      st_vvs.append(list(data.par(idx[1])[flt]))
      st_wws.append(list(data.par(idx[2])[flt]))
      st_intts.append(list(data.par(idx[6])[flt]))
      st_data.append(data.par(idx[7])[flt])
#      print data.par(idx[7])[flt].shape
      if mode == 'check_scan':
        st_data[-1][:,:,:,:,:,:,0] = j + 1.0
        st_data[-1][:,:,:,:,:,:,1] = 0.0
        st_data[-1][:,:,:,:,:,:,2] = 2.0

    t_bsls = list(set(t_bsls))
    t_bsls.sort() # SORTED UNION BASELINES AT A GIVEN TIME

    # PREPARE THE UVW AND INTTIM
    for t_bsl in t_bsls:
      filled = False
      _arr_st_data = np.zeros((1, 1, Nif, 1, 4, 3), dtype=np.float64)
      # ITERATE ON UVFITS
      for (j, st_bsl) in enumerate(st_bsls):
        if t_bsl in st_bsl:
          k = st_bsl.index(t_bsl)
          if not filled:
            comb_data[0].append(st_uus[j][k])
            comb_data[1].append(st_vvs[j][k])
            comb_data[2].append(st_wws[j][k])
            comb_data[6].append(st_intts[j][k])
            filled = True
          # NOW IT IS TIME TO WORK ON IFS !!!
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

  brr_keys = ['TELESCOP', 'INSTRUME', 'OBSERVER', 'OBJECT', 'BUNIT']
  for bk in brr_keys:
    cards.append((bk, borrow_cards(hdr0, bk)))

  eq_keys = ['EQUINOX', 'EPOCH']
  eq_keys = [ek for ek in eq_keys if ek in hdr0.keys()]
  if len(eq_keys) >= 1:
    ek = eq_keys[0]
    try: cards.append(('EPOCH', float(hdr0[ek])))
    except: cards.append(('EPOCH', 2000.0))
  else: cards.append(('EPOCH', 2000.0))

  cards.append(('BSCALE', 1.0))
  cards.append(('BSZERO', 0.0))
  cards.append(('VELREF', 3))
  cards.append(('ALTRVAL', 0.0))
  cards.append(('ALTRPIX', 1.0))
  cards.append(('OBSRA', hdr0['CRVAL6'], ''))
  cards.append(('OBSDEC', hdr0['CRVAL7'], ''))
  cards.append(('RESTFREQ', 0.0))
  for card in cards:
      ghdu.header.append(card)

  # THE VISIBILITIES SHOULD BE ALREADY TB SORTED.
  # ADD IT TO HISTORY SO THAT AIPS CAN RECOGNIZE IT.
  ghdu.header['HISTORY'] = "AIPS   SORT ORDER = 'TB'"

  # REFORMAT AN AND FQ TABLE
  # IN THE CASE OF SINGLE IF, THE ARRAY HAS TO BE TRANSPOSED
  print('Appending AN and FQ Tables ...')
  antab = uvf_hdul[0]['AIPS AN']

  if_freq = (r_bands.real + r_bands.imag)*0.5
  if_freq = (if_freq - if_freq[0])[np.newaxis]
  ch_width = (r_bands.imag - r_bands.real)[np.newaxis]
  tot_bw = ch_width * 1.0
  sb = np.ones_like(tot_bw)

  fq_keys = ['FRQSEL', 'IF FREQ', 'CH WIDTH', 'TOTAL BANDWIDTH', 'SIDEBAND']
  fq_formats = ['1J', '%dD' %Nif, '%dE' %Nif, '%dE' %Nif, '%dJ' %Nif]
  fq_data = [np.array([1]), if_freq, ch_width, tot_bw, sb]

  cols = [pyfits.Column(name=fq_keys[k], format=fq_formats[k], \
          array=fq_data[k]) for k in range(len(fq_keys))]
  fqtab = pyfits.BinTableHDU.from_columns(cols, name='AIPS FQ')
  fh = fqtab.header
  fh['NO_IF'] = Nif

  comb_hdu = pyfits.HDUList([ghdu, antab, fqtab])
  comb_hdu.writeto(outp,
      output_verify = 'silentfix',
      overwrite = True)

  for uvf in uvf_list:
    os.system('rm -rf %s' %uvf)
  print('Done!!! \nMerged visibility data: %s' %outp)


def main():
  parser = argparse.ArgumentParser(
      description ='  UV combine version 1.5 \n'
      '  jliu@mpifr-bonn.mpg.de \n'
      '  date: Nov. 22, 2019 \n\n'
      '  uv_comb.py is designed to merge arbitrary uvfits files, i.e., \n'
      '  uvfits files with different antenna tables, IFs and timestamps. \n'
      '  Currently multiple-source merging is not supported. \n\n'
      '  EXAMPLE: ./uv_comb.py a.uvfits b.uvfits c.uvfits -m normal -o abc.uvfits',
      formatter_class = argparse.RawDescriptionHelpFormatter)

  parser.add_argument('uvfits', help='input uvfits files to be merged',
                      metavar="FILE", nargs='*', default=None)
  parser.add_argument('-m', '--mode', type=str, required=False,
      default='normal', help='merge mode: (i) nomral -- perform a normal merge, '
      'i.e., merging uvfits with different time, IFs, '
      'but with the same observing target. '
      '(ii) check_scan -- fake the amplitudes if one wants to check the '
      'scan time for different sources. '
      'In this case, one has to merge uvfits with '
      'different sources using check_scan mode.'
      '(iii) if_flatten --- flaten multiple IFs into a single one by '
      'shifting the timestamps by one second, the U, V, W and visibilities '
      'are kept untouched.')
  parser.add_argument('-o','--output', metavar='FILE', type=str, required=False,
                    default='merged.uvfits', help='output merged uvfits file')
  args = parser.parse_args()

  outp = args.output
  uvf_list = args.uvfits
  mode = args.mode
  if len(uvf_list) < 2:
    if mode == 'if_flatten':
      uvf_if_flatten(uvf_list[0], outp)
    else:
      print('Error: Please give at least two input UVFITS files for merging mode %s!' %mode)
      exit(0)
  else:
    if mode == 'if_flatten':
      tmp_out = 'tmp.'+outp
      try:
        uvf_combine(uvf_list, tmp_out, 'normal')
        uvf_if_flatten(tmp_out, outp)
      except:
        os.system('rm -rf %s' %tmp_out)
      os.system('rm -rf %s' %tmp_out)
    else:
      uvf_combine(uvf_list, outp, mode)


if __name__ == '__main__':
  main()
