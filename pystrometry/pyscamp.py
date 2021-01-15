"""
Functions to read and interpret results obtained with astromatic's scamp

"""
import copy
import os
from astropy.io.votable import parse
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord

from astropy.table import Table, Column
from astropy.table import vstack as tablevstack
import pylab as pl
import numpy as np

import pickle

try:
    import pystortion
except ImportError:
    print('WARNING: pystortion package not available!')

try:
    import aplpy
except ImportError:
    print('WARNING: aplpy package not available!')



class ScampXml(object):
    """
    Class for handling scamp.xml output file
    can be used as crossmatch table to identify stars across frames
    """
    def __init__(self, scamp_dir):
        self.table, self.source_file = getScampXml(scamp_dir)
        self.source_dir = scamp_dir
        # determine epoch number

        obs_date_mjd = Time(self.table['Observation_Date'].data, format='decimalyear').mjd
        diff_idx = np.where(np.diff(obs_date_mjd) > 0.1)[0]
        ob_numbers = np.ones(len(self.table)).astype(int)
        for j in diff_idx:
            ob_numbers[j + 1:] += 1
        self.table.add_column(Column(data=ob_numbers, name='EPOCH'))
        self.table.add_column(Column(data=obs_date_mjd, name='MJD'))

    # def plot(self, plot_dir):
    #     helpers.make_dir(plot_dir)
    #     helpers.plot_columns_simple(self.table, plot_dir)

    def info(self):
        self.target = str(np.unique(np.array(self.table['Image_Ident'])))
        self.n_frames = len(self.table)
        self.timespan_day = np.ptp(self.table['MJD'])
        self.n_epoch = len(np.unique(self.table['EPOCH']))
        self.average_pixel_scale = np.mean(self.table['Pixel_Scale'].data) * self.table['Pixel_Scale'].unit
        print('{}: target is {}, {} frames, {} epochs, {:2.1f} days timespan'.format(self.source_file, self.target, self.n_frames, self.n_epoch, self.timespan_day))
        # print('{}: Average pixel scale {:3.3f}'.format(self.source_file, self.average_pixel_scale.to(u.milliarcsecond)))

    def get_catalog_index(self, catalog_number=None, catalog_name=None):
        """
        return the table row index of a catalog
        :param catalog_number:
        :param catalog_name:
        :return:
        """
        if catalog_number is not None:
            catalog_index = self.table['Catalog_Number'].tolist().index(catalog_number)
        elif catalog_name is not None:
            catalog_index = self.table['Catalog_Name'].tolist().index((catalog_name.split('.')[0]+'.cat').encode())
        else:
            raise RuntimeError('Specify exactly one argument')
        return catalog_index

    def get_fits_filename(self, catalog_number):
        """
        return the name of the fits file used to create the catalog
        :param catalog_number:
        :return:
        """
        catalog_index = self.get_catalog_index(catalog_number=catalog_number)
        file_name = (self.table['Catalog_Name'][catalog_index]).decode().replace('.cat', '.fits')
        return file_name


class ScampFullCatalog(object):
    """
    Class for handling scamp full.cat output file
    http://scamp.readthedocs.io/en/latest/Output.html#full-catalogues
    """
    def __init__(self, scamp_dir=None, table=None):
        if table is not None:
            self.orig_table = table
            self.source_file = 'table_input'
            self.source_dir = None

        elif scamp_dir is not None:
            self.orig_table, self.source_file = read_scamp_cat(scamp_dir, 'full')
            self.source_dir = scamp_dir

        self.table = self.orig_table.copy()
        if 0 in self.orig_table['CATALOG_NUMBER'].data:
            # remove reference catalog
            self.table.remove_rows( np.where(self.orig_table['CATALOG_NUMBER']==0)[0] )

        self.unique_source_numbers = np.unique(self.table['SOURCE_NUMBER'])
        self.n_science_catalogs = len(np.unique(self.table['CATALOG_NUMBER']))

        # add column to identify fake stars used to stabilise astrometric reduction
        self.table['artificial'] = np.zeros(len(self.table)).astype(np.int)

    def info(self):
        """ Print basic info
        """
        print('{}: {} entries (including reference catalog)'.format(self.source_file, len(self.orig_table)))
        print('{}: {} entries (excluding reference catalog)'.format(self.source_file, len(self.table)))
        print('{}: {} catalogs (catalog 0 is the scamp reference catalog e.g. 2MASS)'.format(self.source_file, len(np.unique(self.orig_table['CATALOG_NUMBER']))))
        print('{}: {} science catalogs '.format(self.source_file, self.n_science_catalogs))
        print('{}: {} unique SOURCE_NUMBERs (excluding reference catalog)'.format(self.source_file, len(self.unique_source_numbers)))


    def complement_missing_catalogs(self, scamp_merged, threshold, verbose=False, target_number=None):
        """Fill in 'zero rows' for stars that have not been measured in all frames.

        if target_number is specified, no artifical measurements are added for the target

        This expands the original table.

        Returns
        -------

        """

        # identify sources that have not been measured in all frames/catalogs
        source_number_to_complement = np.unique(scamp_merged.table['SOURCE_NUMBER'][scamp_merged.table['NPOS_OK'] > threshold])
        if verbose:
            print('{}/{} sources have >{} scamp detections'.format(len(source_number_to_complement),
                                                               len(
                                                                   self.unique_source_numbers),
                                                               threshold))


        complemented_table = copy.deepcopy(self.table)

        all_catalog_numbers = np.unique(self.table['CATALOG_NUMBER'])
        if verbose:
            print('There are {} catalogs'.format(len(all_catalog_numbers)))

        # loop through sources
        for source_number in source_number_to_complement:
            if (target_number is not None) and (source_number==target_number):
                continue
            found_index = complemented_table['SOURCE_NUMBER'] == source_number
            found_in_catalog_numbers = complemented_table['CATALOG_NUMBER'][found_index]
            if verbose:
                print('Source {} found in {} catalogs'.format(source_number, len(found_in_catalog_numbers)))
            catalog_numbers_to_complement = np.setdiff1d(all_catalog_numbers,
                                                         found_in_catalog_numbers)
            if verbose:
                print('Have to complement in {} catalogs'.format(len(catalog_numbers_to_complement)))
            temp = copy.deepcopy(complemented_table[0:len(catalog_numbers_to_complement)])
            for colname in complemented_table.colnames:
                if colname == 'SOURCE_NUMBER':
                    temp[colname] = source_number
                elif colname == 'CATALOG_NUMBER':
                    temp[colname] = catalog_numbers_to_complement
                elif colname == 'artificial':
                    temp[colname] = 1
                # set X,Y pixel positions of artifical stars to something reasonable for reduced coordinates and basic functions
                elif colname in ['X_IMAGE', 'Y_IMAGE']:
                    temp[colname] = np.median(complemented_table[colname][found_index])
                else:
                    temp[colname] = np.zeros(len(temp)).astype(temp[colname].dtype)

            complemented_table = tablevstack((complemented_table, temp))

        if verbose:
            print('Number of rows before {} after {}'.format(len(self.table), len(complemented_table)))

        self.table = complemented_table

    def select_sources(self, catalog_number=None, chip_extension=None, source_number=None, ):
        """
        Select sources in a particular frame/catalog and chip extension
        :param catalog_number:
        :param chip_extension:
        :return:
        """

        table_index = range(len(self.table))

        if catalog_number is not None:
            table_index = np.intersect1d(table_index, np.where((self.table['CATALOG_NUMBER'] == catalog_number))[0])
        if chip_extension is not None:
            table_index = np.intersect1d(table_index, np.where((self.table['EXTENSION'] == chip_extension))[0])
        if source_number is not None:
            table_index = np.intersect1d(table_index, np.where((self.table['SOURCE_NUMBER'] == source_number))[0])

        return self.table[table_index]

    def remove_source(self, source_number):
        """
        Remove a source from catalog table

        :param source_number:
        :return:
        """
        self.table.remove_rows(np.where(self.table['SOURCE_NUMBER'] == source_number)[0])
        self.unique_source_numbers = np.unique(self.table['SOURCE_NUMBER'])

    def identify_source_number(self, RA_deg, DE_deg, catalog_number, chip_extension=1, verbose=False):
        """
        Return the source number that is closest to the (RA_deg, DE_deg position)

        :param RA_deg:
        :param DE_deg:
        :param catalog_number:
        :param chip_extension:
        :return:
        """

        # sources in catalog
        T = self.select_sources(catalog_number, chip_extension)
        cat = SkyCoord(ra=np.array(T['ALPHA_J2000']) * u.degree, dec=np.array(T['DELTA_J2000']) * u.degree)

        # coordinates of target source to identify
        target_cat = SkyCoord(ra=RA_deg * u.degree, dec=DE_deg * u.degree)

        # idx = np.where(target_cat.separation(cat).to(u.arcsec) < radius)[0]
        target_index = np.argmin(target_cat.separation(cat))
        target_source_number = T['SOURCE_NUMBER'][target_index]
        if verbose:
            print('Identified SOURCE_NUMBER {0} as closest {1} to expected target location'.format(target_source_number, np.min(
            target_cat.separation(cat)).to(u.arcsec)))

        return int(target_source_number)

    def plot_sources(self, scamp_xml, catalog_number, data_dir, chip_extension=1, save_plot=False, out_dir=None, RA_deg=None,
                     DE_deg=None, radius=50.*u.arcsec, tag='selected', label=True, external_cat=None):

        T = self.select_sources(catalog_number, chip_extension)
        # cat = SkyCoord(ra=np.array(T['RA_deg']) * u.degree, dec=np.array(T['DE_deg']) * u.degree)
        cat = SkyCoord(ra=np.array(T['ALPHA_J2000']) * u.degree, dec=np.array(T['DELTA_J2000']) * u.degree)


        # name of the individual frame FITS file being plotted
        file_name = scamp_xml.get_fits_filename(catalog_number)
        image_file = os.path.join(data_dir, file_name)

        header = fits.getheader(image_file)

        # catch case of OSIRIS where WCS is in second extension header, replace primary header by header of extension 2
        if 'DATE-OBS' not in header.keys():
            header = fits.getheader(image_file, ext=chip_extension)
            for key in header.keys():
                if 'PV' in key:
                    header.remove(key)

        # to align the plotted symbols with the actual image, use the WCS of the actual frame
        w = WCS(header=header)
        # coord_array = n.array([T['x'].data, T['y'].data]).T
        coord_array = np.array([T['X_IMAGE'].data, T['Y_IMAGE'].data]).T
        coord_array_world = w.wcs_pix2world(coord_array, 0)

        T['RA_deg'] = coord_array_world[:, 0]
        T['DE_deg'] = coord_array_world[:, 1]

        fig = pl.figure(figsize=(7, 7), facecolor='w', edgecolor='k'); pl.clf()
        try:
            gc = aplpy.FITSFigure(image_file, extension=chip_extension, figure=fig,
                                  North=True)  # ,convention='wells')#, dimensions=[1 ,0]);
        except:  # InvalidTransformError
            # for files reduced by Herve that contain the PV1_5 keywords
            data = fits.getdata(image_file, ext=chip_extension)
            hdu = fits.PrimaryHDU(data)
            hdu.header = header
            gc = aplpy.FITSFigure(hdu, extension=chip_extension, figure=fig,
                                  North=True)  # ,convention='wells')#, dimensions=[1 ,0]);

        # gc.show_grayscale(invert=False)  # , stretch='log', vmid=-1)#,vmid=-1)#, aspect = 1., pmax =90 )
        gc.show_grayscale(invert=False, stretch='log')  # , stretch='log', vmid=-1)#,vmid=-1)#, aspect = 1., pmax =90 )


        if RA_deg is not None:
            # target is defined
            gc.recenter(RA_deg, DE_deg, radius=radius.to(u.deg).value)
            gc.show_markers(RA_deg, DE_deg, marker='s', edgecolor='y', linewidth=3, s=30)
            target_cat = SkyCoord(ra=RA_deg * u.degree, dec=DE_deg * u.degree)
            idx = np.where(target_cat.separation(cat).to(u.arcsec) < radius)[0]
            target_source_number = self.identify_source_number(RA_deg, DE_deg, catalog_number, chip_extension, verbose=False)
        else:
            idx = range(len(cat))



        gc.show_markers(T['RA_deg'][idx] - 0 / 3600., T['DE_deg'][idx] - 0 / 3600., marker='o', edgecolor='g')
        if label:
            for j in idx:
                gc.add_label(T['RA_deg'][j] - 0 / 3600., T['DE_deg'][j] + 1 / 3600., T['SOURCE_NUMBER'][j])

        if external_cat is not None:
            gc.show_markers(external_cat.ra, external_cat.dec, marker='s', edgecolor='b', lw=1)


        pl.show()
        if save_plot:
            save_file = os.path.join(out_dir,file_name.replace('.fits', '_{}_sources.pdf'.format(tag)))
            gc.save(save_file, dpi=300)

    def correct_distortion(self, cal_pickle, calibration_catalog_number=None):
        """
        Corrects for distortion in every frame
        creates columns X_Idl and Y_Idl

        if calibration_catalog_number is specified, the full table is corrected for the distortion
        specified in the calibration_catalog. Otherwise every catalog is corrected for it's individual distortion

        :param cal_pickle:
        :return:
        """

        cals = pickle.load(open(cal_pickle, "rb"))  # cals is a dictionary

        self.table['X_Idl'] = np.zeros(len(self.table))
        self.table['Y_Idl'] = np.zeros(len(self.table))

        # this applies the transformation from the star catalog onto the Gaia frame
        evaluation_frame_number = 1

        if calibration_catalog_number is not None:
            cal_obs = cals[calibration_catalog_number]
            x_in = self.table['X_IMAGE']
            y_in = self.table['Y_IMAGE']
            self.table['X_Idl'], self.table['Y_Idl'] = \
                cal_obs.lazAC.apply_polynomial_transformation(evaluation_frame_number, x_in, y_in)
        else:
            catalog_numbers = self.table['CATALOG_NUMBER'].data
            for catalog_number in np.unique(catalog_numbers):
                table_index = np.where(self.table['CATALOG_NUMBER'] == catalog_number)[0]
                cal_obs = cals[catalog_number]

                x_in = self.table['X_IMAGE'][table_index]
                y_in = self.table['Y_IMAGE'][table_index]
                self.table['X_Idl'][table_index], self.table['Y_Idl'][
                    table_index] = cal_obs.lazAC.apply_polynomial_transformation(evaluation_frame_number, x_in, y_in)

    def suitable_reference_sources(self, selected_source_numbers, target_number, chip_extension):
        """
        return suitable reference stars (common in frames containing target, on same chip)
            TODO:
            - review reference star selection
            - check for number of astrometric instruments


        :param scamp_merged:
        :param selected_source_numbers:
        :param target_number:
        :param chip_extension:
        :return:
        """

        target_catalog_numbers = self.table['CATALOG_NUMBER'][self.table['SOURCE_NUMBER'] == target_number]
        target_catalog_numbers.sort()
        # target_frame_numbers = self.select_sources(source_number = target_number)

        print('Target identified in {} frames out of {}'.format(len(target_catalog_numbers), self.n_science_catalogs))


        source_numbers = selected_source_numbers
        good_source_numbers = []
        for i, num in enumerate(source_numbers):
            tmp = np.where(self.table['SOURCE_NUMBER'] == num)[0]
            if ((self.table['EXTENSION'][tmp[0]] == chip_extension) & (self.table['ASTR_INSTRUM'][tmp[0]] == 1) & (
                len(target_catalog_numbers) <= len(tmp))):  # select only stars that have equal or more detections than the target and are located in selected chip and catalogue 1
                # if set(self.table['CATALOG_NUMBER'][tmp]).issubset(set(target_catalog_numbers)):
                if set(target_catalog_numbers).issubset(set(self.table['CATALOG_NUMBER'][tmp])):
                # if all([j in tf['CATALOG_NUMBER'][tmp] for j in targetCatalogNumbers]):
                    good_source_numbers.append(num)
        print ('Selected {} reference stars as potential good reference stars'.format(len(good_source_numbers)));

        if target_number not in good_source_numbers:
            raise RuntimeError("Target not identified in all selected frames");

        return np.array(good_source_numbers).astype(np.int), target_catalog_numbers

    def get_astrometry(self, scamp_xml, red_dir, good_source_numbers, target_catalog_numbers, chip_extension, verbose=True, use_Idl=True):
        """
        Extract astrometry and auxiliary information

        TODO:
        fix assignment of 'sigma_x', 'sigma_y'

        :param scamp_xml:
        :param red_dir:
        :param good_source_numbers:
        :param target_catalog_numbers:
        :param chip_extension:
        :return:
        """
        n_stars = len(good_source_numbers)
        n_frames = len(target_catalog_numbers)

        p = np.zeros((n_frames, n_stars, 12))
        aux = np.zeros((n_frames, 16))

        tmplist = scamp_xml.table['Catalog_Number'].data.tolist()

        for i, catnum in enumerate(target_catalog_numbers):

            # find index in list of catalog numbers
            tmplist_idx = tmplist.index(catnum)

            # identify underlying FITS file
            frame_name = scamp_xml.table['Catalog_Name'][tmplist_idx].decode()
            frame = os.path.join(red_dir, frame_name.replace('.cat','.fits'))
            header = fits.getheader(frame)

            if verbose:
                print('{:03d}/{:03d}: filling catalogue with stars from {}.'.format(i+1,n_frames, frame))


            if header.get('ORIGIN') == None: # OSIRIS
                header = fits.getheader(frame, ext=2);
                aux[i, :] = [scamp_xml.table['Observation_Date'][tmplist_idx], 0.0, scamp_xml.table['AirMass'][tmplist_idx],
                             scamp_xml.table['Exposure_Time'][tmplist_idx], header['TAMBIENT'], header['PRESSURE'], header['HUMIDITY'],
                             header['WINDSPEE'], header['AZIMUTH'], header['ELEVAT'],
                             SkyCoord(dec=header['LATITUDE'], ra='00:00:00.000', unit='deg').dec.deg, header['DECDEG'],
                             header['RADEG'], header['MJD-OBS'], header['AIRMASS'], 0]

            elif header['INSTRUME'].strip() in ['GMOS-S', 'GMOS-N']:
                if header['INSTRUME'].strip() == 'GMOS-S':
                    latitude_string = '-30:14:26.700 '
                elif header['INSTRUME'].strip() == 'GMOS-N':
                    latitude_string = '19:49:25.7016'


                time_difference_day = np.abs(header['MJD-OBS'] - Time(scamp_xml.table['Observation_Date'][tmplist_idx], format='decimalyear').mjd)
                if time_difference_day > 1.:
                    1/0
                # scamp_xml.table['Observation_Date'][tmplist_idx]
                aux[i, :] = [Time(header['MJD-OBS'], format='mjd').decimalyear, catnum, header['AIRMASS'], header['EXPTIME'],
                             header['TAMBIENT'],
                             header['PRESSURE'], header['HUMIDITY'], header['WINDSPEE'], header['AZIMUTH'], header['ELEVATIO'],
                             SkyCoord(ra='00:00:00.000', dec=latitude_string, unit='deg').dec.deg, header['DEC'],
                             header['RA'],
                             header['MJD-OBS'], header['AIRMASS'], 0.0]

            else:  # HAWKI
                aux[i, :] = [scamp_xml.table['Observation_Date'][tmplist_idx], 0.0, header['HIERARCH ESO TEL AIRM START'],
                             header['HIERARCH ESO DET DIT'], header['HIERARCH ESO TEL AMBI TEMP'],
                             header['HIERARCH ESO TEL AMBI PRES START'], header['HIERARCH ESO TEL AMBI RHUM'],
                             header['HIERARCH ESO TEL AMBI WINDSP'], header['HIERARCH ESO TEL AZ'],
                             header['HIERARCH ESO TEL ALT'], header['HIERARCH ESO TEL GEOLAT'], header['DEC'], header['RA'],
                             header['MJD-OBS'], header['HIERARCH ESO TEL AIRM START'],
                             header['HIERARCH ESO ADA ABSROT START']]

            for j, IDN in enumerate(good_source_numbers):
                sidx = np.where((self.table['SOURCE_NUMBER'] == IDN) & (self.table['CATALOG_NUMBER'] == catnum))[0]
                #  flux, background level, X, Y, epochNr, seeing_X, seeing Y, IDN, chip
                if not use_Idl:
                    p[i, j, :] = [self.table['MAG'][sidx], self.table['FLAGS_EXTRACTION'][sidx], self.table['X_IMAGE'][sidx],
                                  self.table['Y_IMAGE'][sidx], i, self.table['ERRA_WORLD'][sidx], IDN, chip_extension,
                                  self.table['ERRA_IMAGE'][sidx], self.table['ERRB_IMAGE'][sidx]]
                else:
                    p[i, j, :] = [self.table['MAG'][sidx], self.table['FLAGS_EXTRACTION'][sidx], self.table['X_Idl'][sidx],
                                  self.table['Y_Idl'][sidx], i, self.table['ERRA_WORLD'][sidx], IDN, chip_extension,
                                  self.table['ERRA_IMAGE'][sidx], self.table['ERRB_IMAGE'][sidx], self.table['CATALOG_NUMBER'][sidx], self.table['artificial'][sidx]]

                # names of third dimension in p
                col_names = np.array(['MAG', 'FLAGS_EXTRACTION', 'x', 'y', 'index', 'sigma_world', 'id' , 'CHIP_EXTENSION', 'sigma_x', 'sigma_y', 'CATALOG_NUMBER', 'artificial'])

        # ATTENTION here: 'epoch_yr' stems from the scamp data 'Observation_Date' whereas 'MJD' comes from the file header 'MJD-OBS'
        aux = Table(aux, names=('scamp_epoch_yr', 'catalog_number', 'airmass', 'exptime', 'temperature', 'pressure' ,'rel_humidity', 'windspeed', 'tel_azimuth', 'tel_altitude', 'geo_latitude', 'dec' , 'ra', 'MJD', 'airmass2', 'abs_rotation'))
        if np.max(np.abs(aux['MJD'] - Time(aux['scamp_epoch_yr'], format='decimalyear').mjd)) > 1e-4:
            raise RuntimeError

        mp = pystortion.distortion.multiEpochAstrometry(p, col_names, aux)

        return mp


class ScampMergedCatalog(object):
    """
    Class for handling scamp merged.cat output file
    http://scamp.readthedocs.io/en/latest/Output.html#merged-catalogues
    """
    def __init__(self, scamp_dir):
        self.table, self.source_file = read_scamp_cat(scamp_dir, 'merged')
        self.source_dir = scamp_dir

    def info(self):
        print('{}: {} entries'.format(self.source_file, len(self.table)))

    def select_alignment_source_number(self, minimum_detections):
        """Return source numbers that will be used for alignment with Gaia/external catalog.

        Parameters
        ----------
        minimum_detections : int
            minimum number of scamp detections

        Returns
        -------
            numpy array

        """
        idx = np.where((self.table['FLAGS_SCAMP'] == 32) & (self.table['NPOS_OK'] > minimum_detections))[0]
        print('{}: {} sources have at least {} scamp detections and FLAGS_SCAMP == 32'.format(self.source_file, len(idx), minimum_detections))

        return np.array(self.table['SOURCE_NUMBER'][idx])


    def select_scamp_full(self, scamp_full, minimum_detections):
        """
        Return a ScampFullCatalog object that contains only sources that pass certain qualifiers

        :param minimum_detections:
        :return:
        """
        source_numbers = self.select_alignment_source_number(minimum_detections)
        idx_full = np.in1d(scamp_full.table['SOURCE_NUMBER'].data, source_numbers)
        scamp_full_select = ScampFullCatalog(table=scamp_full.table[idx_full])
        return scamp_full_select

    def remove_source(self, source_number):
        """
        Remove a source from catalog table

        :param source_number:
        :return:
        """
        self.table.remove_rows(np.where(self.table['SOURCE_NUMBER'] == source_number)[0])


    def source_numbers_within_radius(self, good_source_numbers, RA_deg, DE_deg, radius):
        """
        Return subset of good_source_numbers that lie within radius of Ra,Dec

        :param good_source_numbers:
        :param RA_deg:
        :param DE_deg:
        :param radius:
        :return:
        """
        source_number_list = self.table['SOURCE_NUMBER'].tolist()
        table_index = np.array([source_number_list.index(j) for j in good_source_numbers])
        cat = SkyCoord(ra=self.table['ALPHA_J2000'][table_index]*u.deg, dec=self.table['DELTA_J2000'][table_index]*u.deg)
        index = np.where(SkyCoord(ra=RA_deg*u.deg, dec=DE_deg*u.deg).separation(cat) < radius)[0]
        return good_source_numbers[index]


def getScampXml(targetDir, target_file=None):
    if target_file:
        pass;
    else:
        target_file = 'scamp.xml';
    votable = parse(os.path.join(targetDir, target_file));
    xmatchTable = votable.get_first_table().to_table();
    return xmatchTable, target_file


def read_scamp_cat(targetDir, tag, target_file=None):
    if target_file is None:
        target_file = '{}_1.cat'.format(tag)
    data = fits.getdata(os.path.join(targetDir, target_file), 2);
    tf = Table(data);
    return tf, target_file


def getScampFullCat(targetDir, target_file=None):
    if target_file:
        pass;
    else:
        target_file = 'full_1.cat';
    data = fits.getdata(os.path.join(targetDir, target_file), 2);
    tf = Table(data);
    return tf


def getScampMergedCat(targetDir, target_file=None):
    if target_file:
        pass;
    else:
        target_file = 'merged_1.cat';
    data = fits.getdata(os.path.join(targetDir, target_file), 2);
    tm = Table(data);
    return tm


def xfMakeAplpyPngForFrameInScampXml(tf, xmatchTable, dirlist, chip_extension, outDir, frameIdx=0, RA_deg=None,
                                     DE_deg=None):
    # number of entries in the full table
    scampCatLen = len(tf);
    print('number of entries in the full table %d' % scampCatLen)
    print('Number of individual SOURCE_NUMBER in full table: %d' % len(np.unique(tf['SOURCE_NUMBER'])))

    # name of the individual frame FITS file being plotted
    # frameName = xmatchTable['Catalog_Name'][frameIdx].replace('.cat', '.fits')
    frameName = (xmatchTable['Catalog_Name'][frameIdx]).decode().replace('.cat', '.fits')
    CatalogueNumber = xmatchTable['Catalog_Number'][frameIdx];

    # select stars that were detected in that frame
    starDict = [{"name": str(tf['SOURCE_NUMBER'][i]), "x": tf['X_IMAGE'][i], "y": tf['Y_IMAGE'][i]} for i in
                range(scampCatLen) if ((tf['EXTENSION'][i] == chip_extension) & (tf['ASTR_INSTRUM'][i] == 1) & (
        tf['CATALOG_NUMBER'][i] == CatalogueNumber))];

    T = Table(starDict)
    idx = [i for i, s in enumerate(dirlist) if frameName in s]
    image_file = dirlist[idx[0]]

    #     from astropy import wcs
    from astropy.wcs import WCS
    header = fits.getheader(image_file)

    # catch case of OSIRIS where WCS is in second extension header, replace primary header by header of extension 2
    if 'DATE-OBS' not in header.keys():
        header = fits.getheader(image_file, ext=chip_extension)
        for key in header.keys():
            if 'PV' in key:
                header.remove(key)

                #     1/0
                #     w = WCS(image_file)
    w = WCS(header=header)
    # coordArray = map(lambda x,y:[x,y],T['x'],T['y'])
    coordArray = np.array([T['x'].data, T['y'].data]).T
    # 1/0
    coordArray_world = w.wcs_pix2world(coordArray, 0)

    T['RA_deg'] = coordArray_world[:, 0]
    T['DE_deg'] = coordArray_world[:, 1]
    # find the frame image
    # print dataDir
    # print frameName
    # dirlist1=glob.glob('%s/%s' % (dataDir,frameName) );
    # if not dirlist1:
    #     dirlist1=glob.glob('%s**/**/%s' % (dataDir,frameName) );


    cat = SkyCoord(ra=np.array(T['RA_deg']) * u.degree, dec=np.array(T['DE_deg']) * u.degree)

    fig = pl.figure(figsize=(7, 7), facecolor='w', edgecolor='k');
    pl.clf();

    try:
        gc = aplpy.FITSFigure(image_file, extension=chip_extension, figure=fig,
                              North=True)  # ,convention='wells')#, dimensions=[1 ,0]);

    except:  # InvalidTransformError
        #
        #         # for files reduced by Herve that contain the PV1_5 keywords
        data = fits.getdata(image_file, ext=chip_extension)
        hdu = fits.PrimaryHDU(data)
        hdu.header = header
        gc = aplpy.FITSFigure(hdu, extension=chip_extension, figure=fig,
                              North=True)  # ,convention='wells')#, dimensions=[1 ,0]);

    gc.show_grayscale(
        invert=False)  # , stretch='log', vmid=-1)#,vmid=-1)#, aspect = pixScaleAC_mas/pixScaleAL_mas, pmax =90 )

    radius_as = 50.
    if RA_deg is not None:
        gc.recenter(RA_deg, DE_deg, radius=radius_as / 3600.)

    gc.show_markers(RA_deg, DE_deg, marker='s', edgecolor='y', linewidth=3, s=30)
    if 1 == 1:
        target_cat = SkyCoord(ra=RA_deg * u.degree, dec=DE_deg * u.degree)
        idx = np.where(target_cat.separation(cat).to(u.arcsec).value < radius_as)[0]
        gc.show_markers(T['RA_deg'][idx] - 0 / 3600., T['DE_deg'][idx] - 0 / 3600., marker='o', edgecolor='y')

        #         for j,lbl in  enumerate(T['name']):
        #             gc.add_label(T['RA_deg'][j]-0/3600.,T['DE_deg'][j]+1/3600.,lbl)
        for j in idx:
            gc.add_label(T['RA_deg'][j] - 0 / 3600., T['DE_deg'][j] + 1 / 3600., T['name'][j])

        target_index = np.argmin(target_cat.separation(cat))
        target_name = T['name'][target_index]
        print('Identified SOURCE_NUMBER {0} as closest {1} to expected target location'.format(target_name, np.min(
            target_cat.separation(cat)).to(u.arcsec)))

    saveFile = os.path.join(outDir,
                            xmatchTable.field('Catalog_Name')[frameIdx].decode().replace('.cat', '_SexStars.png'));
    gc.save(saveFile, dpi=300);
    pl.show()

    target_number = np.int(target_name)
    return target_number


def xf_make_frame_png_with_merged_scamp_catalog(tm, xmatchTable, dirlist, chipExtension, outDir, frameIdx=0,
                                                RA_deg=None, DE_deg=None):
    '''
    show target and reference star in FITS image with SOURCE_NUMBER
    fixed 2017-07-26 for WCS bug (yet not fully solved in xfMakeAplpyPngForFrameInScampXml)
    '''

    # number of entries in the full table
    scampCatLen = len(tm);
    print('number of entries in the merged table %d' % scampCatLen)
    print('Number of individual SOURCE_NUMBER in full table: %d' % len(np.unique(tm['SOURCE_NUMBER'])))

    # name of the individual frame FITS file being plotted
    frameName = xmatchTable['Catalog_Name'][frameIdx].replace('.cat', '.fits');
    CatalogueNumber = xmatchTable['Catalog_Number'][frameIdx];

    # select stars from merged catalog whether they were detected or not
    T = tm['SOURCE_NUMBER', 'ALPHA_J2000', 'DELTA_J2000']
    idx = [i for i, s in enumerate(dirlist) if frameName in s]
    image_file = dirlist[idx[0]]

    cat = SkyCoord(ra=np.array(T['ALPHA_J2000']) * u.degree, dec=np.array(T['DELTA_J2000']) * u.degree)

    fig = pl.figure(figsize=(7, 7), facecolor='w', edgecolor='k');
    pl.clf();

    gc = aplpy.FITSFigure(image_file, extension=chipExtension, figure=fig, North=True)
    gc.show_grayscale(
        invert=False)  # , stretch='log', vmid=-1)#,vmid=-1)#, aspect = pixScaleAC_mas/pixScaleAL_mas, pmax =90 )

    radius_as = 30.
    if RA_deg is not None:
        gc.recenter(RA_deg, DE_deg, radius=radius_as / 3600.)

    gc.show_markers(RA_deg, DE_deg, marker='s', edgecolor='y', linewidth=3, s=30)
    if 1 == 1:
        target_cat = SkyCoord(ra=RA_deg * u.degree, dec=DE_deg * u.degree)
        idx = np.where(target_cat.separation(cat).to(u.arcsec).value < radius_as)[0]
        gc.show_markers(T['ALPHA_J2000'][idx] - 0 / 3600., T['DELTA_J2000'][idx] - 0 / 3600., marker='o', edgecolor='y')

        for j in idx:
            gc.add_label(T['ALPHA_J2000'][j] - 0 / 3600., T['DELTA_J2000'][j] + 1 / 3600., T['SOURCE_NUMBER'][j])

        target_index = np.argmin(target_cat.separation(cat))
        print('Identified SOURCE_NUMBER {0} as closest {1} to expected target location'.format(
            T['SOURCE_NUMBER'][target_index], np.min(target_cat.separation(cat)).to(u.arcsec)))

    saveFile = os.path.join(outDir, xmatchTable.field('Catalog_Name')[frameIdx].replace('.cat', '_sources.png'));
    gc.save(saveFile, dpi=300);
    pl.show()


def xfDetermineTargetSourceNumber(tm, tmpIdx, RA_deg, DE_deg):  # , minimum_detections=0):
    #   gidx = np.where( tm['NPOS_OK'] > minimum_detections)[0];
    #     degDiffs = np.array([ np.sqrt( (RA_deg-tm['ALPHA_J2000'][i] )**2 + (DE_deg-tm['DELTA_J2000'][i] )**2 ) for i in tmpIdx ]);
    #     minDistIdx = np.argmin(degDiffs);
    #     target_number = tm['SOURCE_NUMBER'][tmpIdx[minDistIdx]];


    #     unique_SOURCE_NUMBER, unique_source_index = np.unique(tm['SOURCE_NUMBER'], return_index=True)

    print('number of entries in the merged table %d' % len(tm))
    print('Number of individual SOURCE_NUMBER in merged table: %d' % len(np.unique(tm['SOURCE_NUMBER'])))

    #     tm_unique = tm[unique_source_index]

    # catalog of detected stars in merged catalog
    cat = SkyCoord(ra=np.array(tm['ALPHA_J2000']) * u.degree, dec=np.array(tm['DELTA_J2000']) * u.degree)

    # expected position of target
    target_cat = SkyCoord(ra=RA_deg * u.degree, dec=DE_deg * u.degree)

    # separations
    distance_from_target = target_cat.separation(cat)

    # crossmatch radius
    radius = 10 * u.arcsec

    idx = np.where(distance_from_target < radius)[0]
    print('{0} stars in merged table are within {1} of the target'.format(len(idx), radius))
    for j in idx:
        print('{0}, distance {1}'.format(tm['SOURCE_NUMBER'][j], distance_from_target[j].to(u.arcsec)))

    target_index = np.argmin(distance_from_target.value)
    targetNumber = tm['SOURCE_NUMBER'][target_index];

    #     degDiffs = np.array([ np.sqrt( (RA_deg-tm['ALPHA_J2000'][i] )**2 + (DE_deg-tm['DELTA_J2000'][i] )**2 ) for i in tmpIdx ]);
    #     minDistIdx = np.argmin(degDiffs);
    #     target_number = tm['SOURCE_NUMBER'][tmpIdx[minDistIdx]];
    print('Target identified as reference star with SOURCE_NUMBER {:d} at {:3.4f}'.format(targetNumber, np.min(
        distance_from_target).to(u.arcsec)))

    #     print ("Target identified as reference star with SOURCE_NUMBER ##%d at %3.4f pix (%3.4f arcsec)" % (target_number,np.min(degDiffs)*3600./pixelscale,np.min(degDiffs)*3600.))
    if 0 == 1:
        pl.figure(1, figsize=(8, 5), facecolor='w', edgecolor='k');
        pl.clf();
        pl.plot(tmpIdx, degDiffs * 3600., 'ro');
        pl.ylim((0, 1000))
        plt.show()

    return targetNumber


def xfGetSuitableReferenceSources(tf, tm, tmpIdx, targetNumber, chipExtension):
    targetFrameNumbers = np.where(tf['SOURCE_NUMBER'] == targetNumber)[0];
    targetCatalogNumbers = tf['CATALOG_NUMBER'][targetFrameNumbers];
    targetCatalogNumbers.sort();
    print("Target identified in %d frames out of %d" % (len(targetFrameNumbers), len(np.unique(tf['CATALOG_NUMBER']))))

    #     TODO:
    #     - review reference star selection
    #     - check for number of astrometric instruments

    sourceNumbers = tm['SOURCE_NUMBER'][tmpIdx];
    goodSourceNumbers = [];
    for i, num in enumerate(sourceNumbers):
        tmp = np.where(tf['SOURCE_NUMBER'] == num)[0];
        if ((tf['EXTENSION'][tmp[0]] == chipExtension) & (tf['ASTR_INSTRUM'][tmp[0]] == 1) & (
            len(targetFrameNumbers) <= len(tmp))):  # select only stars in selected chip and catalogue 1
            if all([j in tf['CATALOG_NUMBER'][tmp] for j in targetCatalogNumbers]):
                goodSourceNumbers.append(num);
                #     pdb.set_trace()
    print ('Selected %d reference stars as potential good reference stars' % (len(goodSourceNumbers)));

    if targetNumber not in goodSourceNumbers:
        print ("target not identified in all selected frames: ERROR");
        1 / 0

    return np.array(goodSourceNumbers), targetFrameNumbers, targetCatalogNumbers


def xfGetReferenceSourcesInRopt(s_in, tm, RA_deg, DE_deg, R_opt_arcsec):
    # eliminate reference stars outside of circular area
    # s_out = [ j for j in s_in if (np.sqrt( (RA_deg-array(tm.field('ALPHA_J2000')[tm.field('SOURCE_NUMBER').tolist().index(j)]) )**2 + (DE_deg-array(tm.field('DELTA_J2000')[tm.field('SOURCE_NUMBER').tolist().index(j)]) )**2 ) < R_opt_arcsec/3600.) ];
    # degDiffs = np.array([ np.sqrt( (RA_deg-tm['ALPHA_J2000'][i] )**2 + (DE_deg-tm['DELTA_J2000'][i] )**2 ) for i in s_in ]);
    # idxs = np.array(np.where( degDiffs < R_opt_arcsec/3600. )[0]);
    # s_out = s_in[idxs]

    indexInTmOfSourceNumbers = np.array([tm['SOURCE_NUMBER'].tolist().index(j) for j in s_in]);
    degDiffs = np.sqrt((RA_deg - tm['ALPHA_J2000'][indexInTmOfSourceNumbers]) ** 2 + (
    DE_deg - tm['DELTA_J2000'][indexInTmOfSourceNumbers]) ** 2)
    idxs = np.array(np.where(degDiffs < R_opt_arcsec / 3600.)[0]);
    s_out = np.array(s_in[idxs]);
    # s_out = [ j for j in s_in if (np.sqrt( (RA_deg-tm['ALPHA_J2000'][] )**2 + (DE_deg-tm['DELTA_J2000'][tm['SOURCE_NUMBER'].tolist().index(j)] )**2 ) < R_opt_arcsec/3600.) ];
    #     print '%d remaining ref stars within R_opt' % len(s_out);
    return s_out


def xfGetInitialAstrometricData(xmatchTable, tf, dirlist, goodSourceNumbers, targetCatalogNumbers, chipExtension):
    Nstars = len(goodSourceNumbers);
    Nframes = len(targetCatalogNumbers);
    # Nframes  = len(targetFrameNumbers)

    p = numpy.zeros((Nframes, Nstars, 10))
    aux = numpy.zeros((Nframes, 16))
    tmplist = array(xmatchTable['Catalog_Number']).tolist();
    for i, catnum in enumerate(targetCatalogNumbers):
        tmplistIdx = tmplist.index(catnum)
        frameName = xmatchTable['Catalog_Name'][tmplistIdx].decode()
        frame = [dirlist[ii] for ii in range(len(dirlist)) if frameName.replace('.cat', '') in dirlist[ii]][0];
        #         print "%03d/%03d: filling catalogue with stars from %s" % (i+1,Nframes,frame);
        # head = fitsio.read_header(frame)           #
        head = fits.getheader(frame);
        # pdb.set_trace()
        if head.get('ORIGIN') == None:
            #         if not head.has_key('ORIGIN'): # for OSIRIS
            head = fits.getheader(frame, ext=2);
            # head['MJD-OBS']
            # print 'Inserting FITS keyword: MJD-OBS';
            # head.update('MJD-OBS', sla.slalib.sla_epj2d(xmatchTable.field('Observation_Date')[i]));
            # pdb.set_trace()
            aux[i, :] = [xmatchTable['Observation_Date'][i], 0.0, xmatchTable['AirMass'][i],
                         xmatchTable['Exposure_Time'][i], head['TAMBIENT'], head['PRESSURE'], head['HUMIDITY'],
                         head['WINDSPEE'], head['AZIMUTH'], head['ELEVAT'],
                         SkyCoord(ra=head['LATITUDE'], dec='00:00:00.000', unit='deg').ra.deg, head['DECDEG'],
                         head['RADEG'], head['MJD-OBS'], head['AIRMASS'], 0]

        elif head['INSTRUME'].strip() in ['GMOS-S', 'GMOS-N']:
            if head['INSTRUME'].strip() == 'GMOS-S':
                latitude_string = '-30:14:26.700 '
            elif head['INSTRUME'].strip() == 'GMOS-N':
                latitude_string = '19:49:25.7016'


                # removed head['HIERARCH ESO TEL GEOLAT']
            #             aux[i,:] = [  xmatchTable['Observation_Date'][i], 0.0 , head['AIRMASS'], head['EXPTIME'] ,head['TAMBIENT'], head['PRESSURE'], head['HUMIDITY'], head['WINDSPEE'], head['AZIMUTH'], head['ELEVATIO'], astCoords.dms2decimal('-30:14:26.700',':') , head['DEC'], head['RA'], head['MJD-OBS'], head['AIRMASS'], 0.0]
            aux[i, :] = [xmatchTable['Observation_Date'][i], 0.0, head['AIRMASS'], head['EXPTIME'], head['TAMBIENT'],
                         head['PRESSURE'], head['HUMIDITY'], head['WINDSPEE'], head['AZIMUTH'], head['ELEVATIO'],
                         SkyCoord(ra=latitude_string, dec='00:00:00.000', unit='deg').ra.deg, head['DEC'], head['RA'],
                         head['MJD-OBS'], head['AIRMASS'], 0.0]

        else:  # HAWKI
            aux[i, :] = [xmatchTable['Observation_Date'][i], 0.0, head['HIERARCH ESO TEL AIRM START'],
                         head['HIERARCH ESO DET DIT'], head['HIERARCH ESO TEL AMBI TEMP'],
                         head['HIERARCH ESO TEL AMBI PRES START'], head['HIERARCH ESO TEL AMBI RHUM'],
                         head['HIERARCH ESO TEL AMBI WINDSP'], head['HIERARCH ESO TEL AZ'],
                         head['HIERARCH ESO TEL ALT'], head['HIERARCH ESO TEL GEOLAT'], head['DEC'], head['RA'],
                         head['MJD-OBS'], head['HIERARCH ESO TEL AIRM START'], head['HIERARCH ESO ADA ABSROT START']]

        for j, IDN in enumerate(goodSourceNumbers):
            sidx = np.where((tf['SOURCE_NUMBER'] == IDN) & (tf['CATALOG_NUMBER'] == catnum))[0]
            #  flux, background level, X, Y, epochNr, seeing_X, seeing Y, IDN, chip
            # p[i,j,:] = [tf.field('MAG')[sidx], 0. , tf.field('X_IMAGE')[sidx] , tf.field('Y_IMAGE')[sidx] , i , 1 , IDN , chipExtension, tf['ERRA_IMAGE'][sidx] , tf['ERRB_IMAGE'][sidx]];
            p[i, j, :] = [tf.field('MAG')[sidx], tf.field('FLAGS_EXTRACTION')[sidx], tf.field('X_IMAGE')[sidx],
                          tf.field('Y_IMAGE')[sidx], i, tf.field('ERRA_WORLD')[sidx], IDN, chipExtension,
                          tf['ERRA_IMAGE'][sidx], tf['ERRB_IMAGE'][sidx]];
            # p[i,j,:] = [tf.field('MAG')[sidx], 0. , tf.field('X_IMAGE')[sidx] , tf.field('Y_IMAGE')[sidx] , i , int(frame.split('/')[6][2:]) , IDN , chipExtension, tf.field('ERRA_IMAGE')[sidx] , tf.field('ERRB_IMAGE')[sidx]];
            #         p[i,target_index,:] = [flx, bck , params[1] , params[2] , params[3] , params[4] , IDN , chip];

    return p, aux
