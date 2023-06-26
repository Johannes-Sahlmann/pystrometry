"""Some query helpers.

Authors
-------
    Johannes Sahlmann

"""
import logging
import os
from astropy.table import Table
import astropy.units as u
from astroquery.gaia import Gaia, TapPlus
from astropy.time import Time
import pandas as pd


def get_gaiadr_data(analysis_dataset_name, data_dir, source_id_array=None, gaia_data_release='dr3int5',
                    overwrite_query=False, gaia_table_name='gaia_source', shared_user_name=None,
                    gacs_connection=None):
    """Query a Gaia archive table by source_id. Only data corresponding to source_id_array are returned.

    Parameters
    ----------
    analysis_dataset_name
    data_dir
    source_id_array
    gaia_data_release
    overwrite_query
    gaia_table_name

    Returns
    -------

    """

    #     retrieve Gaia DR data by submitting list of source_id to GACS
    output_file = os.path.join(data_dir, '{}_{}_sources.parquet'.format(gaia_data_release, analysis_dataset_name))

    if (not os.path.isfile(output_file)) or (overwrite_query):


        if 'int' in gaia_data_release:
            if gacs_connection is None:
                gaia = TapPlus(url="http://geapre.esac.esa.int/tap-server/tap")
            else:
                gaia = gacs_connection
            if getattr(gaia, '_TapPlus__isLoggedIn') is False:
                gaia.login()
            if shared_user_name is None:
                shared_user_name = gaia_data_release
            table_name = 'user_{}'.format(shared_user_name)
        else:
            gaia = Gaia
            if shared_user_name is not None:
                if getattr(gaia, '_TapPlus__isLoggedIn') is False:
                    gaia.login()
                table_name = 'user_{}'.format(shared_user_name)
            else:
                table_name = '{}'.format(gaia_data_release)

        if source_id_array is not None:
            input_table_name = '{}_source_id'.format(analysis_dataset_name)
            input_table = os.path.join(data_dir, '%s.vot' % input_table_name)
            source_id_column_name = 'source_id'
            Table([source_id_array], names=['source_id']).write(input_table, format='votable', overwrite=True)

            query = '''
            SELECT * FROM
            (select * FROM tap_upload.{0}) AS input
            INNER JOIN
            (select * FROM {1}.{3}) AS gdr
            ON (input.{2} = gdr.source_id)
            ;'''.format(input_table_name, table_name, source_id_column_name, gaia_table_name)

            job = gaia.launch_job_async(query=query, upload_resource=input_table,
                                  upload_table_name=input_table_name, verbose=True)
                                        # dump_to_file=True, output_file=output_file)
        else:
            query = 'select * FROM {}.{};'.format(table_name, gaia_table_name)
            job = gaia.launch_job_async(query=query, verbose=True)

        table = job.get_results()

        df = table.to_pandas()
        df.to_parquet(output_file)
    else:
        df = pd.read_parquet(output_file)

    print('Retrieved {} rows from {}.{}'.format(len(df), gaia_data_release, gaia_table_name))

    return df


def query_dpcg(connection, out_dir, tag='dpcgdata', query=None, reference_time=None,
               selected_source_id_string=None, overwrite=False):

    out_file = os.path.join(out_dir, f'dpcg_{tag}.parquet')

    if overwrite or (os.path.isfile(out_file) is False):
        assert connection.closed == 0


        # reference_time = Time(nss_all['ref_epoch'][0], format='jyear')
        ref_epoch_jd = reference_time.jd
        # selected_source_id_string = ','.join(selected_source_id_array.astype(str))



        if query is None:
            query = f"""
            select 
            sourceid as source_id,
            -- DR3 position reference epoch: 2016-01-01T12:00:00.000000000 (TCB) =  JD 2457389.0 (update at 3 placed below when changed)
            ((t).obstime - {ref_epoch_jd})/365.25 as t_min_t0_yr,
            cos( (t).scanposangle) as cpsi_obs,
            sin( (t).scanposangle) as spsi_obs,
            (t).varpifactoral as ppfact_obs,
            ((t).obstime - {ref_epoch_jd})/365.25*cos( (t).scanposangle) as tcpsi_obs,
            ((t).obstime - {ref_epoch_jd})/365.25*sin( (t).scanposangle) as tspsi_obs,
            (t).centroidposal as da_mas_obs,
            (t).centroidposerroral as errda_mas_obs,
            -- remove last 4 bits as these were added by CU4 to encode ccd number
            (t).transitid/16 as transitid,
            -- convert last 4 bits of CU4 transitid into number
            ( (t).transitid - (t).transitid/16*16 ) as ccdnumber,
            array_length(transits,1) as num_obs_for_src
        
            -- from mdb_gaia_starobject_088 
            -- join lateral unnest(transits) t on true
            from mdb_gaia_starobject_088 so
            join dr3_ops_cs36_mv.dgdreq58_rejected_cu4transitids_astro transrej using (sourceid)
            join lateral unnest(filter_transits(transits,rejected_cu4transitids_astro)) as t on true
            -- provide source id list (comma separated)
            where sourceid in ({selected_source_id_string})
            """

        dpcg_df = pd.read_sql(query, connection)
        dpcg_df.to_parquet(out_file)
        logging.info(f'Wrote {len(dpcg_df)} rows to {out_file}')
    else:
        dpcg_df = pd.read_parquet(out_file)
        logging.info(f'Read {len(dpcg_df)} rows from {out_file}')
    return dpcg_df


def query_dpcg_epochastrometry(connection, out_dir, tag='dpcgdata', query=None, reference_time=None,
               selected_source_id_string=None, overwrite=False):
    """This query already applies the CU4 DU432 pre-processing filter.

    :param connection:
    :param out_dir:
    :param tag:
    :param query:
    :param reference_time:
    :param selected_source_id_string:
    :param overwrite:
    :return:
    """

    out_file = os.path.join(out_dir, f'dpcg_{tag}.parquet')

    if overwrite or (os.path.isfile(out_file) is False):
        assert connection.closed == 0

        # reference_time = Time(nss_all['ref_epoch'][0], format='jyear')
        ref_epoch_jd = reference_time.jd
        # selected_source_id_string = ','.join(selected_source_id_array.astype(str))

        tcb_ref_epoch_jd = Time(2010.0, format='jyear', scale='tcb').jd

        if query is None:
            query = f"""
            select
            sourceid as source_id,
            -- DR3 position reference epoch: 2016-01-01T12:00:00.000000000 (TCB) =  JD 2457389.0 (update at 3 placed below when changed)
            ((t).obstime - {ref_epoch_jd})/365.25 as t_min_t0_yr,
            ((t).obstime - {tcb_ref_epoch_jd})*{u.day.to(u.nanosecond)} as obsTimeTcb,
            ((t).scanposangle) as scanPosAngle,
            cos( (t).scanposangle) as cpsi_obs,
            sin( (t).scanposangle) as spsi_obs,
            (t).varpifactoral as parallaxFactorAl,
            ((t).obstime - {ref_epoch_jd})/365.25*cos( (t).scanposangle) as tcpsi_obs,
            ((t).obstime - {ref_epoch_jd})/365.25*sin( (t).scanposangle) as tspsi_obs,
            (t).centroidposal as centroidPosAl,
            (t).centroidposerroral as centroidPosErrorAl,
            -- remove last 4 bits as these were added by CU4 to encode ccd number
            (t).transitid/16 as transitid,
            -- convert last 4 bits of CU4 transitid into number
            ( (t).transitid - (t).transitid/16*16 ) as ccdnumber,
            (astromsource).alpha as ra0,	
            (astromsource).delta as dec0,	
            array_length(transits,1) as num_obs_for_src

            -- from mdb_gaia_starobject_088
            -- join lateral unnest(transits) t on true
            from mdb_gaia_starobject_088 so
            join dr3_ops_cs36_mv.dgdreq58_rejected_cu4transitids_astro transrej using (sourceid)
            join lateral unnest(filter_transits(transits,rejected_cu4transitids_astro)) as t on true
            -- provide source id list (comma separated)
            where sourceid in ({selected_source_id_string})
            """

        dpcg_df = pd.read_sql(query, connection)
        dpcg_df.to_parquet(out_file)
        logging.info(f'Wrote {len(dpcg_df)} rows to {out_file}')
    else:
        dpcg_df = pd.read_parquet(out_file)
        logging.info(f'Read {len(dpcg_df)} rows from {out_file}')
    return dpcg_df




