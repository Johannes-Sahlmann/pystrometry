"""Some query helpers.

Authors
-------
    Johannes Sahlmann

"""
import os
from astropy.table import Table
from astroquery.gaia import Gaia, TapPlus

import pandas as pd


def get_gaiadr_data(analysis_dataset_name, data_dir, source_id_array=None, gaia_data_release='dr3int5',
                    overwrite_query=False, gaia_table_name='gaia_source', shared_user_name='dr3int5'):
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
            gaia = TapPlus(url="http://geapre.esac.esa.int/tap-server/tap")
            if getattr(gaia, '_TapPlus__isLoggedIn') is False:
                gaia.login()
            table_name = 'user_{}'.format(shared_user_name)
        else:
            gaia = Gaia
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

