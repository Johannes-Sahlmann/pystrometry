#!/usr/bin/env python
"""Tests for the utils.archives module.

Authors
-------
    Johannes Sahlmann

"""
import os
import numpy as np

from ..archives import get_gaiadr_data

def test_basic_query():
    analysis_dataset_name = 'testing'
    data_dir = os.path.join(os.getcwd(), 'tmp')
    source_id_array = np.array([3482973840016015744])
    gaia_data_release = 'gaiadr3'
    gaia_table_name = 'gaia_source'

    df = get_gaiadr_data(analysis_dataset_name, data_dir, source_id_array, gaia_data_release, gaia_table_name)
    assert len(df) == 1
