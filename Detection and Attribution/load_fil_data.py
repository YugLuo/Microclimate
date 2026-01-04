#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import rasterio
from scipy import signal
import glob

def load_tif_data(path, years, model=None):
    data = []
    for year in years:
        if model:
            tif_path = os.path.join(path, model, f"{year}.tif")
        else:
            tif_path = os.path.join(path, f"{year}.tif")
        try:
            with rasterio.open(tif_path) as src:
                data.append(src.read(1))
        except Exception as e:
            print(f"Error loading {tif_path}: {e}")
            return None
    return np.stack(data, axis=0) if data else None

def calculate_area_mean(data):
    return np.nanmean(data, axis=(1, 2)) if data.ndim == 3 else np.nanmean(data, axis=0)

def find_consecutive_blocks(years, block_size=39):
    years = sorted(years)
    blocks = []
    i = 0
    while i < len(years) - block_size + 1:
        if years[i + block_size - 1] - years[i] == block_size - 1:
            blocks.append(years[i:i + block_size])
            i += block_size
        else:
            i += 1
    return blocks

def prefilt(var, in_pth, Ctype, sigma, base_per, start_year, bg, mem):
    BASE_PATH = "/ampha/tenant/fafu/private/user/guanyl/lyg/data/DA-DAMIP/Tunder-yanmo/Asia/"
    OBS_PATH = os.path.join(BASE_PATH, "obs")
    ANT_PATH = os.path.join(BASE_PATH, "ant")
    NAT_PATH = os.path.join(BASE_PATH, "hist-nat")
    PIC_PATH = os.path.join(BASE_PATH, "pic")

    years = list(range(1982, 2021))  # 1982-2020 for obs, ant, nat
    pic_years = list(range(1850, 2401))  # 1850-2100 for pic
    Year = np.array(years)
    base_years = years[:base_per]
    base_ind = list(range(base_per))

    def get_valid_models(path, required_years):
        models = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        valid_models = []
        for model in models:
            all_files_exist = True
            for year in required_years:
                if not os.path.exists(os.path.join(path, model, f"{year}.tif")):
                    all_files_exist = False
                    break
            if all_files_exist:
                valid_models.append(model)
        return valid_models

    ant_models = get_valid_models(ANT_PATH, years)
    nat_models = get_valid_models(NAT_PATH, years)

    print(f"Debug - ant_models: {len(ant_models)} models: {ant_models}")
    print(f"Debug - nat_models: {len(nat_models)} models: {nat_models}")

    block_size = 39
    all_models = [d for d in os.listdir(PIC_PATH) if os.path.isdir(os.path.join(PIC_PATH, d))]
    pic_data_list = []
    valid_blocks_per_model = {}

    for model in all_models:
        available_years = [year for year in pic_years if os.path.exists(os.path.join(PIC_PATH, model, f"{year}.tif"))]
        if len(available_years) < block_size:
            continue
        blocks = find_consecutive_blocks(available_years, block_size)
        if not blocks:
            continue
        model_data = load_tif_data(PIC_PATH, available_years, model=model)
        if model_data is None:
            continue
        model_mean = calculate_area_mean(model_data)
        base_mean = np.mean(model_mean)
        model_anom = model_mean - base_mean
        year_to_idx = {year: idx for idx, year in enumerate(available_years)}
        for block in blocks:
            indices = [year_to_idx[year] for year in block]
            block_anom = model_anom[indices]
            pic_data_list.append(block_anom)
        valid_blocks_per_model[model] = len(blocks)

    if not pic_data_list:
        raise ValueError("No valid PIC blocks found")

    pic_blocks = np.array(pic_data_list).T  # (39, total_blocks)
    pic_models = list(valid_blocks_per_model.keys())

    print(f"Debug - pic_models: {len(pic_models)} models: {pic_models}")
    print(f"Debug - pic_blocks shape: {pic_blocks.shape}")

    for model, num_blocks in valid_blocks_per_model.items():
        print(f"Model {model} has {num_blocks} blocks")

    obs_data = load_tif_data(OBS_PATH, years)
    if obs_data is None:
        raise ValueError("Failed to load OBS data")
    obs_mean = calculate_area_mean(obs_data)

    ant_data = []
    for model in ant_models:
        model_data = load_tif_data(ANT_PATH, years, model=model)
        if model_data is not None:
            ant_data.append(calculate_area_mean(model_data))
    ant_data = np.array(ant_data)  # (num_ant_models, 39)
    if ant_data.size == 0:
        raise ValueError("No valid ANT data loaded")

    nat_data = []
    for model in nat_models:
        model_data = load_tif_data(NAT_PATH, years, model=model)
        if model_data is not None:
            nat_data.append(calculate_area_mean(model_data))
    nat_data = np.array(nat_data)  # (num_nat_models, 39)
    if nat_data.size == 0:
        raise ValueError("No valid NAT data loaded")

    obs_base_mean = np.mean(obs_mean[base_ind])
    ant_base_mean = np.mean(ant_data[:, base_ind], axis=1)
    nat_base_mean = np.mean(nat_data[:, base_ind], axis=1)

    obs_anom = obs_mean - obs_base_mean
    ant_anom = ant_data - ant_base_mean[:, np.newaxis]
    nat_anom = nat_data - nat_base_mean[:, np.newaxis]

    print("Debug - obs_anom shape:", obs_anom.shape)
    print("Debug - obs_anom mean:", np.mean(obs_anom))
    print("Debug - ant_anom shape:", ant_anom.shape)
    print("Debug - ant_anom mean:", np.mean(ant_anom, axis=1))
    print("Debug - nat_anom shape:", nat_anom.shape)
    print("Debug - nat_anom mean:", np.mean(nat_anom, axis=1))
    print("Debug - pic_blocks mean:", np.mean(pic_blocks, axis=0))

    obs = obs_anom  # (39,)
    fp1 = np.mean(ant_anom, axis=0)  # (39,)
    fp2 = np.mean(nat_anom, axis=0)  # (39,)

    ctl = pic_blocks  # (39, total_blocks)

    print("Debug - obs before filter:", obs)
    print("Debug - fp1 before filter:", fp1)
    print("Debug - fp2 before filter:", fp2)
    print("Debug - ctl mean before filter:", np.mean(ctl, axis=0))

    window = np.ones(sigma) / sigma
    obs_ad = signal.lfilter(window, 1, obs)
    fp1_ad = signal.lfilter(window, 1, fp1)
    fp2_ad = signal.lfilter(window, 1, fp2)
    ctl_ad = np.empty_like(ctl)
    for i in range(ctl.shape[1]): 
        ctl_ad[:, i] = signal.lfilter(window, 1, ctl[:, i])

    print("Debug - obs_ad after filter:", obs_ad)
    print("Debug - fp1_ad after filter:", fp1_ad)
    print("Debug - fp2_ad after filter:", fp2_ad)
    print("Debug - ctl_ad mean after filter:", np.mean(ctl_ad, axis=0))

    obs_ad = obs_ad[sigma:]  # (39 - sigma,)
    fp1_ad = fp1_ad[sigma:]  # (39 - sigma,)
    fp2_ad = fp2_ad[sigma:]  # (39 - sigma,)
    ctl = ctl_ad[sigma:, :]  # (39 - sigma, total_blocks)

    fp = np.array([fp1_ad, fp2_ad])  # (2, 39 - sigma)
    nx = np.array([len(ant_models), len(nat_models)])

    print("Debug - Shapes in prefilt:")
    print("Year shape:", Year.shape)
    print("obs shape:", obs_ad.shape)
    print("fp shape:", fp.shape)
    print("nx:", nx)
    print("ctl shape:", ctl.shape)

    return Year[sigma:], obs_ad, fp, nx, ctl

def timedec(beta, year, ind_0):
    """
    estimate time of detection on global scale
    """
    ind_dect = np.empty((beta.shape[2]))
    tdect = np.empty((beta.shape[2]))
    ind_dect[:] = np.nan
    tdect[:] = np.nan
    for f in range(beta.shape[2]):
        temp1 = np.asarray(np.where(np.isnan(beta[:, 0, f])))
        temp2 = np.asarray(np.where(beta[:, 0, f] > beta[:, 2, f]))
        if temp1.size == 0:
            temp1 = np.empty((1, 1))
            temp1[:] = np.nan
        if temp2.size == 0:
            temp2 = np.empty((1, 1))
            temp2[:] = np.nan
        if np.isnan(temp1).all() and np.isnan(temp2).all():
            ind_remove = np.array([[0]])
        else:
            ind_remove = np.nanmax([np.nanmax(temp1), np.nanmax(temp2)]).astype(int) + np.array([[1]])
        
        ind_remove = ind_remove.squeeze()
        inferior = beta[ind_remove:, 0, f]
        betacal = beta[ind_remove:, 1, f]
        if len(inferior[np.isnan(inferior)]) == beta.shape[0]:
            ind_dect[f] = np.nan
            tdect[f] = np.nan
        else:
            t = np.roll(inferior, -1) * inferior < 0
            cross_zero = np.asarray(np.where(t[:-1]))
            if cross_zero.size == 0 and np.all(inferior[~np.isnan(inferior)] > 0):
                if np.sum(betacal) == beta.shape[0]:
                    ind_dect[f] = np.nan
                    tdect[f] = -99
                else:
                    ind_dect[f] = 1 + ind_remove
                    tdect[f] = year[ind_0 + ind_remove - 1]
            else:
                ind_dect[f] = np.max(cross_zero) + 1
                tdect[f] = year[ind_remove + ind_0 + np.max(cross_zero).astype(int) + 1]
    return ind_dect, tdect

def timeattr(beta, year, ind_dect, ind_0):
    """
    estimate time of attribution
    """
    ind_attr = np.empty((beta.shape[2]))
    tattr = np.empty((beta.shape[2]))
    ind_attr[:] = np.nan
    tattr[:] = np.nan
    for f in range(beta.shape[2]):
        superior = beta[:, 2, f]
        betacal = beta[:, 1, f]
        inferior = beta[:, 0, f]
        if np.isnan(ind_dect[f]):
            ind_attr[f] = np.nan
            tattr[f] = np.nan
        else:
            ind_attr_temp = np.array([all(t) for t in zip((superior > 1), (inferior > 0))])
            ind_attr_temp[0:int(ind_dect[f])] = False
            ind_attr_temp = np.array(np.where(ind_attr_temp))
            if ind_attr_temp.size == 0:
                ind_attr[f] = np.nan
                tattr[f] = np.nan
            else:
                ind_attr[f] = ind_attr_temp[0, 0]
                tattr[f] = year[ind_0 + ind_attr_temp[0, 0]]
    return ind_attr, tattr
