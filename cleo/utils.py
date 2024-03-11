# %% imports
import shutil
import xarray as xr
import numpy as np
import pandas as pd
import urllib3
import certifi
from pathlib import Path


# %% functions
def get_coords(x, y, time, dx=0.25, dy=0.25, dt="h", **kwargs):
    """
    Create an cutout coordinate system on the basis of slices and step sizes.

    Parameters
    ----------
    x : slice
        Numerical slices with lower and upper bound of the x dimension.
    y : slice
        Numerical slices with lower and upper bound of the y dimension.
    time : slice
        Slice with strings with lower and upper bound of the time dimension.
    dx : float, optional
        Step size of the x coordinate. The default is 0.25.
    dy : float, optional
        Step size of the y coordinate. The default is 0.25.
    dt : str, optional
        Frequency of the time coordinate. The default is 'h'. Valid are all
        pandas offset aliases.

    Returns
    -------
    ds : xarray.Dataset
        Dataset with x, y and time variables, representing the whole coordinate
        system.
    """
    x = slice(*sorted([x.start, x.stop]))
    y = slice(*sorted([y.start, y.stop]))

    ds = xr.Dataset(
        {
            "x": np.round(np.arange(-180, 180, dx), 9),
            "y": np.round(np.arange(-90, 90, dy), 9),
            "time": pd.date_range(start="1940", end="now", freq=dt),
        }
    )
    ds = ds.assign_coords(lon=ds.coords["x"], lat=ds.coords["y"])
    ds = ds.sel(x=x, y=y, time=time)
    return ds


def download_file(url, save_to, proxy=None, proxy_user=None, proxy_pass=None, overwrite=False):
    """
    downloads a file from a specified url to disk
    :param proxy_pass: proxy password
    :param proxy_user: proxy username
    :param proxy: proxy url:port
    :param url: url-string
    :param save_to: destination file name (string)
    :return:
    """
    dld = False
    if (not Path(save_to).is_file()) or (overwrite is True):
        if proxy is not None:
            default_headers = urllib3.make_headers(proxy_basic_auth=f'{proxy_user}:{proxy_pass}')
            http = urllib3.ProxyManager(proxy, proxy_headers=default_headers, ca_certs=certifi.where())
        else:
            http = urllib3.PoolManager(ca_certs=certifi.where())
        try:
            with http.request('GET', url.replace('"', '').replace(' ', ''),
                              preload_content=False) as r, open(save_to, 'wb') as out_file:
                shutil.copyfileobj(r, out_file)
        except:
            try:
                http = urllib3.PoolManager(cert_reqs='CERT_NONE')
                with http.request('GET', url.replace('"', '').replace(' ', ''),
                                  preload_content=False) as r, open(save_to, 'wb') as out_file:
                    shutil.copyfileobj(r, out_file)
            except:
                raise Exception
        dld = True
    return dld


def weibull_probability_density(u_power_curve, k, A):
    """
    Calculates probability density at points in u_power_curve given Weibull parameters in k and A
    :param u_power_curve:
    :param k:
    :param A:
    :return:
    """
    uar = np.asarray(u_power_curve)
    prb = [(k / A * (z / A) ** (k - 1)) * (np.exp(-(z / A) ** k)) for z in uar]
    pdf = xr.concat(prb, dim='wind_speed')
    pdf = pdf.assign_coords({'wind_speed': u_power_curve})
    pdf = pdf.squeeze()
    return pdf


def capacity_factor(pdf, alpha, u_power_curve, p_power_curve, h_turbine, h_reference=100, correction_factor=1):
    """
    calculates wind turbine capacity factors given Weibull probability density pdf, roughness factor alpha, wind turbine
    power curve data in u_power_curve and p_power_curve, turbine height h_turbine and reference height of wind speed
    modelling h_reference
    :param pdf: probability density function from weibull_probability_density()
    :param alpha: roughness coefficient
    :param u_power_curve:
    :param p_power_curve:
    :param h_turbine:
    :param h_reference:
    :return:
    """
    power_curve = xr.DataArray(data=p_power_curve, coords={'wind_speed': u_power_curve})
    u_adjusted = xr.DataArray(data=u_power_curve, coords={'wind_speed': u_power_curve}) @ (
                h_turbine / h_reference) ** alpha
    cap_factor_values = np.trapz(pdf * power_curve, u_adjusted, axis=0)
    cap_factor = alpha.copy()
    cap_factor.values = cap_factor_values * correction_factor
    return cap_factor


def discount_factor(discount_rate, period):
    """
    Calculate the discount factor for a given discount rate and period.
    :param discount_rate: discount rate (fraction of 1)
    :type discount_rate: float
    :param period: Number of years
    :type period: int
    :return: Discount factor
    :rtype: float
    """
    dcf_numerator = 1 - (1 + discount_rate) ** (-period)
    dcf_denominator = 1 - (1 + discount_rate) ** (-1)
    dcf = dcf_numerator / dcf_denominator
    return dcf


def turbine_overnight_cost(power, hub_height, rotor_diameter, year):
    """
    calculates wind turbine investment cost in EUR per MW based on >>Rinne et al. (2018): Effects of turbine technology
    and land use on wind power resource potential, Nature Energy<<
    :param power: rated power in MW
    :param hub_height: hub height in meters
    :param rotor_diameter: rotor diameter in meters
    :return: overnight investment cost in EUR per kW
    """
    rotor_area = np.pi * (rotor_diameter / 2) ** 2
    spec_power = power * 10**6 / rotor_area
    cost = ((620 * np.log(hub_height)) - (1.68 * spec_power) + (182 * (2016 - year) ** 0.5) - 1005)
    return cost.astype('float')


def grid_connect_cost(power):
    """
    Calculates grid connection cost according to §54 (3,4) ElWOG https://www.ris.bka.gv.at/GeltendeFassung.wxe?Abfrage=Bundesnormen&Gesetzesnummer=20007045
    :param power: power in kW
    :return:
    """
    cost = 50 * power
    return cost


def levelized_cost(power, utilisation, overnight_cost, grid_cost, om_fixed, om_variable, discount_rate, lifetime,
                   hours_per_year=8766, per_mwh=True):
    """
    Calculates wind turbines' levelized cost of electricity in EUR per MWh
    :param per_mwh: Returns LCOE in currency per megawatt hour if true (default). Else returns LCOE in currency per kwh.
    :type per_mwh: bool
    :param hours_per_year: Number of hours per year. Default is 8766 to account for leap years.
    :param power: rated power in kW
    :type power: float
    :param utilisation: wind turbine capacity factor (share of year)
    :type utilisation: xarray.DataArray
    :param overnight_cost: in EUR/MW
    :type overnight_cost: float
    :param grid_cost: cost for connecting to the electricity grid
    :type grid_cost: xarray.DataArray
    :param om_fixed: EUR/kW
    :type om_fixed: float
    :param om_variable: EUR/kWh
    :type om_variable: float
    :param discount_rate: percent
    :type discount_rate: float
    :param lifetime: years
    :type lifetime: int
    :return: lcoe in EUR/kWh
    """
    npv_factor = discount_factor(discount_rate, lifetime)

    # calculate net present amount of electricity generated over lifetime
    npv_electricity = utilisation * hours_per_year * power * npv_factor

    # calculate net present value of cost
    npv_cost = (om_variable * utilisation * hours_per_year + om_fixed) * power * npv_factor
    npv_cost = npv_cost + overnight_cost + grid_cost

    lcoe = npv_cost / npv_electricity

    if per_mwh:
        return lcoe * 1000
    else:
        return lcoe
