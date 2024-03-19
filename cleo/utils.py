# %% imports
import sys
import shutil
import urllib3
import certifi
import logging
import logging.config
import numpy as np
import xarray as xr
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from pathlib import Path


# %% functions
def stylish_tqdm(total, desc):
    return tqdm(
        total=total,
        desc=desc,
        bar_format='{l_bar}{bar}{r_bar}',
        ncols=80,
        file=sys.stdout,
        miniters=1,
        ascii=True,
        dynamic_ncols=True,
        leave=True,
        position=0
    )


def download_file(url, save_to, proxy=None, proxy_user=None, proxy_pass=None, overwrite=False):
    """
    downloads a file from a specified url to disk
    :param overwrite:
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


def _process_chunk(self, processing_func, chunk_size, start_x, start_y, **kwargs):
    end_x = min(start_x + chunk_size, len(self.data.coords["x"]))
    end_y = min(start_y + chunk_size, len(self.data.coords["y"]))

    data_chunks = {}
    for var_name, var in kwargs.items():
        if isinstance(var, (xr.DataArray, xr.Dataset)):
            data_chunks[var_name] = var.isel(x=slice(start_x, end_x), y=slice(start_y, end_y))

    all_args = {**data_chunks, **{k: v for k, v in kwargs.items() if k not in data_chunks}}
    processed_data = processing_func(**all_args)
    return processed_data


def compute_chunked(self, processing_func, chunk_size, **kwargs):
    """
    Process data in chunks and merge the results.

    :param self: an instance of the Atlas-class
    :param processing_func: processing_func (callable): A function that takes data chunks and properties as input and
    returns processed data.
    :type processing_func: Callable
    :param chunk_size: Size of the chunks along x and y coordinates for processing.
    :type chunk_size: int
    :param kwargs: inputs to the processing_func
    :type kwargs: dict
    :return Reassembled dataset containing the processed data.
    :rtype xarray.Dataset
    """
    reassembled_data = xr.Dataset()

    x_chunks = range(0, len(self.data.coords["x"]), chunk_size)
    y_chunks = range(0, len(self.data.coords["y"]), chunk_size)
    total_chunks = len(x_chunks) * len(y_chunks)

    with stylish_tqdm(total=total_chunks, desc="Processing chunks") as pbar:
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    _process_chunk,
                    self,
                    processing_func,
                    chunk_size,
                    start_x,
                    start_y,
                    **kwargs
                )
                for start_x in x_chunks
                for start_y in y_chunks
            ]

            for future in futures:
                processed_chunk = future.result()
                reassembled_data = xr.merge([reassembled_data, processed_chunk])
                pbar.update(1)

    # convert xr.Dataset to xr.DataArray by selecting first data variable in xr.Dataset
    reassembled_data_vars = list(reassembled_data.data_vars)
    return reassembled_data[reassembled_data_vars[0]]


def _setup_logging(self):
    """
    Setup logging in the logs directory
    :param self: an instance of the Atlas class
    """
    fname = self.path / "logs" / "logfile.log"
    colors = {
        'reset': '\33[m',
        'green': '\33[32m',
        'purple': '\33[35m',
        'orange': '\33[33m',
    }

    logging_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'default': {
                'format': '[%(asctime)s] %(levelname)-4s - %(name)-4s - %(message)s'
            },
            'color': {
                'format': f"{colors['green']}[%(asctime)s]{colors['reset']} {colors['purple']}%(levelname)-5s{colors['reset']} - {colors['orange']}%(name)-5s{colors['reset']}: %(message)s"
            }
        },
        'handlers': {
            'stream': {
                'class': 'logging.StreamHandler',
                'formatter': 'color',
            }
        },
        'root': {
            'handlers': ['stream'],
            'level': logging.INFO,
        },
    }

    if fname:
        logging_config['handlers']['file'] = {
            'class': 'logging.FileHandler',
            'formatter': 'default',
            'level': logging.DEBUG,
            'filename': fname,
        }
        logging_config['root']['handlers'].append('file')

    logging.config.dictConfig(logging_config)


def weibull_probability_density(u_power_curve, weibull_k, weibull_a):
    """
    Calculates probability density at points in u_power_curve given Weibull parameters in k and A
    :param u_power_curve: Power curve wind speeds
    :param weibull_k: k-parameter of the Weibull distribution of wind speed
    :param weibull_a: A-parameter of the Weibull distribution of wind speed
    :return:
    """
    uar = np.asarray(u_power_curve)
    prb = [(weibull_k / weibull_a * (z / weibull_a) ** (weibull_k - 1)) * (np.exp(-(z / weibull_a) ** weibull_k)) for z
           in uar]
    pdf = xr.concat(prb, dim='wind_speed')
    pdf = pdf.assign_coords({'wind_speed': u_power_curve})
    pdf = pdf.squeeze().rename("weibull_probability_density")
    return pdf


def capacity_factor(weibull_pdf, terrain_roughness_length, u_power_curve, p_power_curve, h_turbine, h_reference=100,
                    correction_factor=1):
    """
    calculates wind turbine capacity factors given Weibull probability density pdf, roughness factor alpha, wind turbine
    power curve data in u_power_curve and p_power_curve, turbine height h_turbine and reference height of wind speed
    modelling h_reference
    :param correction_factor:
    :param weibull_pdf: probability density function from weibull_probability_density()
    :param terrain_roughness_length: terrain roughness length
    :param u_power_curve: power curve wind speed
    :param p_power_curve: power curve output
    :param h_turbine: hub height of wind turbine in m
    :param h_reference: reference height at which weibull pdf is computed
    :return:
    """
    power_curve = xr.DataArray(data=p_power_curve, coords={'wind_speed': u_power_curve})
    u_adjusted = xr.DataArray(data=u_power_curve, coords={'wind_speed': u_power_curve}) @ (
            h_turbine / h_reference) ** terrain_roughness_length
    cap_factor_values = np.trapz(weibull_pdf * power_curve, u_adjusted, axis=0)
    cap_factor = terrain_roughness_length.copy()
    cap_factor.values = cap_factor_values * correction_factor
    cap_factor.name = "capacity_factor"
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
    spec_power = power * 10 ** 6 / rotor_area
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


def levelized_cost(power, capacity_factors, overnight_cost, grid_cost, om_fixed, om_variable, discount_rate, lifetime,
                   hours_per_year=8766, per_mwh=True):
    """
    Calculates wind turbines' levelized cost of electricity in EUR per MWh
    :param per_mwh: Returns LCOE in currency per megawatt hour if true (default). Else returns LCOE in currency per kwh.
    :type per_mwh: bool
    :param hours_per_year: Number of hours per year. Default is 8766 to account for leap years.
    :param power: rated power in kW
    :type power: float
    :param capacity_factors: wind turbine capacity factor (share of year)
    :type capacity_factors: xarray.DataArray
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
    npv_electricity = capacity_factors * hours_per_year * power * npv_factor

    # calculate net present value of cost
    npv_cost = (om_variable * capacity_factors * hours_per_year + om_fixed) * power * npv_factor
    npv_cost = npv_cost + overnight_cost + grid_cost

    lcoe = npv_cost / npv_electricity

    if per_mwh:
        return lcoe * 1000
    else:
        return lcoe
