import ebf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import pickle
from galpy import potential
from galpy.potential.mwpotentials import MWPotential2014
from galpy.orbit import Orbit
from galpy.util.conversion import get_physical
from astropy import units as u
from kick import NatalKick
from mass import Mass

def calculate_lifetimes(df):
    '''
    Function to calculate the lifetime of each star.

    Adapted from code provided by Sanjib Sharma.
    '''
    with open('../stellar_lifetime.pkl', 'rb') as handle:
        stellar_lifetimes = pickle.load(handle)
    stellar_properties = np.array([np.clip(df['feh'], a_min=-2, a_max=0.49), df['smass']]).T
    df['lifetime'] = 10**stellar_lifetimes(stellar_properties) / 10**9
    return df

def load_data(filename, filter=True):
    '''Load Galaxia data into DataFrame'''
    data = ebf.read(filename, '/')
    centre = np.array(data['center'])
    keys = ['px', 'py', 'pz', 'vx', 'vy', 'vz', 'age', 'smass', 'feh', 'popid']
    useful_data = []
    for key in keys:
        useful_data += [data[key]]
    useful_data = np.array(useful_data).T
    df = pd.DataFrame(useful_data, columns=keys)

    # Convert age to gigayears
    df['age'] = 10**df['age'] / 10**9

    df = calculate_lifetimes(df)

    # Make data centred on galactic centre
    df.loc[:, ['px', 'py', 'pz', 'vx', 'vy', 'vz']] += centre

    # Add rtype column which specifies remnant type
    rtype = []
    for mass in df['smass']:
        if mass > 25:
            rtype.append('Black Hole')
        elif mass > 8:
            rtype.append('Neutron Star')
        else:
            rtype.append('White Dwarf')
    df['rtype'] = rtype

    if filter:
        df = df[df['smass'] > 8]
    return df

def add_masses(df, masses):
    '''Add mass to each entry of the DataFrame'''
    for i in df.index.values:
        df.loc[i, 'mass'] = masses.get_mass(df.loc[i, 'rtype'])
    return df

def add_kicks(df, natal_kicks, verbose=0):
    '''Add kick to each entry of the DataFrame'''
    for prog, i in enumerate(df.index.values):
        if verbose and prog % (df.shape[0]//100) == 0:
            print(f'Creating kicks, progress = {100 * prog / df.shape[0]:.0f}%')

        vx, vy, vz = natal_kicks.get_kick(df.loc[i, 'rtype'], df.loc[i, 'mass'], df.loc[i, 'smass'])

        df.loc[i, ['vx', 'vy', 'vz']] += np.array([vx, vy, vz])

    return df

def update_cylindrical_coords(df):
    '''Updates the cylindrical coordinates based on values of the cartesian coordinates'''
    x = df['px']
    y = df['py']
    vx = df['vx']
    vy = df['vy']
    df['R'] = np.sqrt(x**2 + y**2)
    df['phi'] = np.arctan2(y, x)
    df['vR'] = (x*vx + y*vy)/df['R']
    df['vphi'] = (x*vy - y*vx)/df['R']**2
    df['vT'] = df['R']*df['vphi']
    return df

def update_cartestian_coordinates(df):
    '''Updates the cartestian coordinates based on values of the cylindrical coordinates'''
    R = df['R']
    phi = df['phi']
    vR = df['vR']
    vT = df['vT']
    df['vphi'] = vT/R
    vphi = df['vphi']
    df['px'] = R * np.cos(phi)
    df['py'] = R * np.sin(phi)
    df['vx'] = vR*np.cos(phi) - R*vphi*np.sin(phi)
    df['vy'] = vR*np.sin(phi) + R*vphi*np.cos(phi)
    return df

def get_final_locations(df, timesteps):
    '''Retrieves the final location of the remnant from galpy output'''
    ages = np.array(df['age'] - df['lifetime']).reshape(-1, 1)
    timesteps = timesteps.reshape(1, -1)
    timesteps = np.repeat(timesteps, ages.shape[0], axis=0)

    # Find the argument where the remnant should have finished orbiting
    final_args = np.argmin(np.abs(timesteps.to(u.Gyr).value - ages), axis=1)
    return final_args

def calculate_orbits(df, duration=None):
    """
    Calculates the orbit of each remnant in the provided DataFrame.

    By default the orbits are calculated for a duration based on the age of the
    star but with the lifetime of the original star subtracted (so they are
    evolved for the duration of the remnants life). If a duration is specified
    then all remnants are evolved for this period of time (value is assumed to
    be in Gyr).

    Parameters
    ---------
    df : DataFrame
        pandas DataFrame containing the information on the remnants being evolved
    duration : int (optional)
        The duration (in Gyr) for which to calculate orbits

    Returns
    ----------
    DataFrame
        pandas DataFrame containing the updated remnants with their evolved
        positions
    """
    ro, vo = 8.0, 232.0
    remnant_starts = np.array(df[['R', 'vR', 'vT', 'pz', 'vz', 'phi']])
    conversion_to_natural_units = np.array([ro, vo, vo, ro, vo, 1])
    remnant_starts /= conversion_to_natural_units
    # units = [u.kpc, u.km/u.s, u.km/u.s, u.kpc, u.km/u.s, u.radian]

    if duration is not None:
        if duration < 1e-3:
            # Ensure that for small durations there are at least 10 steps
            step = duration / 10
            timesteps = np.arange(0, duration + step, step)*u.Gyr
        else:
            # Else the step size is set to 100,000 years
            timesteps = np.arange(0, duration + 1e-4, 1e-4)*u.Gyr
    else:
        # Step size is set to 1 million years
        timesteps = np.arange(0, df['age'].max() + 1e-3, 1e-3)*u.Gyr
    orbit_values = []
    step = 512
    complete = 0

    for i in range(0, remnant_starts.shape[0], step):
        if i >= complete / 100 * remnant_starts.shape[0]:
            complete = (100 * i) // (remnant_starts.shape[0])
            print(f'Orbits are {complete}% complete.')
            complete += 1
        # new_orbits = Orbit(remnant_starts[i:min(i+step, remnant_starts.shape[0])], ro=8.0*u.kpc, vo=232.0 * u.km/u.s)
        new_orbits = Orbit(remnant_starts[i:min(i+step, remnant_starts.shape[0])])
        new_orbits.integrate(timesteps, MWPotential2014, method='symplec6_c')
        orbit_values.append(new_orbits.getOrbit())

    print('Finished calculating orbits')
    orbit_values = np.vstack(orbit_values)
    orbit_values *= conversion_to_natural_units
    print('Finished vstacking')
    
    if duration is not None:
        df[['R', 'vR', 'vT', 'pz', 'vz', 'phi']] = orbit_values[:, -1]
    else:
        final_orbits = (np.arange(df.shape[0]), get_final_locations(df, timesteps))
        print(orbit_values.shape, 'Max age:', df['age'].max())
        df[['R', 'vR', 'vT', 'pz', 'vz', 'phi']] = orbit_values[final_orbits]
    return update_cartestian_coordinates(df)

if __name__ == '__main__':
    natal_kicks = NatalKick(distributions={'Black Hole': 'scaled igoshev young', 
                                           'Neutron Star': 'igoshev young'})
    masses = Mass(distributions={'Black Hole': 7.8, 'Neutron Star': 1.35})
    
    # extinct_filename = r'galaxia_f1e-4_bhm2.35.ebf'
    extinct_filename = r'../galaxia_f1e-3_bhm2.35.ebf' # Folder location is taken care of in the loading of this file
    output_filename = f'../kicked_remnants_{distribution}_7.8_DC_{bh_kicks}'
    np.random.seed(0)
    
    
    df = load_data(extinct_filename)
    df = add_masses(df, masses)
    df['will_escape'] = np.sqrt(np.sum(df[['vx', 'vy', 'vz']]**2, axis=1)) >= potential.vesc(MWPotential2014, np.sqrt(np.sum(df[['px', 'py', 'pz']]**2, axis=1))/8.0)*232
    df = add_kicks(df, natal_kicks=natal_kicks, verbose=1)
    df = update_cylindrical_coords(df)
    df['will_escape'] = np.sqrt(np.sum(df[['vx', 'vy', 'vz']]**2, axis=1)) >= potential.vesc(MWPotential2014, np.sqrt(np.sum(df[['px', 'py', 'pz']]**2, axis=1))/8.0)*232
    df.to_csv(f'{output_filename}.csv', index=False)

    # # Data must be split into sections to cope with memory restrictions
    # sections = 10 # Minimum 3 for full run
    # section_length = len(df)//sections + 1
    # section_dfs = [df.iloc[i*section_length:(i+1)*section_length].copy() for i in range(sections)]
    # for i in range(sections):
    #     print('*'*20)
    #     print(f'Dataframe {i+1}/{sections}')
    #     section_dfs[i].loc[:, 'velocity'] = np.sqrt(np.sum(section_dfs[i].loc[:, ['vx', 'vy', 'vz']]**2, axis=1))
    #     section_dfs[i] = calculate_orbits(section_dfs[i])
    #     section_dfs[i].loc[:, 'will_escape'] = np.sqrt(np.sum(section_dfs[i].loc[:, ['vx', 'vy', 'vz']]**2, axis=1)) >= potential.vesc(MWPotential2014,
    #                                                                                                                             np.sqrt(np.sum(section_dfs[i].loc[:, ['px', 'py', 'pz']]**2, axis=1))/8.0)*232
    # df = pd.concat(section_dfs)
        
    # print('Total sources:', len(df))
    
    # df.to_csv(f'{output_filename}_integrated.csv', index=False)

    # ### Evolve data for 200 year intervals
    # df = pd.read_csv(f'../kicked_remnants_{DISTRIBUTION}_7.8_DC_integrated_final_ECSN.csv')
    # for j in range(200, 20001, 200):
    #     # If data is sufficiently large it is split into sections to cope with memory restrictions
    #     sections = 1
    #     section_length = len(df)//sections + 1
    #     section_dfs = [df.iloc[i*section_length:(i+1)*section_length].copy() for i in range(sections)]
    #     for i in range(sections):
    #         print('*'*20)
    #         print(f'Dataframe {i+1}/{sections}')
    #         section_dfs[i].loc[:, 'velocity'] = np.sqrt(np.sum(section_dfs[i].loc[:, ['vx', 'vy', 'vz']]**2, axis=1))
    #         section_dfs[i] = calculate_orbits(section_dfs[i], duration=200*1e-9)
    #         section_dfs[i].loc[:, 'will_escape'] = np.sqrt(np.sum(section_dfs[i].loc[:, ['vx', 'vy', 'vz']]**2, axis=1)) >= potential.vesc(MWPotential2014,
    #                                                                                                                                  np.sqrt(np.sum(section_dfs[i].loc[:, ['px', 'py', 'pz']]**2, axis=1))/8.0)*232
    #     df = pd.concat(section_dfs)


        # df.to_csv(f'../kicked_remnants_{DISTRIBUTION}_7.8_DC_integrated_final_ECSN+{j}.csv', index=False)
