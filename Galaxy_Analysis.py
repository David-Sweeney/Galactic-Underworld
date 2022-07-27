import ebf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd
from kick_utils import kick_distribution, get_kick_distribution_uncertainties
from scipy.stats.kde import gaussian_kde
from scipy.optimize import curve_fit
from sklearn.neighbors import KernelDensity
import copy

NUMBER_OF_STARS = None

def load_data(filename, min_mass=0, max_mass=np.inf, distance=np.inf, number_of_stars=None, df=False, verbose=0):
    if filename.endswith('.csv'):
        data = pd.read_csv(filename)
        if df:
            return data, len(data)
        if verbose:
            print(f'Number of stars: {data.shape[0]}, maximum mass: {data["smass"].max():.2f}')
        coords = np.array(data[(data['smass'] > min_mass) & (data['smass'] < max_mass)][['px', 'py', 'pz']])
        centre = np.array([-8., 0., 0.015])
    else:
        data = ebf.read(filename, '/')
        coords, smass = [data['px'], data['py'], data['pz']], data['smass']
        centre = data['center']
        if verbose:
            print(f'Number of stars: {len(coords[0])}, maximum mass: {max(smass):.2f}')
        coords = np.array(coords).T[(smass > min_mass) & (smass < max_mass)] + np.array(centre[:3])

    if number_of_stars is not None:
        if verbose:
            print(f'Reducing number of stars to {number_of_stars}')
        if number_of_stars > len(coords):
            raise ValueError(f'Cannot reduce to {number_of_stars} because there are only {len(coords)}')
        np.random.seed(1)
        coords = coords[np.random.choice(range(len(coords)), size=number_of_stars, replace=False)]
    else:
        found_stars = len(coords)

    if verbose:
        print(f'Removing stars not within {distance}kpc.')
    coords = coords[(np.abs(coords[:,0])<distance) & (np.abs(coords[:,1])<distance) & (np.abs(coords[:,2])<distance), :]

    if verbose:
        print(f'Number of stars within {distance}kpc with {min_mass}M < m < {max_mass}M: {len(coords)}')

    if number_of_stars is None:
        return coords, centre[:3], found_stars
    return coords, centre[:3]

def get_galactic_density(kde, x_lims, y_lims, points=None, x_points=400, y_points=400):
    if points is not None:
        x_points = y_points = points
    x = np.linspace(*x_lims, x_points)
    y = np.linspace(*y_lims, y_points)
    xx, yy = np.meshgrid(x, y)
    zz = kde(np.append(xx.reshape(-1, 1), yy.reshape(-1, 1), axis=1).T)
    zz = zz.reshape(y_points, x_points)
    return zz

def get_galactic_density_sk(kde, x_lims, y_lims, points=None, x_points=400, y_points=400):
    if points is not None:
        x_points = y_points = points
    x = np.linspace(*x_lims, x_points)
    y = np.linspace(*y_lims, y_points)
    xx, yy = np.meshgrid(x, y)
    zz = np.exp(kde.score_samples(np.append(xx.reshape(-1, 1), yy.reshape(-1, 1), axis=1)))
    zz = zz.reshape(y_points, x_points)
    return zz

def get_radial_bins(min_r, max_r, num_bins):
    '''Calculate num_bins equal bins by area between min_r and max_r'''
    area = np.pi * (max_r**2 - min_r**2)
    area_per_bin = area/num_bins
    bins = [min_r]
    outer_radius = min_r
    for _ in range(num_bins):
        bins.append(np.sqrt(area_per_bin/np.pi + bins[-1]**2))

    return bins

def get_radial_bin_areas(bin_edges):
    areas = []
    for min_r, max_r in zip(bin_edges[:-1], bin_edges[1:]):
        areas.append(np.pi * (max_r**2 - min_r**2))
    return np.array(areas)

def plot_data(style='split', args=[], subplot=0):
    if subplot:
        plt.subplot(subplot)
    else:
        plt.figure()

    filename = 'Galactic_Underworld'

    # extinct_filename = r'../galaxia_f1e-4_bhm2.35.ebf'
    extinct_filename = r'../galaxia_f1e-3_bhm2.35.ebf'
    # if kicked_filename is None: kicked_filename= r'../kicked_remnants.csv'
    stellar_filename = r'../milkyway_f1e-6.ebf'

    distance = np.inf
    # if kicked:
    # extinct_coords, centre, number_of_stars = load_data(kicked_filename, min_mass=8, distance=distance)
    # else:
    #     extinct_coords, centre, number_of_stars = load_data(extinct_filename, min_mass=8, distance=distance)

    stellar_coords, centre, number_of_stars = load_data(stellar_filename, min_mass=0, distance=distance)
    # stellar_coords, _, number_of_stars = load_data(stellar_filename, min_mass=0, number_of_stars=number_of_stars, distance=distance)

    if style == 'split':
        filename += '_split'
        df_extinct = pd.read_csv(kicked_filename)
        plt.scatter(stellar_coords[:, 0][stellar_coords[:, 0] > 0], stellar_coords[:, 2][stellar_coords[:, 0] > 0], s=2, alpha=0.1, label='Visible Galaxy')
        if 'escape' in args:
            filename += '_escape'
            args.remove('escape')
            df_extinct_escaping = df_extinct[df_extinct['will_escape'] == True]
            df_extinct_not_escaping = df_extinct[df_extinct['will_escape'] == False]
            plt.scatter(df_extinct_escaping['px'][df_extinct_escaping['px'] < 0], df_extinct_escaping['pz'][df_extinct_escaping['px'] < 0], s=2, c='tab:red', alpha=0.1, label='GUW ($m_i >$ 8M$_\odot$), $v \geq v_{esc}$')
            plt.scatter(df_extinct_not_escaping['px'][df_extinct_not_escaping['px'] < 0], df_extinct_not_escaping['pz'][df_extinct_not_escaping['px'] < 0], s=2, c='black', alpha=0.1, label='GUW ($m_i >$ 8M$_\odot$), $v < v_{esc}$')
            if 'changed' in args:
                filename += '_changed'
                args.remove('changed')
                df_extinct_changed_now = df_extinct_escaping[df_extinct_escaping['escaping_same'] == False]
                df_extinct_changed_not = df_extinct_not_escaping[df_extinct_not_escaping['escaping_same'] == False]
                plt.scatter(df_extinct_changed_now['px'][df_extinct_changed_now['px'] < 0], df_extinct_changed_now['pz'][df_extinct_changed_now['px'] < 0], s=4, c='fuchsia', alpha=1, label='Galactic Underworld ($m_i >$ 8M$_\odot$), Now Escaping')
                plt.scatter(df_extinct_changed_not['px'][df_extinct_changed_not['px'] < 0], df_extinct_changed_not['pz'][df_extinct_changed_not['px'] < 0], s=4, c='purple', alpha=1, label='Galactic Underworld ($m_i >$ 8M$_\odot$), Now Not Escaping')
        elif 'type' in args:
            filename += '_type'
            args.remove('type')
            df_black_holes = df_extinct[df_extinct['rtype'] == 'Black Hole']
            df_black_holes = df_black_holes[(df_black_holes['px'] < 0) & (df_black_holes['pz'] > 0)]
            df_neutron_stars = df_extinct[df_extinct['rtype'] == 'Neutron Star']
            df_neutron_stars = df_neutron_stars[(df_neutron_stars['px'] < 0) & (df_neutron_stars['pz'] < 0)]
            plt.scatter(df_black_holes['px'], df_black_holes['pz'], s=2, c='black', alpha=0.1, label='Black Hole ($m_i >$ 25M$_\odot$)')
            plt.scatter(df_neutron_stars['px'], df_neutron_stars['pz'], s=2, c='tab:orange', alpha=0.1, label='Neutron Star (8M$_\odot$ < $m_i <$ 25M$_\odot$)')
        else:
            plt.scatter(df_extinct['px'][df_extinct['px'] < 0], df_extinct['pz'][df_extinct['px'] < 0], s=2, c='black', alpha=0.1, label='Galactic Underworld ($m_i >$ 8M$_\odot$)')

        plt.xlabel('$p_x$ (kpc)'); plt.ylabel('$p_z$ (kpc)')
        plt.axis('equal')
        integer_args = [i for i in args if type(i) == int]
        if len(integer_args) > 0:
            args.remove(integer_args[0])
            sizes = [5, 10, 20, 50, 100, 200, 500]
            x_lim = min(sizes, key=lambda x:abs(x-integer_args[0]))
            filename += f'_{int(x_lim)}'
            y_lim = 0.6*x_lim
            plt.gca().set(xlim = ([-x_lim, x_lim]), ylim = ([-y_lim, y_lim]))

        plt.title('Galactic Underworld')
        leg = plt.legend(loc='lower right')
    elif style == 'contour':
        filename += '_contour'

        vmax = 3e-2

        if 'renzo' in args:
            filename += '_renzo'
            args.remove('renzo')
            df_extinct = pd.read_csv(renzo_filename)
        elif 'unkicked' in args:
            filename += '_unkicked'
            args.remove('unkicked')
            df_extinct = pd.read_csv(unkicked_filename)
        else:
            df_extinct = pd.read_csv(kicked_filename)


        top_left_text = left_text = right_text = ''

        integer_args = [i for i in args if type(i) == int]
        if len(integer_args) > 0:
            args.remove(integer_args[0])
            sizes = [5, 10, 20, 50, 100, 200, 500]
            x_lim = min(sizes, key=lambda x:abs(x-integer_args[0]))
            y_lim = 0.6*x_lim
        else:
            x_lim = y_lim = 100

        float_args = [i for i in args if type(i) == float]
        if len(float_args) > 0:
            args.remove(float_args[0])
            filename += f'_{float_args[0]:.1f}'
            bandwidth = float_args[0]
        else:
            bandwidth = 0.7

        kde = KernelDensity(bandwidth = bandwidth, kernel = 'gaussian')

        if 'type' in args:
            # Make a contour plot of black holes vs neutron stars
            filename += '_type'
            args.remove('type')

            if 'thin_disk' in args:
                filename += '_thin_disk'
                args.remove('thin_disk')

                df_extinct = df_extinct[df_extinct['popid'] <= 6]
            elif 'thick_disk' in args:
                filename += '_thick_disk'
                args.remove('thick_disk')

                df_extinct = df_extinct[df_extinct['popid'] == 7]
            elif 'halo' in args:
                filename += '_halo'
                args.remove('halo')

                df_extinct = df_extinct[df_extinct['popid'] == 8]
            elif 'bulge' in args:
                filename += '_bulge'
                args.remove('bulge')

                df_extinct = df_extinct[df_extinct['popid'] == 9]

            df_black_holes = df_extinct[df_extinct['rtype'] == 'Black Hole']
            df_neutron_stars = df_extinct[df_extinct['rtype'] == 'Neutron Star']

            # black_hole_kde = gaussian_kde(df_black_holes[['px', 'pz']].to_numpy().T,
            #                                 bw_method=bandwidth)
            # neutron_star_kde = gaussian_kde(df_neutron_stars[['px', 'pz']].to_numpy().T,
            #                                 bw_method=bandwidth)

            # black_hole_z = get_galactic_density(black_hole_kde, [-x_lim, 0], [-y_lim, y_lim])
            # neutron_star_z = get_galactic_density(neutron_star_kde, [0, x_lim], [-y_lim, y_lim])
            kde.fit(df_black_holes[['px', 'pz']].to_numpy())
            black_hole_z = get_galactic_density_sk(kde, [-x_lim, 0], [-y_lim, y_lim])
            kde.fit(df_neutron_stars[['px', 'pz']].to_numpy())
            neutron_star_z = get_galactic_density_sk(kde, [0, x_lim], [-y_lim, y_lim])
            left_text = 'Black Holes'
            right_text = 'Neutron Stars'

            image = np.hstack((black_hole_z, neutron_star_z))
        elif 'split_type' in args:
            # Make a contour plot of black holes vs neutron stars vs stars
            filename += '_split_type'
            args.remove('split_type')

            df_black_holes = df_extinct[df_extinct['rtype'] == 'Black Hole']
            df_neutron_stars = df_extinct[df_extinct['rtype'] == 'Neutron Star']

            # stellar_kde = gaussian_kde(stellar_coords[:, [0, 2]].T, bw_method=bandwidth)
            # black_hole_kde = gaussian_kde(df_black_holes[['px', 'pz']].to_numpy().T,
            #                                 bw_method=bandwidth)
            # neutron_star_kde = gaussian_kde(df_neutron_stars[['px', 'pz']].to_numpy().T,
            #                                 bw_method=bandwidth)

            points = 400
            # stellar_z = get_galactic_density(stellar_kde, [0, x_lim], [-y_lim, y_lim], points=points)
            # black_hole_z = get_galactic_density(black_hole_kde, [-x_lim, 0],
            #                                     [-y_lim, 0], x_points=points, y_points=points//2)
            # neutron_star_z = get_galactic_density(neutron_star_kde, [-x_lim, 0],
            #                                         [0, y_lim], x_points=points, y_points=points//2)

            kde.fit(stellar_coords[:, [0, 2]].to_numpy())
            stellar_z = get_galactic_density_sk(kde, [0, x_lim], [-y_lim, y_lim], points=points)
            kde.fit(df_black_holes[['px', 'pz']].to_numpy())
            black_hole_z = get_galactic_density_sk(kde, [-x_lim, 0],
                                                [-y_lim, 0], x_points=points, y_points=points//2)
            kde.fit(df_neutron_stars[['px', 'pz']].to_numpy())
            neutron_star_z = get_galactic_density_sk(kde, [-x_lim, 0],
                                                    [0, y_lim], x_points=points, y_points=points//2)

            left_text = 'Black Holes'
            top_left_text = 'Neutron Stars'
            right_text = 'Visible Stars'

            image = np.hstack((np.vstack((black_hole_z, neutron_star_z)), stellar_z))
        elif 'visible' in args:
            # Make a contour plot of black holes vs neutron stars vs stars
            filename += '_visible'
            args.remove('visible')

            # stellar_kde = gaussian_kde(stellar_coords[:, [0, 2]].T, bw_method=bandwidth)
            # image = get_galactic_density(stellar_kde, [-x_lim, x_lim], [-y_lim, y_lim], x_points=800)

            kde.fit(stellar_coords[:, [0, 2]])
            image = get_galactic_density_sk(kde, [-x_lim, x_lim], [-y_lim, y_lim], x_points=800)

            vmax = 3e-2
        else:
            # Make a contour plot of remnants vs alive galaxy
            # stellar_kde = gaussian_kde(stellar_coords[:, [0, 2]].T, bw_method=bandwidth)
            # stellar_z = get_galactic_density(stellar_kde, [0, x_lim], [-y_lim, y_lim])

            kde.fit(stellar_coords[:, [0, 2]])
            stellar_z = get_galactic_density_sk(kde, [0, x_lim], [-y_lim, y_lim])

            if 'escape' in args:
                filename += '_escape'
                args.remove('escape')

                df_extinct_not_escaping = df_extinct[df_extinct['will_escape'] == False][['px', 'pz']].to_numpy()
                # extinct_kde = gaussian_kde(df_extinct_not_escaping.T, bw_method=bandwidth)
                kde.fit(df_extinct_not_escaping)
                vmax = 5e-2
            else:
                # extinct_kde = gaussian_kde(df_extinct[['px', 'pz']].to_numpy().T, bw_method=bandwidth)
                kde.fit(df_extinct[['px', 'pz']].to_numpy())
                vmax = 5e-2

            # extinct_z = get_galactic_density(extinct_kde, [-x_lim, 0], [-y_lim, y_lim])
            extinct_z = get_galactic_density_sk(kde, [-x_lim, 0], [-y_lim, y_lim])
            left_text = 'Galactic Underworld'
            right_text = 'Visible Stars'
            image = np.hstack((extinct_z, stellar_z))

        filename += f'_{int(x_lim)}'

        inferno = copy.copy(matplotlib.cm.get_cmap('inferno'))
        inferno.set_bad('k')
        im = plt.imshow(image, extent=[-x_lim, x_lim, -y_lim, y_lim], origin='lower',
                        cmap=inferno, norm=colors.LogNorm(vmin=max(5e-5, image.min()),
                        # vmax=min(vmax, image.max())))
                        vmax=vmax))
        plt.text(-x_lim/2, -y_lim * 9/10, s=left_text, ha='center', va='baseline',
                bbox={'facecolor':'white', 'alpha':0.5, 'edgecolor':'none', 'pad':1})
        plt.text(x_lim/2, -y_lim * 9/10, s=right_text, ha='center', va='baseline',
                bbox={'facecolor':'white', 'alpha':0.5, 'edgecolor':'none', 'pad':1})
        plt.text(-x_lim/2, y_lim * 9/10, s=top_left_text, ha='center', va='top',
                bbox={'facecolor':'white', 'alpha':0.5, 'edgecolor':'none', 'pad':1})

        if 'lines' in args:
            # plot contour lines from visible galaxy
            filename += '_lines'
            args.remove('lines')

            # stellar_kde = gaussian_kde(stellar_coords[:, [0, 2]].T, bw_method=bandwidth)
            # stellar_z = get_galactic_density(stellar_kde, [-x_lim, x_lim], [-y_lim, y_lim], x_points=800)
            # kde.fit(stellar_coords[:, [0, 2]])
            # stellar_z = get_galactic_density_sk(kde, [-x_lim, x_lim], [-y_lim, y_lim], x_points=800)

            # plt.contour(image, levels=levels, vmin=1e-4, colors='silver', linewidths=1, alpha=0.3, origin='lower', extent=[-x_lim, x_lim, -y_lim, y_lim])
            # if quantities == ['vR', 'vz']:
            #     plt.contour(image, levels=levels, cmap='Greys', norm=colors.LogNorm(vmin=10**-3.6, vmax=10**-1), linewidths=1, alpha=0.3, origin='lower', extent=[-x_lim, x_lim, -y_lim, y_lim])

            levels = 10**np.linspace(-3.6, -1, 8)
            plt.contour(image, levels=levels, vmin=1e-4, cmap='Greys', norm=colors.LogNorm(vmin=10**-3.6, vmax=10**-1), linewidths=1, alpha=0.3, origin='lower', extent=[-x_lim, x_lim, -y_lim, y_lim])

        plt.xlabel('$p_x$ (kpc)'); plt.ylabel('$p_z$ (kpc)')
        # plt.title('Galactic Density')
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes('right', size='3%', pad=0.1)
        cbar = plt.colorbar(im, cax=cax)
        cbar.ax.set_ylabel('Number fraction of total stars', labelpad=15, rotation=270)
    elif style == 'speed':
        filename += '_speed'

        df = pd.read_csv(kicked_filename)
        df_raw = pd.read_csv(raw_kicked_filename)
        df = df[df['rtype'] == 'Neutron Star']
        df_raw = df_raw[df_raw['rtype'] == 'Neutron Star']
        title = 'Velocity Distribution'

        # Filter raw to young pulsars
        # df_raw = df_raw.query('age < 0.5 and 4 < R < 12').copy()
        if 'sample' in args:
            filename += '_sample'
            args.remove('sample')
            df_raw = df_raw.query('4 < R < 16 and -8 < pz < 8').copy()
            df = df.query('4 < R < 16 and -8 < pz < 8').copy()

        float_args = [i for i in args if type(i) == float]
        if len(float_args) > 0:
            filename += f'_{float_args[0]:.3f}'
            args.remove(float_args[0])
            df = df[df['age'] <= float_args[0]]

        # Convert to Local Standard of Rest
        vcirc_data = np.genfromtxt('../vcirc_potential.csv', dtype=None, names=True, delimiter=',') # load vcirc data

        # For integrated remnants
        vcirc = np.interp(df['R'], vcirc_data['r'], vcirc_data['vcirc'])  # circular velocity at the location of a star rgc
        df['vT'] = df['vT'] - np.sign(df['vT'].mean())*vcirc  # subtract the vcric from the azimuthal component of the star

        # For unintegrated remnants
        vcirc = np.interp(df_raw['R'], vcirc_data['r'], vcirc_data['vcirc'])  # circular velocity at the location of a star rgc
        df_raw['vT'] = df_raw['vT'] - np.sign(df_raw['vT'].mean())*vcirc  # subtract the vcric from the azimuthal component of the star

        if 'escaping' in args:
            filename += '_escaping'
            args.remove('escaping')
            df = df[df['will_escape'] == True]
            df_raw = df_raw[df_raw['will_escape'] == True]
        elif 'not_escaping' in args:
            filename += '_not_escaping'
            args.remove('not_escaping')
            df = df[df['will_escape'] == False]
            df_raw = df_raw[df_raw['will_escape'] == False]

        remnant_speeds = np.sqrt(np.sum(df[['vR', 'vT', 'vz']].to_numpy()**2, axis=1))
        raw_remnant_speeds = np.sqrt(np.sum(df_raw[['vR', 'vT', 'vz']]**2, axis=1))


        # Create line plots from Igoshev paper
        xs = np.linspace(0, 1500, 1501)

        ys = kick_distribution(xs, 'igoshev_all')
        norm_factor = np.mean(ys)*1501
        ys /= norm_factor
        if 'all' in args:
            plt.plot(xs, ys, c='k', label='Igoshev (2020) All')

            mins, maxs = get_kick_distribution_uncertainties(xs, 'igoshev_all')
            mins /= norm_factor
            maxs /= norm_factor
            plt.ylim(plt.ylim())
            title = 'Velocity Distribution of Evolved Neutron Stars'
            plt.plot(xs, mins, c='k', ls='--', lw=1, label='Igoshev (2020) 68% C.I.')
            plt.plot(xs, maxs, c='k', ls='--', lw=1)
        elif 'young' not in args:
            plt.plot(xs, ys, label='Igoshev (2020) All')
        # plt.fill_between(xs, mins, maxs, alpha=0.2)

        ys = kick_distribution(xs, 'igoshev_young')
        norm_factor = np.mean(ys)*1501
        ys /= norm_factor
        if 'young' in args:
            plt.plot(xs, ys, c='k', label='Igoshev (2020) Young')

            mins, maxs = get_kick_distribution_uncertainties(xs, 'igoshev_young')
            mins /= norm_factor
            maxs /= norm_factor
            # plt.ylim(plt.ylim())
            # plt.ylim(bottom=0)
            plt.ylim([0, 0.003])
            title = 'Velocity Distribution of Natal Neutron Stars'
            plt.plot(xs, mins, c='k', ls='--', lw=1, label='Igoshev (2020) 68% C.I.')
            plt.plot(xs, maxs, c='k', ls='--', lw=1)

            # df_raw = df[df['age'] - df['lifetime'] < 0.25]
            # print(df_raw.describe()['age'])
            # raw_remnant_speeds = np.sqrt(np.sum(df_raw[['vx', 'vy', 'vz']]**2, axis=1))
        elif 'all' not in args:
            plt.plot(xs, ys, label='Igoshev (2020) Young')

        # plt.fill_between(xs, mins, maxs, alpha=0.2)
        # ys = kick_distribution(xs, 'igoshev_young_half')
        # ys /= np.mean(ys)*1501/0.8
        # plt.plot(xs, ys, '--', label='Igoshev (2020) Young Second Peak')

        if 'histogram' in args:
            filename += '_histogram'
            args.remove('histogram')

            int_kwargs = {'color': 'tab:red',
                        'bins': 50,
                        'alpha': 0.5,
                        'density': True,
                        'label': 'Velocity Distribution, Integrated',
                        'range': (0, 1500)}

            if 'outline' in args:
                filename += '_outline'
                args.remove('outline')
                int_kwargs['histtype'] = 'stepfilled'
                int_kwargs['edgecolor'] = 'k'
                int_kwargs['linewidth'] = 2

            unint_kwargs = int_kwargs.copy()
            unint_kwargs['color'] = 'tab:brown'
            unint_kwargs['label'] = 'Velocity Distribution, Unintegrated'

            if 'texture' in args:
                filename += '_texture'
                args.remove('texture')
                int_kwargs['hatch'] = '/'
                unint_kwargs['hatch'] = '-'

            if 'all' in args:
                filename += '_all'
                args.remove('all')
                int_kwargs['color'] = 'grey'
                int_kwargs['bins'] = 50
                bin_heights, _, _ = plt.hist(remnant_speeds, **int_kwargs)
                plt.ylim([0, bin_heights.max() + .0005])
            elif 'young' in args:
                filename += '_young'
                args.remove('young')
                unint_kwargs['color'] = 'grey'
                unint_kwargs['bins'] = 40
                plt.hist(raw_remnant_speeds, **unint_kwargs)
            else:
                _, edges, _ = plt.hist(remnant_speeds, **int_kwargs)
                if 'both' in args:
                    filename += '_both'
                    args.remove('both')

                    unint_kwargs['bins'] = edges
                    plt.hist(raw_remnant_speeds, **unint_kwargs)

        elif 'line' in args:
            filename += '_line'
            args.remove('line')

            bin_edges = np.linspace(0, 1500, 41)
            x_values = [(bin_edges[i] + bin_edges[i+1])/2 for i in range(len(bin_edges)-1)]
            x_values = [0] + x_values
            remnant_speed_line, _ = np.histogram(remnant_speeds, bins=bin_edges,
                                                density=True)
            remnant_speed_line = np.append(0, remnant_speed_line)

            # plt.plot(x_values, remnant_speed_line, c='tab:blue', linestyle='--',
            plt.plot(x_values, remnant_speed_line, linestyle='--',
                    label='Velocity Distribution, Integrated')

            if 'both' in args:
                filename += '_both'
                args.remove('both')
                raw_remnant_speed_line, _ = np.histogram(raw_remnant_speeds,
                                                            bins=bin_edges,
                                                            density=True)
                raw_remnant_speed_line = np.append(0, raw_remnant_speed_line)
                # plt.plot(x_values, raw_remnant_speed_line, c='tab:orange',
                plt.plot(x_values, raw_remnant_speed_line,
                        linestyle='--', label='Velocity Distribution, Unintegrated')

        plt.xlabel('|Velocity| (km/s)'); plt.ylabel('Fraction')
        plt.title(title)
        plt.xlim([0, 1500])
        leg = plt.legend(loc='upper right')
    elif style == 'marginalise':
        filename += '_marginalise'

        df_guw = pd.read_csv(kicked_filename)
        df_ns = df_guw[df_guw['rtype'] == 'Neutron Star']
        df_bh = df_guw[df_guw['rtype'] == 'Black Hole']
        df_guw_unkicked = pd.read_csv(unkicked_filename)
        df_renzo = pd.read_csv(renzo_filename)
        stellar_coords, _, _ = load_data(stellar_filename)

        colours = ['tab:blue', 'tab:orange', 'k', 'grey', 'tab:green']
        labels = ['Visible Galaxy', 'Neutron Stars', 'Black Holes', 'Unkicked Galactic Underworld', 'Undisrupted Binaries']

        if 'R' in args:
            filename += '_R'
            args.remove('R')

            stellar_radii = np.sqrt(np.sum(stellar_coords[:, :2]**2, axis=1))

            # bin_edges = get_radial_bins(0, 15, 40)
            bin_range = (0, 15)
            bin_edges = None
            results = []
            radii_list = [stellar_radii, df_ns['R'], df_bh['R'], df_guw_unkicked['R'], df_renzo['R']]

            if 'hobbs' in args:
                filename += '_hobbs'
                args.remove('hobbs')

                df_hobbs = pd.read_csv(hobbs_filename).query("rtype == 'Neutron Star'")

                radii_list.append(df_hobbs['R'])
                colours.append('tab:red')
                labels.append('NSs from Hobbs Distribution')

            for radii in radii_list:
                if bin_edges is None:
                    radii_result, bin_edges = np.histogram(radii, bins=30, range=bin_range, density=True)
                else:
                    radii_result, _ = np.histogram(radii, bins=bin_edges, range=bin_range, density=True)

                bin_areas = get_radial_bin_areas(bin_edges)
                radii_result = radii_result / bin_areas
                radii_result = radii_result / (radii_result * bin_areas).sum()
                # radii_result = radii_result / radii_result.sum()
                # print('Total number density:', (radii_result * bin_areas).sum())
                results.append(radii_result)


            plt.xlabel('Cylindrical Radius (kpc)')
            # plt.ylabel('Fraction of number density')
            plt.ylabel('Surface density')
            plt.xlim(bin_range)

            # x = np.linspace(0, 15, 1000)
            # plt.plot(x, np.exp(-x/3.760), color='r', label='Scale length')
            # plt.plot(x, np.exp(-x/0.999), color='r', label='Scale length')
            # plt.plot(x, np.exp(-x/3.760), color='r', label='Scale length')
            # a, b = 0.542, 0.949
            # plt.plot(x, exponential(x, a, b)*0.4, color='r', label='Scale length')
        elif 'z' in args:
            filename += '_z'
            args.remove('z')

            float_args = [i for i in args if type(i) == float]
            if len(float_args) > 0:
                args.remove(float_args[0])
                filename += f'_{float_args[0]:.1f}'
                bin_max = float_args[0]
            else:
                bin_max = 2

            bin_range = (0, bin_max)
            bin_edges = None
            results = []
            zs_list = [np.abs(df_bh['pz']), np.abs(df_ns['pz']), np.abs(stellar_coords[:, 2]), np.abs(df_guw_unkicked['pz']), np.abs(df_renzo['pz'])]

            if 'hobbs' in args:
                filename += '_hobbs'
                args.remove('hobbs')

                df_hobbs = pd.read_csv(hobbs_filename).query("rtype == 'Neutron Star'")

                zs_list.append(df_hobbs['pz'])
                colours.append('tab:red')
                labels.append('NSs from Hobbs Distribution')

            for zs in zs_list:
                if bin_edges is None:
                    z_result, bin_edges = np.histogram(zs, bins='auto', range=bin_range, density=True)
                else:
                    z_result, _ = np.histogram(zs, bins=bin_edges, range=bin_range, density=True)
                results.append(z_result)

            plt.xlabel('Galactic Height (kpc)')
            # plt.ylabel('Fraction of stars')
            plt.ylabel('P')
            plt.xlim(bin_range)

        bin_centres = [(bin_edges[i] + bin_edges[i+1])/2 for i in range(len(bin_edges)-1)]

        for result, colour, label in zip(results, colours, labels):
            plt.plot(bin_centres, result, alpha=0.5, color=colour, label=label)



        if 'log' in args:
            filename += '_log'
            args.remove('log')
            plt.gca().set_yscale('log')
        plt.title('Marginal Distribution of Galaxy')
        plt.ylim(bottom=0)
        plt.legend()
    elif style == 'bar':
        filename += '_bar'
        df_guw = pd.read_csv(kicked_filename)
        df_ns = df_guw.query('rtype == "Neutron Star"')
        df_bh = df_guw.query('rtype == "Black Hole"')
        points = 25

        for df, colour, label in zip([df_ns, df_bh], ['tab:blue', 'k'], ['Neutron Stars', 'Black Holes']):
            # Calculate spherical radial quantities
            x, y, z = df['px'].to_numpy(), df['py'].to_numpy(), df['pz'].to_numpy()
            vx, vy, vz = df['vx'].to_numpy(), df['vy'].to_numpy(), df['vz'].to_numpy()

            circ_r = np.sqrt(x**2 + y**2 + z**2)
            circ_vr = (x*vx + y*vy + z*vz)/circ_r

            bins = np.linspace(0, 15, points, endpoint = False)
            mean_velocities = []
            for i in range(len(bins) - 1):
                velocities = circ_vr[(bins[i] <= circ_r) & (circ_r < bins[i+1])]
                mean_velocities.append(np.mean(velocities))

            smoothing = 1 # 0 is raw data
            smooth_means = [np.mean(mean_velocities[i:i+smoothing+1]) for i in range(len(mean_velocities)-smoothing)]
            bin_centres = [np.mean(bins[i:i+smoothing+2]) for i in range(len(bins)-smoothing-1)]
            plt.plot(bin_centres, smooth_means, '.-', color=colour, label=label)
        plt.xlabel('Radius (kpc)'); plt.ylabel('Mean radial velocity (km/s)')
        plt.xlim(left=0)
        plt.legend()

        # if 'R' in args:
        #     filename += '_R'
        #     args.remove('R')
        #     axis = 'R'
        #     bins = np.linspace(0, 15, points, endpoint = False)
        #
        # elif 'z' in args:
        #     filename += '_z'
        #     args.remove('z')
        #     axis = 'pz'
        #
        #     bins = np.linspace(0, 9, points, endpoint = False)
        # elif 'circ_R' in args:
        #     filename += '_circ_R'
        #     args.remove('circ_R')
        #
        # for df, colour, label in zip([df_ns, df_bh], ['tab:blue', 'k'], ['Neutron Stars', 'Black Holes']):
        #     mean_velocities = []
        #     for i in range(len(bins) - 1):
        #         bin = df[(bins[i] <= df[axis]) & (df[axis] < bins[i+1])]
        #         velocities = np.sqrt(np.sum(bin[['vx', 'vy', 'vz']]**2, axis=1))
        #         mean_velocities.append(np.mean(velocities))
        #
        #     smoothing = 2 # 0 is raw data
        #     smooth_means = [np.mean(mean_velocities[i:i+smoothing+1]) for i in range(len(bins)-smoothing)]
        #     bin_centres = [np.mean(bins[i:i+smoothing+2]) for i in range(len(bins)-smoothing)]
        #     plt.plot(bin_centres, smooth_means, '.-', color=colour, label=label)
        # plt.xlabel(axis); plt.ylabel('Mean velocity (km/s)')
        # plt.xlim(left=0)
        # plt.legend()
    else:
        raise ValueError(f'Unknown style {style}.')

    if style in ['split', 'hist']:
        plt.tight_layout()
        for lh in leg.legendHandles:
            lh.set_alpha(1)

    filename += '_kicked'

    if len(args) != 0:
        raise ValueError(f'Unkown argument passed to function: {args}')
    if not subplot:
        print('Saving plots with filename:', filename)
        plt.tight_layout()
        plt.savefig(f'../{filename}.pdf')
        plt.savefig(f'../{filename}.png', dpi=512)
        # plt.show()
        plt.close()

def within_radius(point, radius, centre, method):
    if method == 'sphere':
        if np.linalg.norm(centre - point)*1e3 <= radius:
            return True
        return False
    elif method == 'torus':
        planar_distance = np.linalg.norm(point[:2]) - np.linalg.norm(centre[:2])
        distance = np.sqrt(planar_distance**2 + point[2]**2)
        if distance*1e3 <= radius:
            return True
        return False
    else:
        raise ValueError(f'Unknown method {method}.')

def calculate_nearest(extinct_coords, radius, centre, method='torus', undersample=1e4):
    nearby_extinct = np.array([i for i in extinct_coords if within_radius(i, radius, centre, method)])
    print(f'Found {len(nearby_extinct)} nearby extinct stars.')
    if method == 'sphere':
        volume = 4/3 * np.pi * radius**3
    elif method == 'torus':
        # V = 2 pi**2 r**2 R
        volume = 2 * np.pi**2 * radius**2 * (np.linalg.norm(centre[:2])*1e3)
    else:
        raise ValueError(f'Unknown method {method}.')

    density = len(nearby_extinct)*undersample/volume
    nearest_volume = 1/density
    print(f'Space density: 1 per {nearest_volume:.1e} pc^3')
    nearest_extinct = np.cbrt(3/(4*np.pi) * nearest_volume)

    return nearest_extinct

def calculate_density(kicked=True, radius=100):
    # radius is in pc
    if kicked:
        # extinct_filename = r'../kicked_remnants.csv'
        extinct_filename = kicked_filename
    else:
        # extinct_filename = r'../galaxia_f1e-4_bhm2.35.ebf'
        extinct_filename = r'../galaxia_f1e-3_bhm2.35.ebf'

    min_masses = [8, 8, 20]
    max_masses = [np.inf, 20, np.inf]
    labels = ['extinct star', 'Neutron Star', 'Black Hole']
    print(f'Using a radius of {radius} pc.')
    for i in range(len(labels)):
        extinct_coords, centre, _ = load_data(extinct_filename, min_mass=min_masses[i], max_mass=max_masses[i], verbose=0)
        nearest_extinct = calculate_nearest(extinct_coords, radius, centre, undersample=1e3)
        print(f'The nearest {labels[i]} is: {nearest_extinct:.2f} pc away.\n')
    print(f'The sun is located at {centre*1e3} pc.')

    _, _, _ = load_data(raw_kicked_filename, min_mass=8, distance=np.inf)
    stellar_filename = r'../milkyway_f1e-6.ebf'
    stellar_coords, centre, _ = load_data(stellar_filename, min_mass=0, number_of_stars=None, distance=np.inf)
    nearest_star = calculate_nearest(stellar_coords, radius, centre, undersample=1e6)
    print(f'For comparison, the nearest star is {nearest_star:.2f} pc away.')

def calculate_escaped():
    df_pre_kick = pd.read_csv(unkicked_filename)
    df_kicked = pd.read_csv(kicked_filename)
    escaped_kicked = df_kicked[df_kicked['will_escape'] == True]

    pre_kick_escaping = len(df_pre_kick[df_pre_kick['will_escape'] == True])/len(df_pre_kick)
    kicked_escaping = len(escaped_kicked)/len(df_kicked)
    ns_escaping = len(escaped_kicked[escaped_kicked['rtype'] == 'Neutron Star'])/len(df_kicked[df_kicked['rtype'] == 'Neutron Star'])
    bh_escaping = len(escaped_kicked[escaped_kicked['rtype'] == 'Black Hole'])/len(df_kicked[df_kicked['rtype'] == 'Black Hole'])

    print(f'Before kicks {pre_kick_escaping:.1%} remnants escaping, after kicks {kicked_escaping:.1%}.')
    print(f'Neutron stars escaping: {ns_escaping:.1%}, black holes escaping: {bh_escaping:.1%}')

    escaped_mass = 1.35*len(escaped_kicked[escaped_kicked['rtype'] == 'Neutron Star']) \
                    + 7.8*len(escaped_kicked[escaped_kicked['rtype'] == 'Black Hole'])
    escaped_mass *= 1e4
    stellar_mass = 5.04e10
    print(f'Total mass esacped: {escaped_mass:.2e} M.')
    print(f'This is equivalent to {escaped_mass/stellar_mass:.1%} of the stellar mass of the galaxy.')

def exponential(x, a, b):
    return a*np.exp(-x/b)

def calculate_scale_dimension(method='exponential', dimension='height', verbose=0):
    points = 10000
    distance = np.inf
    # _, _, number_of_stars = load_data(kicked_filename, min_mass=8, distance=distance)
    _, _, number_of_stars = load_data(raw_kicked_filename, min_mass=8, distance=distance)

    stellar_filename = r'../milkyway_f1e-6.ebf'
    # stellar_coords, centre = load_data(stellar_filename, min_mass=0, number_of_stars=number_of_stars, distance=distance)
    stellar_coords, centre, _ = load_data(stellar_filename, min_mass=0, distance=distance)
    df_extinct = pd.read_csv(kicked_filename)

    if method == 'kde':
        stellar_kde = gaussian_kde(stellar_coords[:, [0, 2]].T)
        stellar_z = get_galactic_density(stellar_kde, [-1, 1], [-3, 3], points=points)

        extinct_kde = gaussian_kde(df_extinct[['px', 'pz']].to_numpy().T)
        extinct_z = get_galactic_density(extinct_kde, [-1, 1], [-20, 20], points=points)

        plt.plot(stellar_z[:, 1], label='Stellar')
        plt.plot(extinct_z[:, 1], label='Dark')
        plt.legend()
        plt.savefig('../Galactic_cross_section.png')

        stellar_z = stellar_z[np.argmax(stellar_z[:, 1]):, 1]
        extinct_z = extinct_z[np.argmax(extinct_z[:, 1]):, 1]

        scale_height = np.argmax(stellar_z <= stellar_z[0]/np.e)
        print(f'Scale height of visible galaxy: {scale_height * 6/10:.1f} pc')
        scale_height = np.argmax(extinct_z <= extinct_z[0]/np.e)
        print(f'Scale height of Galactic Underworld: {scale_height * 40/10:.1f} pc')
    elif method == 'hist':
        d = 0.15
        x, y, z = stellar_coords[:, 0], stellar_coords[:, 1], stellar_coords[:, 2]

        stellar_central_density = len(stellar_coords[(0-d < x) & (x < 0+d) & (0-d < y) & (y < 0+d) & (0-d < z) & (z < 0+d)])
        print('*'*20)
        print('Visible galaxy')
        print(f'Centre density: {stellar_central_density}')
        book_scale_density = len(stellar_coords[(0-d < x) & (x < 0+d) & (0-d < y) & (y < 0+d) & (0.4-d < z) & (z < 0.4+d)])
        test_scale_density = len(stellar_coords[(0-d < x) & (x < 0+d) & (0-d < y) & (y < 0+d) & (0.65-d < z) & (z < 0.65+d)])
        print(f'Theoretical scale height density: {book_scale_density}')
        print(f'Test scale height density: {test_scale_density}')
        print(f'Scale value should be {stellar_central_density/np.e:.1f}')

        d = 0.3
        x, y, z = df_extinct['px'].to_numpy(), df_extinct['py'].to_numpy(), df_extinct['pz'].to_numpy()

        print('*'*20)
        print('Galactic Underworld')
        stellar_central_density = len(df_extinct[(0-d < x) & (x < 0+d) & (0-d < y) & (y < 0+d) & (0-d < z) & (z < 0+d)])
        print(f'Centre density: {stellar_central_density}')
        for i in np.linspace(0.5, 1.5, 11):
            test_scale_density = len(df_extinct[(0-d < x) & (x < 0+d) & (0-d < y) & (y < 0+d) & (i-d < z) & (z < i+d)])
            print(f'Test scale height density at {i:.1f}: {test_scale_density}')

        print(f'Scale value should be {stellar_central_density/np.e:.1f}')
        print(f'Second height value should be {stellar_central_density/np.e**2:.1f}')
    elif method == 'exponential':
        if dimension == 'height':
            # min_rad, max_rad = 7.8, 8.2
            min_rad, max_rad = 7.5, 8.5
            stellar_radii = np.sqrt(np.sum(stellar_coords[:, :2]**2, axis=1))
            stellar_z = stellar_coords[(stellar_radii > min_rad) & (stellar_radii < max_rad)][:, 2]
            df_guw = pd.read_csv(kicked_filename)
            # df_guw = pd.read_csv(unkicked_filename)
            # df_guw = pd.read_csv(hobbs_filename)
            df_guw = df_guw[(df_guw['R'] > min_rad) & (df_guw['R'] < max_rad)]
            ns_z = df_guw[df_guw['rtype'] == 'Neutron Star']['pz']
            bh_z = df_guw[df_guw['rtype'] == 'Black Hole']['pz']

            for z, name in zip([stellar_z, df_guw['pz'], ns_z, bh_z], ['visible galaxy', 'Galactic Underworld', 'neutron stars', 'black holes']):
                # z_selected = z[z <= np.percentile(z, 50)]
                hist, bin_edges = np.histogram(np.abs(z), bins='auto', range=(0, 2.5), density=True)
                bin_centres = [(bin_edges[i] + bin_edges[i+1])/2 for i in range(len(bin_edges)-1)]
                pars, cov = curve_fit(f=exponential, xdata=bin_centres, ydata=hist,
                                                        p0=[1, 1], bounds=(-np.inf, np.inf))
                stdevs = np.sqrt(np.diag(cov))
                if verbose:
                    plt.figure()
                    plt.plot(bin_centres, hist, label=name)
                    plt.plot(bin_centres, exponential(np.array(bin_centres), pars[0], pars[1]), '--', label=name)
                    plt.show()
                print(f'For the {name}, scale height = {pars[1]*1e3:.0f} +/- {stdevs[1]*1e3:.0f} pc')

        elif dimension == 'length':
            # max_z = 0.1
            max_z = 0.2
            # max_z = 0.5
            stellar_radii = np.sqrt(np.sum(stellar_coords[:, :2]**2, axis=1))
            stellar_radii = stellar_radii[np.abs(stellar_coords[:, 2]) < max_z]
            df_guw = pd.read_csv(kicked_filename)
            # df_guw = pd.read_csv(unkicked_filename)
            # df_guw = pd.read_csv(hobbs_filename)
            df_guw = df_guw[np.abs(df_guw['pz']) < max_z]
            ns_r = df_guw[df_guw['rtype'] == 'Neutron Star']['R']
            bh_r = df_guw[df_guw['rtype'] == 'Black Hole']['R']

            for r, name in zip([stellar_radii, df_guw['R'], ns_r, bh_r], ['visible galaxy', 'Galactic Underworld', 'neutron stars', 'black holes']):
                radii = r[r <= np.percentile(r, 50)]
                # print('Max radii:', np.percentile(radii, 50))
                hist, bin_edges = np.histogram(np.abs(radii), bins=40, range=(0, 15), density=True)
                bin_areas = get_radial_bin_areas(bin_edges)
                hist = hist / bin_areas

                # bin_edges = get_radial_bins(0, 15, 40)
                # hist, _ = np.histogram(np.abs(r), bins=bin_edges, range=(0, 15), density=True)

                bin_centres = [(bin_edges[i] + bin_edges[i+1])/2 for i in range(len(bin_edges)-1)]
                pars, cov = curve_fit(f=exponential, xdata=bin_centres, ydata=hist,
                                                        p0=[1, 1], bounds=(-np.inf, np.inf))
                stdevs = np.sqrt(np.diag(cov))

                if verbose:
                    plt.figure()
                    plt.plot(bin_centres, hist, label=name)
                    plt.plot(bin_centres, exponential(np.array(bin_centres), pars[0], pars[1]), '--', label=name)
                    plt.yscale('log')
                    plt.show()
                print(f'For the {name}, 50 percentile scale length = {pars[1]*1e3:.0f} +/- {stdevs[1]*1e3:.0f} pc')
                # print(f'{name} params: {pars[0]:.3f}, {pars[1]:.3f}')

        else:
            raise ValueError(f'Unknown dimension {dimension}.')
    else:
        raise ValueError(f'Unkown method: {method}')

def create_subplots(subplot = 'speed_histogram'):
    global kicked_filename, raw_kicked_filename
    old_kicked, old_raw = kicked_filename, raw_kicked_filename

    if subplot == 'speed_histogram':
        plt.figure(figsize=(6.4, 4.8*1.5))
        raw_kicked_filename = r'../kicked_remnants_igoshev_young_7.8_DC_integrated_sanjib_0myr.csv'
        plot_data(style='speed', args=['histogram', 'young'], subplot=311)
        kicked_filename = r'../kicked_remnants_igoshev_young_7.8_DC_integrated_final.csv'
        plot_data(style='speed', args=['histogram', 'all'], subplot=312)
        kicked_filename = r'../kicked_remnants_igoshev_young_7.8_DC_integrated_peter_25myr.csv'
        plot_data(style='speed', args=['histogram', 'all', 'sample'], subplot=313)
        plt.title('Velocity Distribution of Sampled Neutron Stars')
        plt.tight_layout()
        print('Saving Galactic_Underworld_speed_histograms')
        plt.savefig('../Galactic_Underworld_speed_histograms_25myr.pdf')
        plt.savefig('../Galactic_Underworld_speed_histograms_25myr.png', dpi=512)
    elif subplot == 'contours':
        plt.figure(figsize=(6.4*2, 4.8*2))
        plot_data(style='contour', args=[20, 'visible', 'lines'], subplot=221)
        plot_data(style='contour', args=[20, 'lines'], subplot=222)
        plot_data(style='contour', args=[20, 'type', 'lines'], subplot=223)
        plot_data(style='contour', args=[20, 'renzo', 'type', 'lines'], subplot=224)
        plt.tight_layout()
        print('Saving Galactic_Underworld_contour_comparison')
        plt.savefig('../Galactic_Underworld_contour_comparison.pdf')
        plt.savefig('../Galactic_Underworld_contour_comparison.png', dpi=512)
    elif subplot == 'sanjib_evolution':
        plt.figure(figsize=(6.4, 15))
        kicked_filename = r'../kicked_remnants_igoshev_young_7.8_DC_integrated_sanjib_0myr.csv'
        plot_data(style='speed', args=['histogram', 'all'], subplot=511)
        plt.title('Unevolved NSs')
        kicked_filename = r'../kicked_remnants_igoshev_young_7.8_DC_integrated_sanjib.csv'
        plot_data(style='speed', args=['histogram', 'all', 0.01], subplot=512)
        plt.title('NSs evolved for 10 Myr')
        kicked_filename = r'../kicked_remnants_igoshev_young_7.8_DC_integrated_sanjib.csv'
        plot_data(style='speed', args=['histogram', 'all', 0.025], subplot=513)
        plt.title('NSs evolved for 25 Myr')
        kicked_filename = r'../kicked_remnants_igoshev_young_7.8_DC_integrated_sanjib.csv'
        plot_data(style='speed', args=['histogram', 'all', 0.05], subplot=514)
        plt.title('NSs evolved for 50 Myr')
        kicked_filename = r'../kicked_remnants_igoshev_young_7.8_DC_integrated_sanjib.csv'
        plot_data(style='speed', args=['histogram', 'all', 0.1], subplot=515)
        plt.title('NSs evolved for 100 Myr')
        plt.tight_layout()
        print('Saving Galactic_Underworld_speed_histograms_for_sanjib_filter')
        plt.savefig('../Galactic_Underworld_speed_histograms_for_sanjib_filter.pdf')
        plt.savefig('../Galactic_Underworld_speed_histograms_for_sanjib_filter.png', dpi=512)
    elif subplot == 'peter_evolution':
        cuts = [1, 10, 25, 50, 100, 200]
        plt.figure(figsize=(6.4, 2.4*len(cuts)))
        for i, cut in enumerate(cuts):
            subplot_num = int(str(len(cuts)) + '1' + str(i+1))
            kicked_filename = f'../kicked_remnants_igoshev_young_7.8_DC_integrated_peter_{cut}myr.csv'
            plot_data(style='speed', args=['histogram', 'all', 'sample'], subplot=subplot_num)
            plt.title(f'NSs evolved for {cut} Myr')

        plt.tight_layout()
        print('Saving Galactic_Underworld_speed_histograms_for_peter_1-200myr_filter')
        plt.savefig('../Galactic_Underworld_speed_histograms_for_peter_1-200myr_filter.pdf')
        plt.savefig('../Galactic_Underworld_speed_histograms_for_peter_1-200myr_filter.png', dpi=512)

    elif subplot == 'hobbs':
        kicked_filename = hobbs_filename
        plot_data(style='contour', args=[20, 'type', 'lines'], subplot=111)
        plt.tight_layout()
        print('Saving Galactic_Underworld_contour_hobbs')
        plt.savefig('../Galactic_Underworld_contour_hobbs.pdf')
        plt.savefig('../Galactic_Underworld_contour_hobbs.png', dpi=512)
    else:
        raise ValueError(f'Unknown subplot: {subplot}')
    kicked_filename, raw_kicked_filename = old_kicked, old_raw


# kicked_filename = r'../kicked_remnants_igoshev_young_7.8_DC_integrated.csv'
# kicked_filename = r'../kicked_remnants_igoshev_young_7.8_DC_integrated_2.csv'
kicked_filename = r'../kicked_remnants_igoshev_young_7.8_DC_integrated_final.csv'
renzo_filename = r'../kicked_remnants_renzo_integrated_final.csv'
hobbs_filename = r'../kicked_remnants_hobbs_integrated.csv'
# unkicked_filename = r'../kicked_remnants_no_kick.csv'
unkicked_filename = r'../kicked_remnants_no_kick_final.csv'
# raw_kicked_filename = r'../kicked_remnants_igoshev_young_7.8_DC.csv'
raw_kicked_filename = r'../kicked_remnants_igoshev_young_7.8_DC_final.csv'
# raw_kicked_filename = r'../kicked_remnants_renzo.csv'
kicked = True

# create_subplots('speed_histogram')
# create_subplots('sanjib_evolution')
# create_subplots('contours')
# create_subplots('hobbs')
# create_subplots('peter_evolution')

# plot_data(style='contour', args=[20, 'visible', 'lines'])
# plot_data(style='contour', args=[20, 'lines'])
# plot_data(style='contour', args=[20, 'type', 'lines'])
# plot_data(style='contour', args=[20, 'renzo', 'type', 'lines'])

# # plot_data(style='speed', args=['histogram', 'both'])
# for species in ['young', 'all']:
#     plot_data(style='speed', args=['histogram', species])
#
# # for i in np.linspace(0.01, 0.05, 5):
# #     plot_data(style='contour', args=[10, float(i)])
# # plot_data(style='contour', args=[20])
# # plot_data(style='contour', args=[20, 'type'])
# # plot_data(style='contour', args=[20, 'escape'])

# for i in [5, 10, 20]:
#     plot_data(style='contour', args=[i, 'split_type'])
#     plot_data(style='contour', args=[i, 'type'])
#     plot_data(style='contour', args=[i, 'type', 'lines'])
#     plot_data(style='contour', args=[i, 'escape'])
#     plot_data(style='contour', args=[i])
#     plot_data(style='contour', args=[i, 'lines'])
    # plot_data(style='contour', args=[i, 'visible', 'lines'])
    # plot_data(style='contour', args=[i, 'type', 'unkicked', 'lines'])

# plot_data(style='contour', args=[20, 'type', 'thin_disk', 'lines'])
# plot_data(style='contour', args=[20, 'type', 'thick_disk', 'lines'])
# plot_data(style='contour', args=[20, 'type', 'halo', 'lines'])
# plot_data(style='contour', args=[20, 'type', 'bulge', 'lines'])
# plot_data(style='contour', args=[20, 'type', 'lines'])
# plot_data(style='contour', args=[20, 'visible', 'lines'])
# plot_data(style='contour', args=[20, 'renzo', 'type', 'lines'])
# plot_data(style='contour', args=[20, 'lines'])
# plot_data(style='contour', args=[20, 'lines', 'type', 'unkicked'])
# plot_data(style='contour', args=[20, 'type', 'velocity'])
# plot_data(style='contour', args=[20, 'type', 'unkicked', 'lines'])

# plot_data(style='bar', args=[])
# plot_data(style='bar', args=['R'])
# plot_data(style='bar', args=['z'])

# for i in [5, 10, 20, 50, 100, 200, 500]:
#     plot_data(style='split', args=[i])
#     plot_data(style='split', args=[i, 'escape'])
#     plot_data(style='split', args=[i, 'type'])

# for axis in ['R', 'z']:
#     # plot_data(style='marginalise', args=[axis])
#     # plot_data(style='marginalise', args=[axis, 'log'])
#     plot_data(style='marginalise', args=[axis, 'hobbs'])
#     plot_data(style='marginalise', args=[axis, 'hobbs', 'log'])

# plot_data(style='marginalise', args=['R', 'log'])

# for radius in [50, 100, 200, 500, 1000]:
#     calculate_density(radius=radius)
#     input('*'*10)

# calculate_escaped()
#
# for dimension in ['height', 'length']:
#     calculate_scale_dimension(dimension=dimension)
#     print('*'*5)
#
