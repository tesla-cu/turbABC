import logging
import os
import sys

import matplotlib as mpl
mpl.use('pdf')
import matplotlib.pyplot as plt
import numpy as np
import yaml
from numpy.fft import fftfreq, fftn


LES_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
REPO_ROOT = os.path.abspath(os.path.join(LES_ROOT, '..'))
for path in [REPO_ROOT, LES_ROOT]:
    if path not in sys.path:
        sys.path.insert(0, path)

import data

single_column = 252


def fig_size(width_column):
    fig_width_pt = width_column
    inches_per_pt = 1.0 / 72.27  # Convert pt to inches
    golden_mean = (np.sqrt(5) - 1.0) / 2.0  # Aesthetic ratio
    fig_width = fig_width_pt * inches_per_pt  # width in inches
    fig_height = fig_width * golden_mean  # height in inches
    return fig_width, fig_height


mpl.rcParams['font.size'] = 9
mpl.rcParams['axes.titlesize'] = 1.2 * plt.rcParams['font.size']
mpl.rcParams['axes.labelsize'] = plt.rcParams['font.size']
mpl.rcParams['legend.fontsize'] = plt.rcParams['font.size']
mpl.rcParams['xtick.labelsize'] = 0.8*plt.rcParams['font.size']
mpl.rcParams['ytick.labelsize'] = 0.8*plt.rcParams['font.size']
mpl.rcParams['font.family'] = 'serif'
mpl.rc('text', usetex=True)
# plt.rcParams['savefig.dpi'] = 2 * plt.rcParams['savefig.dpi']
mpl.rcParams['xtick.major.size'] = 3
mpl.rcParams['xtick.minor.size'] = 3
mpl.rcParams['xtick.major.width'] = 1
mpl.rcParams['xtick.minor.width'] = 0.5
mpl.rcParams['ytick.major.size'] = 3
mpl.rcParams['ytick.minor.size'] = 3
mpl.rcParams['ytick.major.width'] = 1
mpl.rcParams['ytick.minor.width'] = 1
# mpl.rcParams['legend.frameon'] = False
# plt.rcParams['legend.loc'] = 'center left'
plt.rcParams['axes.linewidth'] = 1


def imagesc(Arrays, map_bounds, titles, name=None):

    Arrays = np.array(Arrays)
    if titles is not None:
        titles = np.array(titles)
    cmap = plt.cm.jet  # define the colormap
    cmaplist = [cmap(i) for i in range(cmap.N)]  # extract all colors from the .jet map
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)  # create the new map
    norm = mpl.colors.BoundaryNorm(map_bounds, cmap.N)

    axis = [0, 2 * np.pi, 0, 2 * np.pi]
    ticks = ([0, np.pi, 2*np.pi], ['0', 'r$\pi$', 'r$2\pi$'])
    if len(Arrays) > 1:
        fig_width, fig_height = fig_size(single_column)
        print(Arrays.shape)
        nrows, ncols, _, _ = Arrays.shape
        print('nrows, ncols', nrows, ncols)
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharey=True, figsize=(fig_width, fig_width))
        im = []
        for row in range(nrows):
            for col in range(ncols):
                im. append(axes[row, col].imshow(Arrays[row, col].T, origin='lower', cmap=cmap,
                                                 norm=norm, interpolation="nearest", extent=axis))
                # axes[row, col].set_title(titles[row, col])
                axes[row, col].set_adjustable('box')
                axes[row, col].xaxis.set_ticks(ticks[0])
                axes[row, col].xaxis.set_ticklabels(ticks[1])
                axes[row, col].yaxis.set_ticks(ticks[0])
                axes[row, col].yaxis.set_ticklabels(ticks[1])
                if row == nrows-1:
                    axes[row, col].set_xlabel(r'$x$')
                else:
                    axes[row, col].xaxis.set_major_formatter(plt.NullFormatter())
                if col == 0:
                    axes[row, col].set_ylabel(r'$y$')
                else:
                    axes[row, col].yaxis.set_major_formatter(plt.NullFormatter())
        # cbar_ax = fig.add_axes([0.89, 0.18, 0.017, 0.68])  # ([0.85, 0.15, 0.05, 0.68])
        cbar_sigma = fig.add_axes([0.83, 0.45, 0.017, 0.67])
        cbar_prod = fig.add_axes([0.83, 0.2, 0.017, 0.43])
        fig.subplots_adjust(left=0.11, right=0.8, wspace=0.1, bottom=0.2, top=0.9)
        fig.colorbar(im[0], cax=cbar_sigma, ax=axes.ravel().tolist())
        fig.colorbar(im[-1], cax=cbar_prod, ax=axes.ravel().tolist())
    if name is not None:
        fig.savefig(name)
    plt.close('all')


def shell_average(spectrum_3d, n_points, wave_numbers):
    values = np.zeros(n_points ** 3)
    radii = np.empty_like(values)
    index = 0
    for ind_x, kx in enumerate(wave_numbers[0]):
        for ind_y, ky in enumerate(wave_numbers[1]):
            for ind_z, kz in enumerate(wave_numbers[2]):
                radius = round(np.sqrt(kx ** 2 + ky ** 2 + kz ** 2))
                radii[index] = radius
                values[index] = 2 * np.pi * radius ** 2 * spectrum_3d[ind_x, ind_y, ind_z]
                index += 1

    spectrum = sorted(zip(radii, values))
    x = [spectrum[0][0]]
    y = [spectrum[0][1]]
    count = 1
    for radius, value in spectrum[1:]:
        if radius == x[-1]:
            count += 1
            y[-1] += value
        else:
            y[-1] /= count
            x.append(radius)
            y.append(value)
            count = 1
    y[-1] /= count
    return x, y


def spectral_density(velocity_arrays, dx, n_points, output_path):
    wave_numbers = 2 * np.pi * np.array([
        fftfreq(n_points[0], dx[0]),
        fftfreq(n_points[1], dx[1]),
        fftfreq(n_points[2], dx[2]),
    ])
    spectrum_3d = 0
    for array in velocity_arrays:
        fft_array = fftn(array)
        spectrum_3d += np.real(fft_array * np.conj(fft_array))

    _, spectrum = shell_average(spectrum_3d, n_points[0], wave_numbers)
    with open(output_path, 'w') as f:
        f.writelines(f'{value}\n' for value in spectrum)


def plot_spectra(output_folder):
    spectra_files = sorted(
        file_name for file_name in os.listdir(output_folder) if file_name.endswith('.spectra')
    )
    if not spectra_files:
        logging.warning('No spectra files found in %s', output_folder)
        return

    fig = plt.figure(figsize=(4, 3))
    ax = plt.gca()
    labels = ['DNS', 'LES', 'TEST']
    for index, file_name in enumerate(spectra_files):
        values = np.loadtxt(os.path.join(output_folder, file_name))
        x = np.arange(len(values))
        label = labels[index] if index < len(labels) else file_name
        ax.loglog(x, values, '-', linewidth=2, label=label)

    y = 7.2e14 * np.power(x, -5 / 3)
    ax.loglog(x, y, 'r--', label=r'$-5/3$ slope')
    ax.set_title('Spectra')
    ax.set_ylabel(r'$E$')
    ax.set_xlabel(r'$k$')
    ax.axis(ymin=1e6)
    plt.legend(loc=0)
    fig.subplots_adjust(left=0.16, right=0.95, bottom=0.2, top=0.87)
    fig.savefig(os.path.join(output_folder, 'spectra.pdf'))
    plt.close(fig)

def field_gradient(field, dx):
    """Calculate tensor of gradients of self.field.
    :return:      dictionary of gradient tensor
    """
    grad = dict()
    grad['uu'], grad['uv'], grad['uw'] = np.gradient(field['u'], dx[0], dx[1], dx[2])
    grad['vu'], grad['vv'], grad['vw'] = np.gradient(field['v'], dx[0], dx[1], dx[2])
    grad['wu'], grad['wv'], grad['ww'] = np.gradient(field['w'], dx[0], dx[1], dx[2])
    return grad

def calc_strain_tensor(field, dx):
    """Calculate strain tensor S_ij = 1/2(du_i/dx_j+du_j/dx_i) of given field.
    :return:      dictionary of strain tensor
    """
    A = field_gradient(field, dx)
    tensor = dict()
    for i in ['u', 'v', 'w']:
        for j in ['u', 'v', 'w']:
            tensor[i + j] = 0.5 * (A[i + j] + A[j + i])
    return tensor


def deviatoric_stresses(field):
    """Calculate pdf of deviatoric stresses using DNS data.
        sigma_ij = tau_ij - 1/3 tau_kk*delta_ij
    :return:     dictionary of deviatoric stresses
    """
    tau = dict()
    for i in ['u', 'v', 'w']:
        for j in ['u', 'v', 'w']:
            tau[i + j] = field[i + j] - np.multiply(field[i], field[j])
    trace = tau['uu'] + tau['vv'] + tau['ww']
    for i in ['uu', 'vv', 'ww']:
        tau[i] -= 1 / 3 * trace
    return tau

def production(field, dx):
    sigma = deviatoric_stresses(field)
    prod_rate = 0
    S = calc_strain_tensor(field, dx)
    for i in ['u', 'v', 'w']:
        for j in ['u', 'v', 'w']:
            prod_rate += sigma[i + j] * S[i + j]
    return prod_rate


def sigma_production_fields(field, dx, output_folder):
    sigma = deviatoric_stresses(field)
    prod = production(field, dx)
    mid_plane = field['u'].shape[2] // 2
    map_bounds = np.linspace(-0.2, 0.2, 9)

    arrays = [
        [sigma['uu'][:, :, mid_plane], sigma['uv'][:, :, mid_plane]],
        [sigma['uw'][:, :, mid_plane], prod[:, :, mid_plane]],
    ]
    output_path = os.path.join(output_folder, 'sigma_production.pdf')
    imagesc(arrays, map_bounds, None, name=output_path)


def resolve_paths(params_path, params):
    base_dir = os.path.dirname(os.path.abspath(params_path))
    valid_folder = os.path.normpath(os.path.join(base_dir, params['path']['valid_data']))
    data_name = os.path.basename(valid_folder)
    plot_folder = os.path.join(base_dir, 'plots', 'data_plots', data_name)
    return valid_folder, plot_folder


def main(args=None):

    params_path = os.path.join(LES_ROOT, 'params.yml')

    with open(params_path, 'r') as f:
        params = yaml.safe_load(f)

    spectra = 1

    logging.basicConfig(format='%(levelname)s: %(name)s: %(message)s', level=logging.DEBUG)

    valid_folder, plot_folder = resolve_paths(params_path, params)
    if not os.path.isdir(valid_folder):
        raise FileNotFoundError(f'{valid_folder} not found.')
    if not os.path.isdir(plot_folder):
        os.makedirs(plot_folder)

    data_params = params['data_params']
    dx = np.array([data_params['lx'] / data_params['N_point']] * 3)

    logging.info('Load HIT data from %s', valid_folder)
    hit_data = data.load_HIT_data(valid_folder, data_params['N_point'])

    logging.info('Filter HIT data')
    les_data = data.DataFiltered.filter3d_dict(hit_data, data_params['LES_scale'], dx)

    logging.info('Write sigma and production figure')
    sigma_production_fields(les_data, dx, plot_folder)

    if spectra:
        logging.info('Write spectra files')
        spectral_density(
            [hit_data['u'], hit_data['v'], hit_data['w']],
            dx,
            [data_params['N_point']] * 3,
            os.path.join(plot_folder, 'DNS.spectra'),
        )
        spectral_density(
            [les_data['u'], les_data['v'], les_data['w']],
            dx,
            [data_params['N_point']] * 3,
            os.path.join(plot_folder, 'LES.spectra'),
        )
        plot_spectra(plot_folder)


if __name__ == '__main__':
    main()
