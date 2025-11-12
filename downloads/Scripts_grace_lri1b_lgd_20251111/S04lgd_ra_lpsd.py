from scipy.io import loadmat
import numpy as np
from scipy.signal import filtfilt, windows, kaiserord, firwin
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import os
from datetime import datetime, timedelta

import warnings

# 忽略所有警告
warnings.filterwarnings("ignore")

def lpsd(x, windowfcn, fmin, fmax, Jdes, Kdes, Kmin, fs, xi):
    """
    LPSD Power spectrum estimation with a logarithmic frequency axis.
    Estimates the power spectrum or power spectral density of the time series x at JDES frequencies equally spaced (on
    a logarithmic scale) from FMIN to FMAX.
    Originally at: https://github.com/tobin/lpsd
    Translated from Matlab to Python by Rudolf W Byker in 2018.
    The implementation follows references [1] and [2] quite closely; in
    particular, the variable names used in the program generally correspond
    to the variables in the paper; and the corresponding equation numbers
    are indicated in the comments.
    References:
        [1] Michael Tröbs and Gerhard Heinzel, "Improved spectrum estimation
        from digitized time series on a logarithmic frequency axis," in
        Measurement, vol 39 (2006), pp 120-129.
            * http://dx.doi.org/10.1016/j.measurement.2005.10.010
            * http://pubman.mpdl.mpg.de/pubman/item/escidoc:150688:1
        [2] Michael Tröbs and Gerhard Heinzel, Corrigendum to "Improved
        spectrum estimation from digitized time series on a logarithmic
        frequency axis."
    """

    # Sanity check the input arguments
    if not callable(windowfcn):
        raise TypeError("windowfcn must be callable")
    if not (fmax > fmin):
        raise ValueError("fmax must be greater than fmin")
    if not (Jdes > 0):
        raise ValueError("Jdes must be greater than 0")
    if not (Kdes > 0):
        raise ValueError("Kdes must be greater than 0")
    if not (Kmin > 0):
        raise ValueError("Kmin must be greater than 0")
    if not (Kdes >= Kmin):
        raise ValueError("Kdes must be greater than or equal to Kmin")
    if not (fs > 0):
        raise ValueError("fs must be greater than 0")
    if not (0 <= xi < 1):
        raise ValueError("xi must be: 0 <= xi 1")

    N = len(x)  # Table 1
    jj = np.arange(Jdes, dtype=int)  # Table 1

    if not (fmin >= float(fs) / N):  # Lowest frequency possible
        raise ValueError("The lowest possible frequency is {}, but fmin={}".format(float(fs) / N), fmin)
    if not (fmax <= float(fs) / 2):  # Nyquist rate
        raise ValueError("The Nyquist rate is {}, byt fmax={}".format(float(fs) / 2, fmax))

    g = np.log(fmax) - np.log(fmin)  # (12)
    f = fmin * np.exp(jj * g / float(Jdes - 1))  # (13)
    rp = fmin * np.exp(jj * g / float(Jdes - 1)) * (np.exp(g / float(Jdes - 1)) - 1)  # (15)

    # r' now contains the 'desired resolutions' for each frequency bin, given the rule that we want the resolution to be
    # equal to the difference in frequency between adjacent bins. Below we adjust this to account for the minimum and
    # desired number of averages.

    ravg = (float(fs) / N) * (1 + (1 - xi) * (Kdes - 1))  # (16)
    rmin = (float(fs) / N) * (1 + (1 - xi) * (Kmin - 1))  # (17)

    case1 = rp >= ravg  # (18)
    case2 = np.logical_and(rp < ravg, np.sqrt(ravg * rp) > rmin)  # (18)
    case3 = np.logical_not(np.logical_or(case1, case2))  # (18)

    rpp = np.zeros(Jdes)

    rpp[case1] = rp[case1]  # (18)
    rpp[case2] = np.sqrt(ravg * rp[case2])  # (18)
    rpp[case3] = rmin  # (18)

    # r'' contains adjusted frequency resolutions, accounting for the finite length of the data, the constraint of the
    # minimum number of averages, and the desired number of averages.  We now round r'' to the nearest bin of the DFT
    # to get our final resolutions r.
    L = np.around(float(fs) / rpp).astype(int)  # segment lengths (19)
    r = float(fs) / L  # actual resolution (20)
    m = f / r  # Fourier Tranform bin number (7)

    Pxx = np.zeros(Jdes)
    S1 = np.zeros(Jdes)
    S2 = np.zeros(Jdes)

    # Loop over frequencies.  For each frequency, we basically conduct Welch's method with the fourier transform length
    # chosen differently for each frequency.
    for jj in range(len(f)):
        # Calculate the number of segments
        D = int(np.around((1 - xi) * L[jj]))  # (2)
        K = int(np.floor((N - L[jj]) / float(D) + 1))  # (3)

        # reshape the time series so each column is one segment  <-- FIXME: This is not clear.
        a = np.arange(L[jj])
        b = D * np.arange(K)
        ii = a[:, np.newaxis] + b  # Selection matrix
        data = x[ii]  # x(l+kD(j)) in (5)

        # Remove the mean of each segment.
        data = data - np.mean(data, axis=0)  # (4) & (5)

        # Compute the discrete Fourier transform
        window = windowfcn(L[jj]+2)[1:-1]  # (5) #signal.hann is equivalent to Matlab hanning, however, the first and the last elements are zeros, need to be removed
        window = window[:, np.newaxis]

        sinusoid = np.exp(-2j * np.pi * np.arange(L[jj])[:, np.newaxis] * m[jj] / L[jj])  # (6)
        data = data * (sinusoid * window)  # (5,6)

        # Average the squared magnitudes
        Pxx[jj] = np.mean(np.abs(np.sum(data, axis=0)) ** 2)  # (8) #python sum over column should be np.sum(data, axis=0) insteads of np.sum(data)

        # Calculate some properties of the window function which will be used during calibration
        S1[jj] = sum(window)  # (23)
        S2[jj] = sum(window ** 2)  # (24)

    # Calculate the calibration factors
    C = {
        'PS': 2. / (S1 ** 2),  # (28)
        'PSD': 2. / (fs * S2)  # (29)
    }

    return Pxx, f, C



def lgd_ra_lasd(date_str, groops_workspace, data_type='ra', instrument='LRI'):
    """
    计算 LGD 时间序列的 LPSD 功率谱密度
    lgd和ra数据的单位是m/s^2，计算功率谱密度（PSD）的单位是m²/s⁴/Hz，振幅谱密度（ASD）的单位是m/s²/√Hz。本方法计算的是LRI-LGD的振幅谱密度。
    """

    input_dir = os.path.join(groops_workspace, 'results')
    output_dir = os.path.join(groops_workspace, 'results')

    ori_filename = os.path.join(input_dir,  f'time-{data_type}-{date_str}.mat') # 原始数据文件路径
    cwt_filename = os.path.join(input_dir, f'cwt_time-{data_type}-{date_str}.mat')  # 小波重构数据文件路径

    if not os.path.exists(ori_filename):
        raise FileNotFoundError(f'原始数据文件不存在: {ori_filename}')
    if not os.path.exists(cwt_filename):
        raise FileNotFoundError(f'小波重构数据文件不存在: {cwt_filename}')

    ori_data = loadmat(ori_filename)[f'time_{data_type}'].astype(np.float64)
    cwt_data = loadmat(cwt_filename)

    cwt_time = cwt_data['time'].squeeze()  # 当日时间，累积秒，如5、10、15、20、...
    cwt_var = cwt_data[f'cwt_{data_type}'].squeeze()  # 原始数据，ra和lgd的单位都为m/s^2

    ori_time = cwt_time
    ori_var = ori_data[:, 1]    # 滤波后数据，ra和lgd的单位都为m/s^2

    fs = 0.5 if instrument == 'LRI' else 0.2    # LRI频率为0.5，KBR频率为0.2
    fmax = 2e-1 if instrument == 'LRI' else 1e-1
    x_ins, f_ins, c_ins = lpsd(ori_var, windows.nuttall, 1e-4, fmax, 600, 12, 4, fs, 0.5)

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.loglog(f_ins,
              np.sqrt(x_ins*c_ins['PSD']),    # ASD 是振幅谱密度，是 PSD 的平方根
              linewidth=4,
              color='green',
              label=f"{instrument}-{data_type.upper()}")

    ax.tick_params(labelsize=25, width=2.9)
    ax.set_xlabel('Frequency [Hz]', fontsize=20)
    ax.set_xlim([1e-4, 5e-1])
    ax.yaxis.get_offset_text().set_fontsize(24)
    ax.set_ylabel(r'ASD [m/s$^2 / \sqrt{Hz}$]', fontsize=20)
    ax.legend(fontsize=15, loc='best', frameon=False)
    ax.grid(True, which='both', ls='dashed', color='0.5', linewidth=0.6)
    plt.setp(ax.spines.values(), linewidth=3)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: r"$10^{%d}$" % np.log10(x)))
    ax.set_title(f'{data_type.upper()}-ASD for {date_str}', fontsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{data_type.upper()}-ASD for {date_str}.png'),
                dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    # plt.show()



def run(start_date, end_date, groops_workspace, instrument='LRI'):
    """
    运行程序, 绘制ra和lgd的ASD
    :param start_date: 起始日期，格式：'2020-06-15'
    :param end_date: 结束日期，格式：'2020-06-15'
    :param groops_workspace: GROOPS工作目录，格式：'G:\GROOPS\PNAS2020Workspace'
    :param instrument: 仪器类型，'LRI'或'KBR'
    """

    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')

    dates = [(start + timedelta(days=x)).strftime('%Y-%m-%d')
             for x in range((end - start).days + 1)]
    for date_str in dates:
        lgd_ra_lasd(date_str, groops_workspace, 'lgd', instrument)
        lgd_ra_lasd(date_str, groops_workspace, 'ra', instrument)

# if __name__ == '__main__':
#     start_date = '2020-06-15'
#     end_date = '2020-06-15'
#     groops_workspace = 'G:\GROOPS\PNAS2020Workspace'
#     run(start_date, end_date, groops_workspace)