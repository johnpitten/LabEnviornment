import skrf as rf
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress


def db(x):
    return 20*np.log10(np.abs(x))

#TODO: make this accept rf.Network object instead?
def find_resonators(freqs, S21, savedir = False):
    #np.array_split data into 80 subarrays
    #find the deltas for each subarray
    #argsort the delta array, grab the indices of the eight largest deltas
    #sort the indices
    print('finding subarrays containing resonators')

    sdata = np.array_split(S21, 80)
    fdata = np.array_split(freqs, 80)

    delta = np.empty(len(sdata))
    for ind in range(len(sdata)):
        #calculate the linear fit to db(S21)
        reg = linregress(x = fdata[ind], y = db(sdata[ind]))
        resids = db(sdata[ind]) - (reg.intercept + reg.slope*fdata[ind])
        delta[ind] = np.max(resids) - np.min(resids)

    #grab the indices corresponding to blocks of data which contain resonators, then sort them
    w = np.sort(np.argsort(delta)[-8:])

    #TODO: make a mosaic plot of this data as a visual check
    if savedir:
        fig, ax = plt.subplots(nrows = 2, ncols = 4, layout = 'constrained', figsize = (6.4*2,5.2))
        ax = ax.reshape((8,)) #reshape ax from 2D array (shape (4,2)) into 1D array (shape (8,))

    #TODO: initialize linewidths to be larger than df_cutoff = 1MHz
    #TODO: while loop to keep recalculating linewidths after stepping up the sigma multiplier by 0.5, break once it hits 6.5
    center_freqs = np.empty(len(w))
    linewidths = 1.1e6*np.ones(len(w))
    for n in range(len(w)):
        #once again calculate the residsuals
        ind = w[n]
        reg = linregress(x=fdata[ind], y=db(sdata[ind]))
        resids = db(sdata[ind]) - (reg.intercept + reg.slope * fdata[ind])
        sigma = np.std(resids)
        #mask fdata and sdata where resids are within 4 standard deviations of the linear trend

        #TODO: insert while loop here, recalculate linewidth until plausible, or until masking 6.5 sigma data
        z = 4
        df_cutoff = 1e6
        while linewidths[n] >= df_cutoff:
            data_mask = np.isclose(db(sdata[ind]), reg.intercept + reg.slope * fdata[ind], rtol = 0, atol = z*sigma)
            msdata = np.ma.masked_array(sdata[ind], data_mask)
            mfdata= np.ma.masked_array(fdata[ind], data_mask)

            #TODO: round center_freqs entries to nearest kHz
            center_freqs[n] = np.ma.mean(mfdata)
            linewidths[n] = np.ma.max(mfdata)-np.ma.min(mfdata)
            if linewidths[n] == 0:
                linewidths[n] = 3000
            elif linewidths[n] > df_cutoff:
                z = z + 0.5
            if z >=7:
                break





            if savedir and linewidths[n] < 1e6:
                ax[n].plot(fdata[ind], db(sdata[ind]), color='b', label='data')
                ax[n].plot(fdata[ind], reg.intercept + reg.slope * fdata[ind], color = 'k', label ='linear fit')
                ax[n].plot(fdata[ind], reg.intercept + reg.slope * fdata[ind]+4*sigma, color='k', linestyle = 'dashed', label = fr'{z}$\sigma$')
                ax[n].plot(fdata[ind], reg.intercept + reg.slope * fdata[ind]-4*sigma, color='k', linestyle = 'dashed')
                ax[n].set_title(f'Resonator {n+1}')
                ax[n].set_xlabel('Frequency')
                ax[n].set_ylabel('Magnitude (dB)')
                ax[n].legend()

    if savedir is True:
        plt.show()
    elif type(savedir) == str:
        plt.savefig(fname = savedir + '/resonator_search')

    return center_freqs, linewidths