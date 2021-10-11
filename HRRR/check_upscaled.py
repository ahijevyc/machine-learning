import numpy as np
import time, os, sys
from mpl_toolkits.basemap import *
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import pygrib, pickle

#fh = pygrib.open('/glade/scratch/sobash/HRRR/2021050900/hrrr.t00z.wrfsfcf00.grib2')
#sblcl = fh[151].values
#fh.close()
#print(sblcl.max(), sblcl.min())

mask  = pickle.load(open('/glade/work/sobash/NSC_objects/HRRR/usamask_mod.pk', 'rb'))
#mask = np.logical_not(mask)
mask = mask.reshape((65,93))

data = np.load('/glade/work/sobash/NSC/2021050400_HRRR_upscaled.npz', allow_pickle=True)
upscaled_fields = data['a'].item()

for key in upscaled_fields.keys():
    this_field0 = np.array(upscaled_fields[key])[0,:]
    this_field1 = np.array(upscaled_fields[key])[1,:]
    this_field2 = np.array(upscaled_fields[key])[2,:]
    print(key,this_field0.max(), this_field1.max(), this_field2.max())


# map projection info
awips = Basemap(projection='lcc', llcrnrlon=-133.459, llcrnrlat=12.19, urcrnrlon=-49.38641, urcrnrlat=57.2894, lat_1=25.0, lat_2=25.0, lon_0=-95,area_thresh=10000.)
grid81 = awips.makegrid(93, 65, returnxy=True)
x81, y81 = awips(grid81[0], grid81[1])

# plot stuff
plotting = True
if plotting:
    plot_field = np.array(upscaled_fields['T500'])
    #plot_field = np.amax(plot_field, axis=0)

    #levels = np.arange(250,5000,250)
    #levels = np.arange(0,100,2.5)
    #test = readNCLcm('MPL_Reds')[10:]
    #cmap = colors.ListedColormap(test)
    #norm = colors.BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    ##awips.pcolormesh(x81, y81, np.ma.masked_less(u_interp, 100.0), cmap=cmap, norm=norm)
    awips.pcolormesh(x81, y81, np.ma.masked_array(plot_field[0,:], mask=np.logical_not(mask)))
    #awips.pcolormesh(x81, y81, env['b'], cmap=cmap, norm=norm)
    awips.drawstates()
    awips.drawcountries()
    awips.drawcoastlines()
    plt.savefig('test.png')
