import numpy as np
import pygrib
import write_grib as wg
import datetime

field = np.random.random(size=(6,65,93))
print(field.max())
print(field[0,:])

wg.write_grib(field, datetime.datetime(2021,3,1,0,0,0), 2, 'test.grb')

fh = pygrib.open('test.grb')
data = fh[1].values
fh.close()

print(data.max())
print(data)
