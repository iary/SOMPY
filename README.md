SOMPY
-----
A Python Library for Self Organizing Map (SOM), credits at the end of this doc.

This version has been forked and modified to be used for astronomical purposes. [ID]

-----

### New features


##### Class ``Whatever``

PLEASE ALEX PUT HERE A DESCRIPTION OF WHAT YOU DID.

##### Function ``project_realdata`` 

Similar to ``project_data``, but it takes also an array-like input with standard deviation of the data to map. The basic idea is that the nearest BMU is assigned with a chi<sup>2</sup>-distance, i.e. (X-Y)<sup>2</sup>/Sigma<sup>2</sup>.
Example:
```python
# given two datasets of galaxy colors, one to train (data_train) 
# and one to be mapped (data_map). For the latter we also know
# 1sigma error for each color (errdata_map).
som = sompy.SOMFactory.build(data_train, mapsize=(20,20))
som.train(shared_memory='yes')
ac = som.bmu_ind_to_xy(som.project_realdata(data_map,errdata_map))
```



[Basice Example](https://gist.github.com/sevamoo/035c56e7428318dd3065013625f12a11)

### Citation

There is no published paper about this library. However if possible, please cite the library as follows:

```
Main Contributers:
Vahid Moosavi @sevamoo
Sebastian Packmann @sebastiandev
Iván Vallés @ivallesp 
```



