# genper
https://travis-ci.com/estanislaoledesma/genper.svg?branch=master
GenPer es un software de tomografía por microondas capaz de reconstruir imágenes de permitividades de dieléctricos a 
partir de sus datos de dispersión.

## Estructura del proyecto

### configs
Contiene todos los hiperparámetros, rutas de archivos y cualquier cosa configurable del proyecto.

### dataloader
Contiene todas las clases para carga y preprocesamiento de datos.

### docs
Contiene toda la documentación del proyecto.

### evaluation
Contiene todas las clases encargadas de evaluar el rendimiento y exactitud del modelo.

### executor
Contiene todas las clases encargadas de entrenar el modelo, tanto en CPU como GPU.

### model
Contiene todas las clases que conforman la red neuronal en sí.

### scripts
Contiene los scripts para ejecutar cada parte del código por separado.

### tests
Contiene todos los archivos de tests.

### utils
Contiene todas las clases que pueden ser reutilizadas a lo largo de todo el proyecto.

## Referencias

Z. Wei and X. Chen, “Deep learning schemes for full-wave nonlinear inverse scattering problems,” IEEE Transactions on Geoscience and Remote Sensing, 57 (4), pp. 1849-1860, 2019.

Z. Wei# and X. Chen, “Uncertainty Quantification in Inverse Scattering Problems with Bayesian Convolutional Neural Networks” IEEE Transactions on Antennas and Propagation, 10.1109/TAP.2020.3030974, 2020.

K. H. Jin, M. T. McCann, E. Froustey, and M. Unser, “Deep convolutional neural network for inverse problems in imaging,” IEEE Transactions on Image Processing, vol. 26, no. 9, pp. 4509–4522, 2017.