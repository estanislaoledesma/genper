# genper
[![Build Status](https://app.travis-ci.com/estanislaoledesma/genper.svg?branch=master)](https://app.travis-ci.com/estanislaoledesma/genper)
[![codecov](https://codecov.io/gh/estanislaoledesma/genper/branch/master/graph/badge.svg?token=EFvAqvxTxQ)](https://codecov.io/gh/estanislaoledesma/genper)

GenPer es un software de tomografía por microondas capaz de reconstruir imágenes de permitividades de dieléctricos a 
partir de sus datos de dispersión.

## Estructura del proyecto

### configs
Contiene todos los hiperparámetros, rutas de archivos y cualquier cosa configurable del proyecto.

### data
Contiene todos los archivos generados durante el procesamiento del software.

### dataloader
Contiene todas las clases para carga y preprocesamiento de datos.

### docs
Contiene toda la documentación del proyecto.

### evaluation
Contiene todas las clases encargadas de evaluar el rendimiento y exactitud del modelo.

### executor
Contiene todas las clases encargadas de entrenar el modelo, tanto en CPU como GPU.

### logs
Contiene los logs de cada ejecución.

### model
Contiene todas las clases que conforman la red neuronal en sí.

### scripts
Contiene los scripts para ejecutar cada parte del código por separado.

#### generate_images.py
Mediante este script se llama al generador de imágenes, el cual generará un archivo images.h5 dentro de data/image_generator,  
el cual será utilizado por el preprocesador. También generará los correspondientes logs dentro de logs/image_generator junto 
con algunos archivos png de muestra de las imágenes generadas. Se le puede pasar la opción *-t* o *--test* para ejecutar 
en modo testing.

#### preprocess_images.py
Mediante este script se llama al preprocesador de imágenes, el cual cargará el archivo images.h5 generado por el generador 
de imágenes dentro de data/image_generator. A estas imágenes las procesará y generará su correspondiente archivo 
preprocessed_images.h5 dentro de data/preprocessor. También generará los correspondientes logs dentro de logs/preprocessor. 
Se le puede pasar la opción *-t* o *--test* para ejecutar en modo testing.

#### train_model.py
Mediante este script se llama al entrenador del modelo, el cual creará la red neuronal u-net basado en los parámetros 
configurables dentro de configs/basic_parameters.json. También cargará el set de datos generado por *preprocess_images.py*, 
el cual será dividido en tres, un set de entrenamiento, otro de validación y un último de testing de manera aleatoria. 
Entrenará y validará el modelo, el cual será guardado en data/trainer/trained_model.pt. El set de datos de testing será 
guardado en data/trainer/test_images.pt También generará los correspondientes logs dentro de logs/trainer.
Se le puede pasar la opción *-t* o *--test* para ejecutar en modo testing, *-l* o *--load* para retomar un entrenamiento 
previamente interrumpido.

### tests
Contiene todos los archivos de tests.

### utils
Contiene todas las clases que pueden ser reutilizadas a lo largo de todo el proyecto.

## Referencias

Z. Wei and X. Chen, “Deep learning schemes for full-wave nonlinear inverse scattering problems,” IEEE Transactions on Geoscience and Remote Sensing, 57 (4), pp. 1849-1860, 2019.

Z. Wei# and X. Chen, “Uncertainty Quantification in Inverse Scattering Problems with Bayesian Convolutional Neural Networks” IEEE Transactions on Antennas and Propagation, 10.1109/TAP.2020.3030974, 2020.

K. H. Jin, M. T. McCann, E. Froustey, and M. Unser, “Deep convolutional neural network for inverse problems in imaging,” IEEE Transactions on Image Processing, vol. 26, no. 9, pp. 4509–4522, 2017.
