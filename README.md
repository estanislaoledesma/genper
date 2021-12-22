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
Contiene los logs de cada ejecución junto con imagenes generadas durante la misma.

### model
Contiene todas las clases que conforman la red neuronal en sí.

### scripts
Contiene los scripts para ejecutar cada parte del código por separado. Los mismos se ejecutan a través de `python <opciones> <script.py>`

#### generate_images.py
Mediante este script se llama al generador de imágenes, el cual creará un archivo *images.pkl* dentro de *data/image_generator*,  
el cual será utilizado por el preprocesador. También generará los correspondientes logs dentro de *logs/image_generator* junto 
con algunos archivos png de muestra de las imágenes generadas. Se le puede pasar la opción *-t* o *--test* para ejecutar 
en modo testing y *-t* o *--test* para que las imágenes contengan rectángulos en lugar de círculos.

#### generate_matlab_images.py
Mediante este script se llama al generador de imágenes de matlab, el cual creará un archivo *images.pkl* dentro de 
*data/matlab_image_generator*, el cual será utilizado por el preprocesador. También generará los correspondientes 
logs dentro de *logs/matlab_image_generator* junto con algunos archivos png de muestra de las imágenes generadas. 
Se le debe pasar la opción *-f* o *--file* y el nombre del archivo .mat que tiene los parámetros de las imágenes 
a crear. Para más información acerca de la estructura del archivo, referirse al manual de instrucciones de GenPer 
en docs/instructions.pdf.

#### generate_mnist_dataset.py
Mediante este script se llama al generador de dataset de mnist, el cual creará un archivo *images.pkl* dentro de 
*data/mnist_image_generator/mnist_training_images* y otro similar en *data/mnist_image_generator/mnist_training_images*, 
los cuales serán utilizados por el preprocesador. También generará los correspondientes logs dentro de *logs/mnist_image_generator* 
junto con algunos archivos png de muestra de las imágenes generadas. Este script descarga el dataset MNIST la primera vez y lo 
procesa de manera que genera imagenes de hasta 2 dígitos superpuestos.

#### preprocess_images.py
Mediante este script se llama al preprocesador de imágenes, el cual cargará el archivo *images.pkl* generado por el generador 
de imágenes dentro de *data/image_generator*. A estas imágenes las procesará y generará su correspondiente archivo 
*preprocessed_images.pkl* dentro de *data/preprocessor*. También generará los correspondientes logs dentro de *logs/preprocessor*, 
junto con algunos archivos png de muestra de las imágenes generadas. Se le puede pasar la opción *-t* o *--test* para ejecutar 
en modo testing.

#### preprocess_matlab_images.py
Mediante este script se llama al preprocesador de imágenes, el cual cargará el archivo *images.pkl* generado por el generador 
de imágenes de matlab dentro de *data/matab_image_generator*. A estas imágenes las procesará y generará su correspondiente archivo 
*preprocessed_images.pkl* dentro de *data/preprocessor/matlab_images*. También generará los correspondientes logs dentro de *logs/preprocessor*, 
junto con algunos archivos png de muestra de las imágenes generadas.

#### preprocess_mnist_dataset.py
Mediante este script se llama al preprocesador de imágenes, el cual cargará el archivo *images.pkl* generado por el generador 
de imágenes de mnist dentro de *data/mnist_dataset_generator/mnist_training_images* y *data/mnist_dataset_generator/mnist_testing_images*. 
A estas imágenes las procesará y generará su correspondiente archivo *preprocessed_images.pkl* dentro de *data/preprocessor/mnist_training_images* 
y *data/preprocessor/mnist_testing_images*. También generará los correspondientes logs dentro de *logs/preprocessor/mnist_preprocessed_images*, 
junto con algunos archivos png de muestra de las imágenes generadas.

#### train_model.py
Mediante este script se llama al entrenador del modelo, el cual creará la red neuronal u-net basado en los parámetros 
configurables dentro de *configs/basic_parameters.json*. También cargará el set de datos generado por *preprocess_images.py*, 
el cual será dividido en tres, un set de entrenamiento, otro de validación y un último de testing de manera aleatoria. 
Entrenará y validará el modelo, el cual será guardado en *data/trainer/trained_model.pt*. También generará los correspondientes 
logs dentro de logs/trainer junto con algunos archivos png de muestra de las imágenes generadas y un gráfico de errores de 
entrenamiento/validación vs iteración. Se le puede pasar la opción *-t* o *--test* para ejecutar en modo testing, 
*-l* o *--load* para retomar un entrenamiento previamente interrumpido.

#### train_model_with_matlab_images.py
Mediante este script se entrena y valida el modelo con las imagenes generadas por *preprocess_matlab_images.py*, el cual será 
guardado en *data/trainer/matlab/trained_model.pt*. También generará los correspondientes logs dentro de *logs/trainer/matlab*, 
junto con algunos archivos png de muestra de las imágenes generadas y un gráfico de errores de entrenamiento/validación vs 
iteración. Se le puede pasar la opción *-l* o *--load* para retomar un entrenamiento previamente interrumpido.

#### train_model_with_matlab_images.py
Mediante este script se entrena y valida el modelo con las imagenes generadas por *preprocess_mnist_images.py*, el cual será 
guardado en *data/trainer/mnist/trained_model.pt*. También generará los correspondientes logs dentro de *logs/trainer/mnist*, 
junto con algunos archivos png de muestra de las imágenes generadas y un gráfico de errores de entrenamiento/validación vs 
iteración. Se le puede pasar la opción *-l* o *--load* para retomar un entrenamiento previamente interrumpido.

#### test_model.py
Mediante este script se testea el modelo previamente entrenado por *train_model.py* con el set de datos de testing. Generará 
los correspondientes logs con información de error total y medio dentro de *logs/tester* junto con algunos archivos png de 
muestra de las imágenes generadas. Se le puede pasar la opción *-t* o *--test* para ejecutar en modo testing.

#### test_model_with_matlab_images.py
Mediante este script se testea el modelo previamente entrenado por *train_model_with_matlab_images.py* con el set de datos de testing. 
Generará los correspondientes logs con información de error total y medio dentro de *logs/tester/matlab* junto con algunos archivos png de 
muestra de las imágenes generadas.

#### test_model_with_mnist_images.py
Mediante este script se testea el modelo previamente entrenado por *train_model_with_mnist_images.py* con el set de datos de testing. 
Generará los correspondientes logs con información de error total y medio dentro de *logs/tester/mnist* junto con algunos archivos png de 
muestra de las imágenes generadas.

### tests
Contiene todos los archivos de tests unitarios y funcionales.

### utils
Contiene todas las clases que pueden ser reutilizadas a lo largo de todo el proyecto.

## Referencias

Z. Wei and X. Chen, “Deep learning schemes for full-wave nonlinear inverse scattering problems,” IEEE Transactions on Geoscience and Remote Sensing, 57 (4), pp. 1849-1860, 2019.

Z. Wei# and X. Chen, “Uncertainty Quantification in Inverse Scattering Problems with Bayesian Convolutional Neural Networks” IEEE Transactions on Antennas and Propagation, 10.1109/TAP.2020.3030974, 2020.

K. H. Jin, M. T. McCann, E. Froustey, and M. Unser, “Deep convolutional neural network for inverse problems in imaging,” IEEE Transactions on Image Processing, vol. 26, no. 9, pp. 4509–4522, 2017.

Deng, L., 2012. The mnist database of handwritten digit images for machine learning research. IEEE Signal Processing Magazine, 29(6), pp.141–142.
