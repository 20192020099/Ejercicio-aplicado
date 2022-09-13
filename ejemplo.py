# Cargamos las paqueterías
import pandas_datareader as pdr
from datetime import datetime
import tensorflow as tf
import matplotlib.pyplot as plt

# Cargamos los datos
Petroleo = pdr.DataReader("CL=F", "yahoo", start='2021-05-10', end='2022-05-10')
Oro = pdr.DataReader("GC=F", "yahoo", start='2021-05-10', end='2022-05-10')

# Visualizamos los datos a usar
print (Petroleo.Close, Oro.Close) 

# creamos una capa densa para conectar los datos de entrada con los de salida
capa = tf.keras.layers.Dense(units = 1, input_shape=[1])

# Creamos un modelo sequencial para este caso
# Utilizamos una función de pérdida de error cuadrático medio
# Usamos el optimizador de Adamax para que el modelo mejore en lugar de empeorar conforme se entrena

modelo = tf.keras.Sequential([capa])
modelo.compile(
  optimizer=tf.keras.optimizers.Adamax(0.1),
  loss="mean_squared_error"
)

# Entrenamos el modelo. Le decimos a la función fit que lo haga 100 veces con epoch. 
historico = modelo.fit(Petroleo.Close, Oro.Close, epochs=100, verbose=False)

plt.xlabel("Número de pruebas")
## Text(0.5, 0, 'Número de pruebas')
plt.ylabel("Magnitud de pérdida")
## Text(0, 0.5, 'Magnitud de pérdida')
plt.plot(historico.history["loss"])
## [<matplotlib.lines.Line2D object at 0x7fac33ea1e48>]
plt.show()

print("Si el precio del petroleo es de 50 dólares el barril, entonces el precio del oro tenderá a ser de", modelo.predict([50]), "dolares la onza")
