# AI in a Storage System

Con este proyecto busco el crear un sistema de prediccion inteligente, capaz de identificar patrones de acceso a disco, para decidir que datos mover a almacenamiento frio, como ejemplo: (AWS Glacier) y cuales mantener en almacenamiento continuo activo.

El Dataset utilizado es **MSNFS Traces** (Microsoft Network File System), que son logs reales de operaciones de disco capturados de servidores de produccion de Microsoft en 2008. Obtenidos mediante IOTTA Repository.  Donde hay 29 millones de operaciones en total, divididos en 36 archivos CSV, con cada archivo teniendo 10 minutos de operaciones.
http://iotta.snia.org/traces/block-io/158?n=100&page=1


El preprocesado esta en: preprocesado.py, donde  realiza lo siguiente:

1:Leer los 36 CSVs y extraer operaciones de DiskRead y DiskWrite.

2:Agrupa las operaciones por archivo y ventana temporal, y calcula 8 features:

| Feature | Descripcion |
|---------|-------------|
| num_reads | Cantidad de lecturas en la ventana |
| num_writes | Cantidad de escrituras en la ventana |
| total_ops | Total de operaciones (reads + writes) |
| ratio_rw | Proporcion lecturas/escrituras |
| avg_io_size | Tamaño promedio de operacion en bytes |
| avg_elapsed | Tiempo promedio de cada operacion |
| std_elapsed | Desviacion estandar del tiempo |
| rango_tiempo | Dispersion temporal de los accesos |


3:Se aplica StandardScaler para normalizar todas las features a media 0 y desviacion estandar 1.

4:Se divide en 70% entrenamiento y 30% prueba.


