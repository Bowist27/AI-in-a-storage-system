import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ========================================
# PASO 1: Leer y parsear los archivos de traces
# ========================================

carpeta = "Traces"
archivos = sorted([f for f in os.listdir(carpeta) if f.endswith(".csv")])

print(f"Se encontraron {len(archivos)} archivos de traces")

# usar todos los archivos disponibles
archivos_usar = archivos
print(f"Usando {len(archivos_usar)} archivos")

todas_las_operaciones = []

for i, archivo in enumerate(archivos_usar):
    ruta = os.path.join(carpeta, archivo)
    print(f"Leyendo archivo {i+1}/{len(archivos_usar)}: {archivo}...")

    with open(ruta, "r", encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()
            if linea.startswith("DiskRead,") or linea.startswith("DiskWrite,"):
                partes = [p.strip() for p in linea.split(",")]
                if len(partes) >= 15:
                    try:
                        evento = partes[0]
                        timestamp = int(partes[1])
                        io_size = int(partes[6], 16)
                        elapsed = int(partes[7])
                        disk_num = int(partes[8])
                        nombre = partes[14].strip('"')

                        todas_las_operaciones.append({
                            "ventana": i,
                            "evento": evento,
                            "timestamp": timestamp,
                            "io_size": io_size,
                            "elapsed_time": elapsed,
                            "disco": disk_num,
                            "archivo": nombre
                        })
                    except:
                        continue

print(f"\nTotal de operaciones parseadas: {len(todas_las_operaciones)}")

df = pd.DataFrame(todas_las_operaciones)
print(f"Shape del dataframe: {df.shape}")
print(df.head(10))

# ========================================
# PASO 2: Crear features por archivo y ventana
# ========================================

print("\n" + "="*50)
print("PASO 2: Generando features por archivo/ventana")
print("="*50)

grupos = df.groupby(["archivo", "ventana"])

features_lista = []

for (archivo, ventana), grupo in grupos:
    reads = grupo[grupo["evento"] == "DiskRead"]
    writes = grupo[grupo["evento"] == "DiskWrite"]

    num_reads = len(reads)
    num_writes = len(writes)
    total_ops = num_reads + num_writes

    if num_writes > 0:
        ratio_rw = num_reads / num_writes
    else:
        ratio_rw = num_reads

    avg_io_size = grupo["io_size"].mean()
    avg_elapsed = grupo["elapsed_time"].mean()
    std_elapsed = grupo["elapsed_time"].std()
    if pd.isna(std_elapsed):
        std_elapsed = 0

    rango_tiempo = grupo["timestamp"].max() - grupo["timestamp"].min()

    features_lista.append({
        "archivo": archivo,
        "ventana": ventana,
        "num_reads": num_reads,
        "num_writes": num_writes,
        "total_ops": total_ops,
        "ratio_rw": ratio_rw,
        "avg_io_size": avg_io_size,
        "avg_elapsed": avg_elapsed,
        "std_elapsed": std_elapsed,
        "rango_tiempo": rango_tiempo,
        "disco": grupo["disco"].iloc[0]
    })

df_features = pd.DataFrame(features_lista)
print(f"Se generaron {len(df_features)} registros de features")
print(df_features.head(10))

# ========================================
# PASO 3: Preprocesado y escalamiento
# ========================================

print("\n" + "="*50)
print("PASO 3: Preprocesado de datos")
print("="*50)

cols_numericas = ["num_reads", "num_writes", "total_ops", "ratio_rw",
                  "avg_io_size", "avg_elapsed", "std_elapsed", "rango_tiempo"]

# revisar valores nulos
print("\nValores nulos por columna:")
print(df_features[cols_numericas].isnull().sum())

df_features[cols_numericas] = df_features[cols_numericas].fillna(0)

# aplicar StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_features[cols_numericas])

df_scaled = pd.DataFrame(X_scaled, columns=cols_numericas)
df_scaled["archivo"] = df_features["archivo"].values
df_scaled["ventana"] = df_features["ventana"].values

print("\nEstadisticas despues del escalamiento:")
print(df_scaled[cols_numericas].describe())

# ========================================
# PASO 4: Separacion train/test
# ========================================

print("\n" + "="*50)
print("PASO 4: Separacion train/test")
print("="*50)

X = df_scaled[cols_numericas].values

X_train, X_test = train_test_split(X, test_size=0.3, random_state=42)

print(f"Set de entrenamiento: {X_train.shape}")
print(f"Set de prueba: {X_test.shape}")

# ========================================
# PASO 5: Guardar datos procesados
# ========================================

print("\n" + "="*50)
print("PASO 5: Guardando datos")
print("="*50)

df_features.to_csv("dataset_features.csv", index=False)
df_scaled.to_csv("dataset_escalado.csv", index=False)

train_df = pd.DataFrame(X_train, columns=cols_numericas)
train_df.to_csv("train_set.csv", index=False)

test_df = pd.DataFrame(X_test, columns=cols_numericas)
test_df.to_csv("test_set.csv", index=False)

print("Archivos guardados:")
print("  - dataset_features.csv")
print("  - dataset_escalado.csv")
print("  - train_set.csv")
print("  - test_set.csv")

# ========================================
# PASO 6: Graficas
# ========================================

print("\n" + "="*50)
print("PASO 6: Generando graficas")
print("="*50)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# top 15 archivos
top_archivos = df_features.groupby("archivo")["total_ops"].sum().sort_values(ascending=False).head(15)
axes[0].barh(range(len(top_archivos)), top_archivos.values, color="steelblue")
axes[0].set_yticks(range(len(top_archivos)))
axes[0].set_yticklabels(top_archivos.index, fontsize=7)
axes[0].set_title("Top 15 Archivos por Total de Operaciones")
axes[0].set_xlabel("Total de operaciones I/O")

# reads vs writes
axes[1].scatter(df_features["num_reads"], df_features["num_writes"],
                  alpha=0.6, s=20, color="steelblue")
axes[1].set_xlabel("Numero de Reads")
axes[1].set_ylabel("Numero de Writes")
axes[1].set_title("Reads vs Writes")

# histograma
axes[2].hist(df_features["total_ops"], bins=30, color="steelblue", edgecolor="black")
axes[2].set_title("Distribucion de Total de Operaciones")
axes[2].set_xlabel("Total de operaciones")
axes[2].set_ylabel("Frecuencia")

plt.tight_layout()
plt.savefig("graficas_preprocesado.png", dpi=150)
plt.close()

print("Grafica guardada como 'graficas_preprocesado.png'")
print("\nPreprocesado completado!")

