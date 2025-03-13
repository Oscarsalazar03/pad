import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.stats import gaussian_kde

class Actividad_2:
    def __init__(self):
        self.resultados = []
        self.output_dir = "src/actividad-2"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def ejercicio_1(self):
        array = np.arange(10, 30)
        self.resultados.append(("Ejercicio 1", str(array.tolist())))
        return array
    
    def ejercicio_2(self):
        matriz = np.ones((10, 10))
        suma = matriz.sum()
        self.resultados.append(("Ejercicio 2", str(suma)))
        return suma
    
    def ejercicio_3(self):
        array1 = np.random.randint(1, 11, 5)
        array2 = np.random.randint(1, 11, 5)
        producto = array1 * array2
        self.resultados.append(("Ejercicio 3", str(producto.tolist())))
        return producto
    
    def ejercicio_4(self):
        matriz = np.fromfunction(lambda i, j: i + j, (4, 4))
        inversa = np.linalg.pinv(matriz)
        self.resultados.append(("Ejercicio 4", str(inversa.tolist())))
        return inversa
    
    def ejercicio_5(self):
        array = np.random.rand(100)
        max_index = np.argmax(array)
        min_index = np.argmin(array)
        self.resultados.append(("Ejercicio 5 - Máximo", str(max_index)))
        self.resultados.append(("Ejercicio 5 - Mínimo", str(min_index)))
        return max_index, min_index
    
    def ejercicio_6(self):
        array1 = np.ones((3, 1))
        array2 = np.ones((1, 3))
        resultado = array1 + array2
        self.resultados.append(("Ejercicio 6", str(resultado.tolist())))
        return resultado
    
    def ejercicio_7(self):
        matriz = np.random.randint(1, 10, (5, 5))
        submatriz = matriz[1:3, 1:3]
        self.resultados.append(("Ejercicio 7", str(submatriz.tolist())))
        return submatriz
    
    def ejercicio_8(self):
        array = np.zeros(10)
        array[3:7] = 5
        self.resultados.append(("Ejercicio 8", str(array.tolist())))
        return array
    
    def ejercicio_9(self):
        matriz = np.random.randint(1, 10, (3, 3))
        invertida = matriz[::-1]
        self.resultados.append(("Ejercicio 9", str(invertida.tolist())))
        return invertida
    
    def ejercicio_10(self):
        array = np.random.rand(10)
        seleccionados = array[array > 0.5]
        self.resultados.append(("Ejercicio 10", str(seleccionados.tolist())))
        return seleccionados
    
    def ejercicio_11(self):
        x = np.random.rand(100)
        y = np.random.rand(100)
        plt.scatter(x, y, color='blue', alpha=0.5)
        plt.xlabel("Eje X")
        plt.ylabel("Eje Y")
        plt.title("Gráfico de Dispersión")
        plt.grid(True)
        grafico_path = os.path.join(self.output_dir, "grafico_dispersion.png")
        plt.savefig(grafico_path)  # Guardar el gráfico en la carpeta especificada
        plt.show()
        self.resultados.append(("Ejercicio 11", "Gráfico guardado en " + grafico_path))

    def ejercicio_12(self):
        x = np.linspace(-2 * np.pi, 2 * np.pi, 100)
        y_sin = np.sin(x)
        ruido = np.random.normal(0, 0.1, size=x.shape)
        y_noisy = y_sin + ruido
            
        plt.scatter(x, y_noisy, color='red', alpha=0.5, label='sin(x) + ruido')
        plt.plot(x, y_sin, color='blue', label='sin(x)')
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Gráfico de Dispersión con Ruido Gaussiano")
        plt.legend()
        plt.grid(True)
            
        grafico_path = os.path.join(self.output_dir, "grafico_sin_ruido.png")
        plt.savefig(grafico_path)
        plt.show()
        self.resultados.append(("Ejercicio 12", "Gráfico guardado en " + grafico_path))

    def ejercicio_13(self):
        x = np.linspace(-2 * np.pi, 2 * np.pi, 100)
        y = np.linspace(-2 * np.pi, 2 * np.pi, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.cos(X) + np.sin(Y)
        
        plt.contourf(X, Y, Z, cmap='viridis')
        plt.colorbar()
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Gráfico de Contorno de z = cos(x) + sin(y)")
        
        grafico_path = os.path.join(self.output_dir, "grafico_contorno.png")
        plt.savefig(grafico_path)
        plt.show()
        self.resultados.append(("Ejercicio 13", "Gráfico guardado en " + grafico_path)) 

    def ejercicio_14(self):
        x = np.random.rand(1000)
        y = np.random.rand(1000)
        xy = np.vstack([x, y])
        densidad = gaussian_kde(xy)(xy)
        
        plt.scatter(x, y, c=densidad, cmap='plasma', alpha=0.5)
        plt.colorbar(label='Densidad')
        plt.xlabel("Eje X")
        plt.ylabel("Eje Y")
        plt.title("Gráfico de Dispersión con Densidad de Puntos")
        
        grafico_path = os.path.join(self.output_dir, "grafico_densidad.png")
        plt.savefig(grafico_path)
        plt.show()
        self.resultados.append(("Ejercicio 14", "Gráfico guardado en " + grafico_path)) 

    def ejercicio_15(self):
        x = np.linspace(-2 * np.pi, 2 * np.pi, 100)
        y = np.sin(x) + np.random.normal(0, 0.1, size=x.shape)
        X, Y = np.meshgrid(x, y)
        Z = np.sin(X) + np.cos(Y)
        
        plt.contourf(X, Y, Z, cmap='plasma')
        plt.colorbar()
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Gráfico de Contorno Lleno de sin(x) + cos(y)")
        
        grafico_path = os.path.join(self.output_dir, "grafico_contorno_lleno.png")
        plt.savefig(grafico_path)
        plt.show()
        self.resultados.append(("Ejercicio 15", "Gráfico guardado en " + grafico_path)) 

    def ejercicio_16(self):
        x = np.linspace(-2 * np.pi, 2 * np.pi, 100)
        y_sin = np.sin(x)
        ruido = np.random.normal(0, 0.1, size=x.shape)
        y_noisy = y_sin + ruido

        plt.scatter(x, y_noisy, color='red', alpha=0.5, label='$sin(x) + ruido$')
        plt.plot(x, y_sin, color='blue', label='$sin(x)$')
        plt.xlabel("$Eje X$")
        plt.ylabel("$Eje Y$")
        plt.title("$Gráfico de Dispersión$")
        plt.legend()
        plt.grid(True)

        grafico_path = os.path.join(self.output_dir, "grafico_sin_ruido_etiquetas.png")
        plt.savefig(grafico_path)
        plt.show()
        self.resultados.append(("Ejercicio 16", "Gráfico guardado en " + grafico_path)) 

    def ejercicio_17(self):
        data = np.random.normal(0, 1, 1000)
        plt.hist(data, bins=30, alpha=0.7, color='blue', edgecolor='black')
        plt.xlabel("Valor")
        plt.ylabel("Frecuencia")
        plt.title("Histograma de distribución normal")
        plt.grid(True)

        grafico_path = os.path.join(self.output_dir, "histograma.png")
        plt.savefig(grafico_path)
        plt.show()
        self.resultados.append(("Ejercicio 17", "Gráfico guardado en " + grafico_path)) 

    def ejercicio_18(self):
        data = np.random.normal(0, 1, 1000)
        bins_values = [10, 30, 50]
        
        for bins in bins_values:
            plt.hist(data, bins=bins, alpha=0.7, color='blue', edgecolor='black')
            plt.xlabel("Valor")
            plt.ylabel("Frecuencia")
            plt.title(f"Histograma con {bins} bins")
            plt.grid(True)
            
            grafico_path = os.path.join(self.output_dir, f"histograma_{bins}_bins.png")
            plt.savefig(grafico_path)
            plt.show()
            self.resultados.append((f"Ejercicio 18 - {bins} bins", f"Gráfico guardado en {grafico_path}"))

    def ejercicio_19(self):
        data = np.random.normal(0, 1, 1000)
        plt.hist(data, bins=30, alpha=0.7, color='blue', edgecolor='black')
        plt.axvline(data.mean(), color='red', linestyle='dashed', linewidth=2, label=f'Media: {data.mean():.2f}')
        plt.xlabel("Valor")
        plt.ylabel("Frecuencia")
        plt.title("Histograma con línea de media")
        plt.legend()
        plt.grid(True)
        grafico_path = os.path.join(self.output_dir, "histograma_media.png")
        plt.savefig(grafico_path)
        plt.show()
        self.resultados.append(("Ejercicio 19", "Gráfico guardado en " + grafico_path))  

    def ejercicio_20(self):
        data1 = np.random.normal(0, 1, 1000)
        data2 = np.random.normal(2, 1.5, 1000)
        plt.hist(data1, bins=30, alpha=0.5, color='blue', label='Dataset 1')
        plt.hist(data2, bins=30, alpha=0.5, color='red', label='Dataset 2')
        plt.xlabel("Valor")
        plt.ylabel("Frecuencia")
        plt.title("Histogramas Superpuestos")
        plt.legend()
        plt.grid(True)
        grafico_path = os.path.join(self.output_dir, "histogramas_superpuestos.png")
        plt.savefig(grafico_path)
        plt.show()
        self.resultados.append(("Ejercicio 20", "Gráfico guardado en " + grafico_path))                               
    
    def ejecutar_todos(self):
        for i in range(1, 21):
            getattr(self, f'ejercicio_{i}')()
        
        resultado_path = os.path.join(self.output_dir, "actividad_2.xlsx")
        df = pd.DataFrame(self.resultados, columns=["# Ejercicio", "Valor"])
        df.to_excel(resultado_path, index=False)
        print(f"Ejercicios guardados en {resultado_path}")

# Crear instancia y ejecutar todo
actividad = Actividad_2()
actividad.ejecutar_todos()