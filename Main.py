from DatasetHandler import DatasetHandler  # Asegúrate de usar el nombre correcto del archivo


class Main:
    def __init__(self):
        print("Inicializando el programa...")
        self.dataset_handler = DatasetHandler()  # Llama a tu clase DatasetHandler

    def run(self):
        print("Ejecutando el programa...")
        # Aquí puedes añadir el código para entrenar tu red neuronal o cualquier otra lógica
        print("Datos normalizados (X_train):")
        print(self.dataset_handler.X_train)
        print("Etiquetas (y_train):", self.dataset_handler.y_train)

if __name__ == "__main__":
    main_program = Main()  # Crea una instancia de la clase Main
    main_program.run()      # Llama al método run de la clase Main
