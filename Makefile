# Makefile para el proyecto SPH Shock Tube

# Compilador y banderas
CXX = g++ -fopenmp
CXXFLAGS = -std=c++17 -Wall -O2

# Directorio de los archivos fuente y de encabezado
SRC_DIR = .
INC_DIR = .

# Archivos fuente
SRCS = $(wildcard $(SRC_DIR)/*.cpp)

# Archivos objeto
OBJS = $(SRCS:.cpp=.o)

# Ejecutable
TARGET = simulacion

# Regla por defecto
all: $(TARGET)

# Regla para el ejecutable
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS)

# Regla para los archivos objeto
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -I$(INC_DIR) -c $< -o $@

# Limpiar archivos objeto y ejecutable
clean:
	rm -f $(OBJS) $(TARGET) 
	rm -rf outputs plots
