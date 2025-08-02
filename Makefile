# =============================
# Variables de carpeta
# =============================
SRC_DIR    := src
CORE_DIR   := $(SRC_DIR)/core
IC_DIR     := $(SRC_DIR)/ic
BNDRY_DIR  := $(SRC_DIR)/boundaries
MAIN_DIR   := $(SRC_DIR)/main
INCLUDE_DIR := include

# =============================
# Archivos fuente
# =============================
CORE_SRCS   := $(wildcard $(CORE_DIR)/*.cpp)
IC_SRCS     := $(wildcard $(IC_DIR)/*.cpp)
BNDRY_SRCS  := $(wildcard $(BNDRY_DIR)/*.cpp)
MAIN_SRCS   := $(wildcard $(MAIN_DIR)/*.cpp)

SRCS := $(CORE_SRCS) $(IC_SRCS) $(BNDRY_SRCS) $(MAIN_SRCS)
OBJS := $(SRCS:.cpp=.o)

# =============================
# Compilador
# =============================
CXX := g++
CXXFLAGS := -std=c++17 -O3 -Wall -Wextra -fopenmp \
    -I$(CORE_DIR) \
    -I$(IC_DIR) \
    -I$(BNDRY_DIR) \
    -I$(MAIN_DIR) \
    -I$(INCLUDE_DIR)

# =============================
# Target principal
# =============================
TARGET := rsph

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(OBJS) -o $@ $(CXXFLAGS)

# =============================
# Limpieza
# =============================
clean:
	rm -f $(OBJS) $(TARGET)
