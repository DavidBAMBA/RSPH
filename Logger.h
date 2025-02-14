#ifndef LOGGER_H
#define LOGGER_H

#include <fstream>
#include <iostream>
#include <string>
#include <cmath>
#include <memory>

class Logger {
public:
    Logger(const std::string &filename) {
        logFile.open(filename, std::ios::out);
        if (!logFile) {
            std::cerr << "Error opening log file: " << filename << std::endl;
        }
    }

    ~Logger() {
        if (logFile.is_open())
            logFile.close();
    }

    // Registra un mensaje (se escribe tanto en std::cerr como en el archivo)
    void log(const std::string &message) {
        std::cerr << message;
        if (logFile.is_open()) {
            logFile << message;
            logFile.flush();  // Fuerza a escribir inmediatamente
        }
    }

private:
    std::ofstream logFile;
};

// Función inline para loguear errores (evita múltiples definiciones)
inline void logError(const std::string& context, const std::string& message, double value = NAN) {
    std::string msg = "[" + context + "] Error: " + message;
    if (!std::isnan(value)) {
        msg += " (valor: " + std::to_string(value) + ")";
    }
    msg += "\n";
    std::cerr << msg;
    extern std::unique_ptr<Logger> g_logger; // Declaración del logger global definido en Globals.h
    if (g_logger) {
        g_logger->log(msg);
    }
}

#endif // LOGGER_H
