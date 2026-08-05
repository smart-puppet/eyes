CUDA_VER ?= 12.6
CC := g++
CFLAGS := -Wall -std=c++14 -shared -fPIC -Wno-error=deprecated-declarations
CFLAGS += -I/opt/nvidia/deepstream/deepstream/sources/includes
CFLAGS += -I/usr/local/cuda-$(CUDA_VER)/include

TARGET := libnvdsinfer_yolov8_parser.so
SRC := nvdsinfer_yolov8_parser.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $@ $^

clean:
	rm -f $(TARGET)
