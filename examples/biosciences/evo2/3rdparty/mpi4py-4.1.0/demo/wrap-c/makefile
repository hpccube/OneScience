include ../config.mk
.PHONY: default build test clean
default: build test clean

MODULE = helloworld
SOURCE = $(MODULE).c
TARGET = $(MODULE)$(EXT_SUFFIX)

$(TARGET): $(SOURCE)
	$(MPICC) $(MPI4PY_INCLUDE) $(CC_FLAGS_SHARED) -o $@ $<

build: $(TARGET)

test: build
	$(MPIEXEC_PYTHON) test.py

clean:
	$(RM) -r $(TARGET) __pycache__
