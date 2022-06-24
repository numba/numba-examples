NUMBA_INCLUDE_DIR = $(shell python -c 'import os, numba; print(os.path.dirname(os.path.dirname(numba.__file__)))')

libextarray.so: extarray.c
	$(CC) -I$(NUMBA_INCLUDE_DIR) -shared -o libextarray.so extarray.c

test: libextarray.so
	pytest extarray.py

all: test

clean:
	rm libextarray.so
