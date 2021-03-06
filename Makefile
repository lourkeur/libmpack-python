MPACK_VERSION ?= 1.0.5
MPACK_URL ?= https://github.com/tarruda/libmpack/archive/$(MPACK_VERSION).tar.gz

# Command that will download a file and pipe it's contents to stdout
FETCH ?= curl -L -o -
# Command that will gunzip/untar a file from stdin to the current directory,
# stripping one directory component
UNTGZ ?= tar xfz - --strip-components=1

all: build

mpack-src:
	dir="mpack-src"; \
	mkdir -p $$dir && cd $$dir && \
	$(FETCH) $(MPACK_URL) | $(UNTGZ)

clean:
	rm -rf mpack-src
	python setup.py clean

test: build
	python tests/test_doctest.py

build: mpack-src
	python setup.py build_ext --inplace

.PHONY: all build test clean
