
build:
	@mkdir build
	@cd build && cmake .. && make -j8

test: build
	@cd build && make test

example:
	g++ examples/$(EXAMPLE).cpp -o build/$(EXAMPLE) -Iinclude -Lbuild/ -ltilegraph

clean:
	@rm -rf build