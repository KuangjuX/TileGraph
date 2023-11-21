
build:
	@mkdir build
	@cd build && cmake .. && make -j8

test: build
	@cd build && make test

clean:
	@rm -rf build