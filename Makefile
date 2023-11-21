
build:
	@mkdir build && cd build  
	@cmake .. && make -j8

test:
	@cd build && make test