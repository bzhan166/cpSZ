
all: cpsz cusz_lrzc cusz_lrzx

cpsz: sz_compress_cp_preserve_2d_gpu.cu cusz_lrzc
	nvcc sz_compress_cp_preserve_2d_gpu.cu \
		-I../include \
		-Icusz_partial \
		-I$(HOME)/install/include/cusz \
		--expt-relaxed-constexpr \
		-DPSZ_USE_CUDA=1 \
		lproto_c.o \
		-o cpszg

cusz_lrzc: lproto_c.o
cusz_lrzx: lproto_x.o

lproto_c.o: cusz_partial/lproto_c.cu
	nvcc cusz_partial/lproto_c.cu -c \
		-I$(HOME)/install/include/cusz \
		-Icusz_partial \
		-DPSZ_USE_CUDA=1 

lproto_x.o: cusz_partial/lproto_x.cu
	nvcc cusz_partial/lproto_x.cu -c \
		-I$(HOME)/install/include/cusz \
		-Icusz_partial \
		-DPSZ_USE_CUDA=1 

clean:
	rm -f cpszg lproto_c.o lproto_x.o