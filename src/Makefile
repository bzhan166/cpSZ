
all: cpsz cpsz3d cusz_lrzc cusz_lrzx

cusz_lrzc: lproto_c.o
cusz_lrzx: lproto_x.o

lproto_c.o: cusz_partial/lproto_c.cu
	nvcc cusz_partial/lproto_c.cu -c \
		-I$(HOME)/install/include/cusz \
		-Icusz_partial \
    -gencode arch=compute_75,code=sm_75 \
    -gencode arch=compute_80,code=sm_80 \
    -gencode arch=compute_86,code=sm_86 \
		-DPSZ_USE_CUDA=1 

lproto_x.o: cusz_partial/lproto_x.cu
	nvcc cusz_partial/lproto_x.cu -c \
		-I$(HOME)/install/include/cusz \
		-Icusz_partial \
    -gencode arch=compute_75,code=sm_75 \
    -gencode arch=compute_80,code=sm_80 \
    -gencode arch=compute_86,code=sm_86 \
		-DPSZ_USE_CUDA=1 

cpsz: sz_compress_cp_preserve_2d_gpu.cu cusz_lrzc cusz_lrzx
	nvcc -O3 --extended-lambda sz_compress_cp_preserve_2d_gpu.cu \
		-I../include \
		-Icusz_partial \
		-I$(HOME)/install/include/cusz \
		--expt-relaxed-constexpr \
		-DPSZ_USE_CUDA=1 \
    -gencode arch=compute_75,code=sm_75 \
    -gencode arch=compute_80,code=sm_80 \
    -gencode arch=compute_86,code=sm_86 \
		lproto_c.o \
		lproto_x.o \
		-o cpszg_2d

cpsz3d: sz_compress_cp_preserve_3d_gpu.cu cusz_lrzc cusz_lrzx
	nvcc -O3 --extended-lambda sz_compress_cp_preserve_3d_gpu.cu \
		-I../include \
		-Icusz_partial \
		-I$(HOME)/install/include/cusz \
		--expt-relaxed-constexpr \
		-DPSZ_USE_CUDA=1 \
    -gencode arch=compute_75,code=sm_75 \
    -gencode arch=compute_80,code=sm_80 \
    -gencode arch=compute_86,code=sm_86 \
		lproto_c.o \
		lproto_x.o \
		-o cpszg_3d

clean:
	rm -f cpszg_2d cpszg_3d lproto_c.o lproto_x.o