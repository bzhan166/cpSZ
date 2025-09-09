# cpSZ-GPU
Critical point preserving compression for vector fields (GPU Implementation)

# Installation
cd src <br>
make <br>
Dependancies: ZSTD

# Run compression
./cpszg_2d ../data/uf.dat ../data/vf.dat 2400 3600 0.1 <br>
Decompressed data files are data/uf.dat.out and data/vf.dat.out

# Evaluation
cd .. <br>
./external/ftk/build/bin/ex_cp_2d 3600 2400 ./data/uf.dat ./data/vf.dat <br>