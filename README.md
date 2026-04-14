# cpSZ-GPU
Critical point preserving compression for vector fields (GPU Implementation)

# Installation
Intall cpu verison cpsz <br>
Dependancies: ZSTD <br>
sh build_script.sh <br>
<br>
Install gpu verison cpsz <br> 
cd src <br>
make <br>

# Run compression
./cpszg_2d ../data/uf.dat ../data/vf.dat 2400 3600 0.1 <br>
Decompressed data files are data/uf.dat.out and data/vf.dat.out

# Evaluation
cd .. <br>
./build/build/bin/cp_extraction_2d data/uf.dat data/vf.dat 3600 2400 <br>
