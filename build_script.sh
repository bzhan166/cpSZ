#!/bin/bash
src_dir=`pwd`

# build FTK for evaluation
mkdir -p ${src_dir}/external
cd ${src_dir}/external
git clone https://github.com/lxAltria/ftk.git
cd ftk
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=${src_dir}/external/ftk/install ..
make -j 4
make install

# build cpSZ
cd ${src_dir}
mkdir build
cd build
cmake ..
make -j 4
