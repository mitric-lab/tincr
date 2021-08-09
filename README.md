<div align="left">
  <img src="https://github.com/hochej/tincr/blob/master/tincr.png" height="120"/>
</div>


## Install
##### The features of the program are currently limited! 
To be able to compile and run the __tincr__ program it is necessary to have the Intel MKL libary
installed, as the linear algebra operations are based on the MKL LAPACK and BLAS implementations. 
You can just download the MKL library from the [Intel webpage](https://software.intel.com/content/www/us/en/develop/articles/oneapi-standalone-components.html) and after installation make sure that the 
enviroment variables are set.  
Since the end of 2020/beginning of 2021, MKL was made part of Intel's oneAPI toolkits. MKL can be installed via the link as before. 
However, the environment variable $MKLROOT must now be set with the `setvars.sh` script, which is located in the root
installation directory of MKL. In the case of the older versions you have to execute the `mklvars.sh` script that is located
in the installation directory of the MKL Library. 
```bash
source /path/to/MKL/mklvars.sh intel64
```  
Make sure that the environment variable `$MKLROOT` was set.  
Furthermore, you need Open-SSL as some used Rust libraries depend on it. You can follow the guide 
shown [here](https://www.howtoforge.com/tutorial/how-to-install-openssl-from-source-on-linux/) to install Open-SSL on 
linux.  
Of course you also need Rust itself. This is straightforward to install and explained in 
detail on the [official site](https://www.rust-lang.org/tools/install). Furthermore, you need the [Rusty-FITPACK](https://github.com/mitric-lab/Rusty-FITPACK) ([see Documentation for details](http://jhoche.de/Rusty-FITPACK/rusty_fitpack/)) crate
for the spline interpolation. This can be cloned from the Github repository and installed in the same way.

Then just clone the repository to your local machine
```bash
git clone https://github.com/hochej/tincr.git
```
Go into the new directory
```bash
cd tincr
```
and build the executable with the package manager Cargo
```bash
cargo build --release
```
The option `--release` enables all optimization during the build and ensures fast runtimes, but can
result in very long compile times. If you want to compile often e.g. in the case of debugging, then 
it makes sense to just execute
```bash
cargo build
``` 
To be able to execute the `tincr` programm you should set `TINCR_SRC_DIR` to the installation directory and you 
can add the binary path to your `PATH` environment variable.

### Example installation
This example shows the installation on a Debian machine as a local user: 
```bash
source /opt/local/intel/compilers_and_libraries_2019.4.243/linux/mkl/bin/mklvars.sh intel64
cd $HOME/software
git clone https://github.com/mitric-lab/Rusty-FITPACK.git
git clone https://github.com/mitric-lab/tincr
cd tincr
```
Update the path to the Rusty-Fitpack directory in `Cargo.toml`
```
cargo build --release
export TINCR_SRC_DIR="$HOME/software/tincr"
export PATH=$PATH:$TINCR_SRC_DIR/target/release
```




