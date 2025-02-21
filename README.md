# RandomMPSMPO
A randomized algorithm for the compressed MPS-MPO product

## Installation Instructions

---

### **System Requirements**
All experiments were run using:
- **LAPACK**: 3.12.0
- **OpenBLAS**: 0.3.28

### **Environment Setup**
This project requires a Conda environment defined in `environment.yml`. To create and activate it, run:

```bash
conda env create -f environment.yml
conda activate tensor2
```

### (Optional) Optimized Incremental QR Build
To enable the fastest version of the randomized MPO-MPS algorithm, you can build our custom C++ incremental QR implementation. This step may be more complex for Windows users.

With the environment activated, execute the setup script from the project root:

```bash
bash setup_QR.sh
```

If the build is successful, running the following command from the project root:

```bash
python code/tensornetwork/incrementalqr.py
```

should print:

```
Using C++ implementation for incQR
```

at the start of the program, confirming that the build was successful.

