# Randomized MPS-MPO Contraction
![image](https://github.com/user-attachments/assets/a5b459df-9637-47fc-ab68-07b71f9da004)


## A Randomized Algorithm for the Compressed MPS-MPO Product
<DESCRIPTION> <PAPER LINK> <RESOURCES>
---

## **Installation Instructions**

### **System Requirements**
This project has been tested with the following dependencies:

- **LAPACK**: `3.12.0`
- **OpenBLAS**: `0.3.28`

Ensure these libraries are installed and available in your environment for optimal performance.

---

### **Environment Setup**
To set up the required Conda environment, use the provided `environment.yml` file. Run the following commands from the project root:

```bash
conda env create -f environment.yml
conda activate randomTensor
```

This will create and activate the Conda environment named `tensor2` with all necessary dependencies.

---

### **(Optional) Optimized Incremental QR Build**
For optimal performance, we provide a custom C++ implementation of the incremental QR decomposition. If you choose not to build it, a Python version written in  `scipy` will be used (which is slower).

#### **Building the Optimized Incremental QR**
With the Conda environment activated, run the following command from the project root:

```bash
bash setup_QR.sh
```

#### **Verifying a Successful Build**
After building, you can verify that the optimized C++ implementation is being used by running:

```bash
python code/tensornetwork/incrementalqr.py
```

If the build was successful, you should see the following message at the start of the output:

```
Using C++ implementation for incQR
```

If this message does not appear, the build may have failed, and the default Python implementation will be used instead.

---

### **Notes for Windows Users**
Building the optimized incremental QR decomposition may require additional configuration of `cmake` and a compatible C++ compiler. Ensure you have a properly configured build system before proceeding.

---

### **Support & Contributions**
If you encounter issues or have suggestions, feel free to open an issue or contribute to the project.

---
