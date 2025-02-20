import numpy as np 
import sys
import time 
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import pandas as pd
pd.options.display.float_format = '{:.2e}'.format
from datetime import datetime

sys.path.append('../code')
from tensornetwork.MPS import MPS
from tensornetwork.MPO import MPO
from tensornetwork.contraction import *
from tensornetwork.stopping import *
# from tensornetwork.util import * 
from util.plotting import plot_accuracy,plot_times2,plot_accuracy2,plot_times_reversed,plot_accuracy_reversed
from tensornetwork.incrementalqr import *




# ===========================================================================
# Benchmark Experiment 1 : Random Tensor Experiment (variable compressibility)
# ===========================================================================

def generate_baseline(N, m, a=-0.5, dtype=float):
    def random_tensor(*args, a=a, b=1, dtype=dtype):
        output = a + (b - a) * np.random.rand(*args).astype(dtype)
        output /= np.linalg.norm(output)
        return output

    print("Generating baseline contraction...")
    mpo = MPO.random_mpo(N, m, random_tensor=random_tensor,dtype=dtype)
    mps = MPS.random_mps(N, m, random_tensor=random_tensor,dtype=dtype)
    baseline = mps_mpo_blas(mps, mpo, stop=Cutoff(1e-15))
    return mpo, mps, baseline

def fixed_synth_tensor_experiment(mpo, mps, baseline, bond_dims, names, num_runs=1, a=-.5, b=1, highres=False, return_data=False,
                                  fit_sweeps=1, sketch_increment=2, sketch_dim=3):
    times = {name: [] for name in names}
    accs = {name: [] for name in names}
    std_times = {name: [] for name in names}
    std_accs = {name: [] for name in names}

    baseline.canonize()
    print(baseline.norm())
    
    column_names = ['Bond Dimension'] + [f'{name} {metric}' for name in names for metric in ['Mean Time', 'Time Std', 'Mean Accuracy', 'Accuracy Std']]
    results_df = pd.DataFrame(columns=column_names)
    display(results_df)

    for bond_dim in bond_dims:
        temp_times = {name: [] for name in names}
        temp_accs = {name: [] for name in names}

        for run in range(num_runs):
            for name in names:
                print(name)
                start = time.time()
                if name == 'naive':
                    result = mps_mpo_blas(mps, mpo, stop=FixedDimension(bond_dim), round_type="dass_blas")
                elif name == 'rand_then_orth':
                    result = mps_mpo_blas(mps, mpo, stop=FixedDimension(bond_dim), round_type="rand_then_orth_blas", final_round=False)
                elif name == 'nyst':
                    result = mps_mpo_blas(mps, mpo, stop=FixedDimension(bond_dim), round_type="nystrom_blas", final_round=False)
                elif name == 'random+oversample':
                    result = random_contraction_inc(mpo, mps, stop=FixedDimension(max(int(np.ceil(1.5*bond_dim)),bond_dim+10)), accuracychecks=False, 
                                                    finalround=FixedDimension(bond_dim), sketchincrement=sketch_increment, sketchdim=sketch_dim)
                elif name == 'random':
                    result = random_contraction_inc(mpo, mps, stop=FixedDimension(bond_dim), accuracychecks=False, 
                                                    finalround=None, sketchincrement=sketch_increment, sketchdim=sketch_dim)
                elif name == 'density':
                    result = density_matrix(mpo, mps, stop=FixedDimension(bond_dim))
                elif name == 'zipup':
                    result = zipup(mpo, mps, stop=FixedDimension(bond_dim), finalround=None,conditioning=True)
                elif name == 'fit':
                    result = fit(mpo, mps, max_sweeps=fit_sweeps, stop=FixedDimension(bond_dim))
                else:
                    print("Invalid algorithm choice for ", name, " review your inputted algorithm names")
                    return
                temp_times[name].append(time.time() - start)
                # print(baseline[0].shape,result[0].shape)
                # for t in result.tensors:
                #     print(t.shape)
                temp_accs[name].append((baseline - result).norm() / baseline.norm())

        # Compute the mean and std for this bond dimension
        new_row_data = {'Bond Dimension': bond_dim}
        for name in names:
            times_mean = np.mean(temp_times[name])
            times_std = np.std(temp_times[name])
            acc_mean = np.mean(temp_accs[name])
            acc_std = np.std(temp_accs[name])
            
            new_row_data[f'{name} Mean Time'] = times_mean
            new_row_data[f'{name} Time Std'] = times_std
            new_row_data[f'{name} Mean Accuracy'] = acc_mean
            new_row_data[f'{name} Accuracy Std'] = acc_std
            
            times[name].append(times_mean)
            std_times[name].append(times_std)
            accs[name].append(acc_mean)
            std_accs[name].append(acc_std)
        
        new_row_df = pd.DataFrame([new_row_data]) 
        results_df = pd.concat([results_df, new_row_df], ignore_index=True)
        clear_output(wait=True)
        display(results_df)

    # # Save data to a CSV file in the Figure2_data directory
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # file_name = f"figure2_data_{mps.N}_{mps.max_bond_dim()}_{timestamp}_{num_runs}.csv"
    # file_path = os.path.join('Figure2_data', file_name)
    # results_df.to_csv(file_path, index=False)
    # print(f"Data saved to {file_path}")
        
    if highres:
         # High resolution separate plots
        plt.figure(dpi=500)
        plot_times2(names, bond_dims, times, std_times)
        plt.tight_layout()
        plt.grid(True)
        plt.show()

        plt.figure(dpi=500)
        plot_accuracy2(names, bond_dims, accs, std_accs)
        plt.tight_layout()
        plt.grid(True)

        plt.show()
    else:
        # Standard multi-plot
        fig, axs = plt.subplots(1, 2, figsize=(12, 5)) 
        plot_times2(names, bond_dims, times, std_times, ax=axs[0])
        plot_accuracy2(names, bond_dims, accs, std_accs, ax=axs[1])
    if return_data:
        return times, std_times, accs, std_accs
def cutoff_synth_tensor_experiment(mpo,mps,baseline,cutoffs,names,a=-.5,b=1,highres=False,return_data=False,
                                   fit_sweeps=8,sketch_dim=10,sketch_increment=10):
    # baseline.canonize()
    # print(baseline.norm())
    
    times = {name: [] for name in names}
    accs = {name: [] for name in names}

    column_names = ['Cutoff Value'] + [f'{name} {metric}' for name in names for metric in ['Time', 'Accuracy']]  
    results_df = pd.DataFrame(columns=column_names)
    display(results_df)


    for cutoff in cutoffs:
        for name in names:
            start = time.time()
            if name == 'naive':
                result = mps_mpo_blas(mps, mpo, stop=Cutoff(cutoff),round_type = "dass_blas")
            elif name == 'random':
                result = random_contraction_inc(mpo, mps, stop=Cutoff(cutoff), accuracychecks=False, 
                                              finalround=None,sketchincrement=sketch_increment,sketchdim=sketch_dim)
            elif name == 'density':
                result = density_matrix(mpo, mps,stop=Cutoff(cutoff))
            elif name == 'zipup':
                result = zipup(mpo, mps, stop=Cutoff(cutoff),finalround=False,conditioning=True)
            elif name == 'fit':
                result = fit(mpo, mps, max_sweeps=fit_sweeps,stop=Cutoff(cutoff))
            else:
                print("Invalid algorithm choice for ", name, " review your inputted algorithm names")
                return
            
            times[name].append(time.time() - start)
            accs[name].append((baseline - result).norm() / baseline.norm())

        # Data Frame population
        new_row_data = {'Cutoff Value': cutoff}
        for name in names:
            new_row_data[f'{name} Time'] = times[name][-1]
            new_row_data[f'{name} Accuracy'] = accs[name][-1]
        new_row_df = pd.DataFrame([new_row_data])
        results_df = pd.concat([results_df, new_row_df], ignore_index=True)
        clear_output(wait=True)
        display(results_df)
        
    if highres:
         # High resolution separate plots
        plt.figure( dpi=400)
        plot_times_reversed(names, cutoffs, times)
        plt.tight_layout()
        plt.grid(True)
        plt.show()

        plt.figure( dpi=400)
        plot_accuracy_reversed(names, cutoffs, accs)
        plt.tight_layout()
        plt.grid(True)

        plt.show()
    else:
        # Standard multi-plot
        fig, axs = plt.subplots(1, 2, figsize=(12, 5)) 
        plot_times_reversed(names, cutoffs, times, ax=axs[0])
        plot_accuracy_reversed(names, cutoffs, accs, ax=axs[1])
    if return_data:
        return times,accs

# ===========================================================================
# Library Benchmark 0 : Incremental QR (Averaged )
# ===========================================================================

def incQr_benchmark(max_size=500, complex_matrices=False,log_scale=False):
    sizes = range(100, max_size, 50)  # Matrix sizes from 100 to maxsize with steps of 50
    results_df = pd.DataFrame(columns=['Matrix Size', 'C++ Time Mean', 'SciPy Time Mean', 'NumPy Time Mean']) 

    cpp_times_std = []
    scipy_times_std = []
    numpy_times_std = []

    for size in sizes:
        cpp_times = []
        scipy_times = []
        numpy_times = []
        
        if complex_matrices:
            A = np.random.randn(size, int(size / 50)) + 1j * np.random.randn(size, int(size / 50))
        else:
            A = np.random.randn(size, int(size / 50))
        
        for _ in range(5):
            # Timing C++ Implementation 
            start_time = time.time()
            iqr = IncrementalQR(A)  
            Q_cpp = iqr.get_q()
            cpp_times.append(time.time() - start_time)

            # Timing SciPy Implementation
            start_time = time.time()
            iqr = IncrementalQR(A, use_cpp_if_available=False) 
            Q_sp = iqr.get_q()
            scipy_times.append(time.time() - start_time)

            # Timing NumPy QR Decomposition
            start_time = time.time()
            Q_np, _ = np.linalg.qr(A, mode='reduced')
            numpy_times.append(time.time() - start_time)

        cpp_mean = np.mean(cpp_times)
        scipy_mean = np.mean(scipy_times)
        numpy_mean = np.mean(numpy_times)
        
        cpp_std = np.std(cpp_times)
        scipy_std = np.std(scipy_times)
        numpy_std = np.std(numpy_times)
        

        cpp_times_std.append(cpp_std)
        scipy_times_std.append(scipy_std)
        numpy_times_std.append(numpy_std)

        new_row_data = {
            'Matrix Size': size,
            'C++ Time Mean': cpp_mean,
            'SciPy Time Mean': scipy_mean,
            'NumPy Time Mean': numpy_mean
        }
        new_row_df = pd.DataFrame([new_row_data])
        results_df = pd.concat([results_df, new_row_df], ignore_index=True)
        
        clear_output(wait=True)
        display(results_df)

    # Convert results to numpy arrays for better compatibility with matplotlib
    matrix_sizes = results_df['Matrix Size'].values.astype(float)
    cpp_time_mean = results_df['C++ Time Mean'].values.astype(float)
    scipy_time_mean = results_df['SciPy Time Mean'].values.astype(float)
    numpy_time_mean = results_df['NumPy Time Mean'].values.astype(float)
    cpp_times_std = np.array(cpp_times_std, dtype=float)
    scipy_times_std = np.array(scipy_times_std, dtype=float)
    numpy_times_std = np.array(numpy_times_std, dtype=float)


    plt.figure(figsize=(10, 6))

    plt.plot(matrix_sizes, cpp_time_mean, label='C++ Incremental', marker='o', markersize=3)
    plt.fill_between(matrix_sizes, 
                    cpp_time_mean - cpp_times_std, 
                    cpp_time_mean + cpp_times_std, 
                    alpha=0.2)

    plt.plot(matrix_sizes, scipy_time_mean, label='SciPy Incremental', marker='s', markersize=3)
    plt.fill_between(matrix_sizes, 
                    scipy_time_mean - scipy_times_std, 
                    scipy_time_mean + scipy_times_std, 
                    alpha=0.2)

    plt.plot(matrix_sizes, numpy_time_mean, label='NumPy QR', marker='^', markersize=3)
    plt.fill_between(matrix_sizes, 
                    numpy_time_mean - numpy_times_std, 
                    numpy_time_mean + numpy_times_std, 
                    alpha=0.2)

    plt.xlabel('Matrix Size N x N/50')
    if log_scale:
        plt.yscale('log')
    plt.ylabel('Time (seconds)')
    plt.title('QR Decomposition Timing Comparison 5 run average')
    plt.legend()
    plt.grid(True)
    plt.show()
