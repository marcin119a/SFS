"""
Example usage of the SFS Python package.

This script demonstrates how to use the main functions of the SFS package
to find the set of feasible solutions from a non-negative matrix factorization.
"""
import numpy as np
import matplotlib.pyplot as plt
from sfs import NMFPois, sampleSFS, samplesToSVD


def main():
    """Main example function demonstrating SFS package usage."""
    
    print("SFS Package Example Usage")
    print("=" * 50)
    
    # Create example data matrix M (4 x 6)
    M = np.array([[20, 3, 24, 19, 2, 15],
                  [9, 14, 25, 30, 15, 10],
                  [30, 6, 12, 10, 11, 7],
                  [9, 27, 5, 11, 19, 15]], dtype=float)
    
    print(f"Data matrix M shape: {M.shape}")
    print("Data matrix M:")
    print(M)
    print()
    
    # Step 1: Create initial NMF solution
    print("Step 1: Creating initial NMF solution...")
    rank = 3
    initial_fit = NMFPois(M, rank)
    
    P = initial_fit['P']
    E = initial_fit['E']
    gkl = initial_fit['gkl']
    
    print(f"Initial P shape: {P.shape}")
    print(f"Initial E shape: {E.shape}")
    print(f"Initial GKLD: {gkl:.6f}")
    print()
    
    # Check approximation quality
    approximation = P @ E
    relative_error = np.mean(np.abs(M - approximation) / (M + 1e-10))
    print(f"Relative approximation error: {relative_error:.6f}")
    print()
    
    # Step 2: Find the set of feasible solutions (SFS)
    print("Step 2: Finding the set of feasible solutions...")
    sfs_result = sampleSFS(P, E, maxIter=10000, check=1000)
    
    print(f"Total iterations: {sfs_result['totalIter']}")
    print(f"Final average change: {sfs_result['avgChangeFinal']:.2e}")
    print()
    
    # Step 3: Display results
    print("Step 3: SFS Results Summary")
    print("-" * 30)
    
    P_min = sfs_result['Pminimum']
    P_max = sfs_result['Pmaximum']
    E_min = sfs_result['Eminimum']
    E_max = sfs_result['Emaximum']
    
    print("P matrix ranges (min, max):")
    for i in range(rank):
        print(f"  Column {i+1}:")
        for j in range(P.shape[0]):
            print(f"    Row {j+1}: [{P_min[i,j]:.4f}, {P_max[i,j]:.4f}]")
    
    print("\nE matrix ranges (min, max):")
    for i in range(rank):
        print(f"  Row {i+1}:")
        for j in range(E.shape[1]):
            print(f"    Column {j+1}: [{E_min[i,j]:.4f}, {E_max[i,j]:.4f}]")
    
    # Step 4: Optional - Convert to SVD representation
    print("\nStep 4: Converting to SVD representation...")
    
    # Get some results from the last check
    P_last = sfs_result['P_lastCheckResults']
    E_last = sfs_result['E_lastCheckResults']
    
    # Convert to SVD representation
    svd_result = samplesToSVD(P_last, E_last, rank)
    
    P_points = svd_result['P.points']
    E_points = svd_result['E.points']
    
    print(f"SVD P points shape: {P_points.shape}")
    print(f"SVD E points shape: {E_points.shape}")
    
    # Step 5: Create visualization (if matplotlib is available)
    try:
        create_visualization(sfs_result, rank)
        print("\nVisualization saved as 'sfs_results.png'")
    except ImportError:
        print("\nMatplotlib not available - skipping visualization")
    
    print("\nExample completed successfully!")


def create_visualization(sfs_result, rank):
    """Create visualization of SFS results."""
    fig, axes = plt.subplots(rank, 2, figsize=(12, 3*rank))
    if rank == 1:
        axes = axes.reshape(1, -1)
    
    P_min = sfs_result['Pminimum']
    P_max = sfs_result['Pmaximum']
    E_min = sfs_result['Eminimum']
    E_max = sfs_result['Emaximum']
    
    # Plot SFS for P
    for i in range(rank):
        ax = axes[i, 0]
        x = np.arange(P_min.shape[1])
        ax.bar(x, P_min[i, :], alpha=0.7, color='grey', label='Minimum')
        ax.bar(x, P_max[i, :] - P_min[i, :], bottom=P_min[i, :], 
               alpha=0.7, color='red', label='Range')
        ax.set_title(f'SFS for column {i+1} in P')
        ax.set_xlabel('Row index')
        ax.set_ylabel('Value')
        ax.legend()
    
    # Plot SFS for E
    for i in range(rank):
        ax = axes[i, 1]
        x = np.arange(E_min.shape[1])
        ax.bar(x, E_min[i, :], alpha=0.7, color='grey', label='Minimum')
        ax.bar(x, E_max[i, :] - E_min[i, :], bottom=E_min[i, :], 
               alpha=0.7, color='red', label='Range')
        ax.set_title(f'SFS for row {i+1} in E')
        ax.set_xlabel('Column index')
        ax.set_ylabel('Value')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('sfs_results.png', dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    main()
