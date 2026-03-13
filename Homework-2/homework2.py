# from numba import cuda
# import numpy as np
# print ( cuda.gpus )

# Launch the kernel (1 block, 1 thread)
# simple_kernel[1, 1]()
import numpy as np
from numba import cuda
# Define the CUDA kernel
@cuda.jit
def add_kernel (a , b , c ) :
    # Get the global index of the current thread
    # grid (1) returns a tuple (idx ,). We take the first element .
    idx = cuda . grid (1)
    # Check bounds to ensure we don ’t access memory outside the array
    if idx < a.size :
        c[idx] = a [ idx ] + b [ idx ]
    #     c[0] = a [ idx ] + b [ idx ]
    #     print( c[0] )
# --- Host Code ( CPU) ---
n = 1000000
# Create data on the host ( CPU)
a_host = np . random . randn ( n ) . astype ( np . float32 )
b_host = np . random . randn ( n ) . astype ( np . float32 )
c_host = np . empty_like ( a_host )
# 1. Allocate memory on the device (GPU)
a_device = cuda.to_device( a_host )
b_device = cuda.to_device( b_host )
c_device = cuda.device_array_like( a_host )
# 2. Configure the kernel launch parameters
threads_per_block = 256
# Calculate how many blocks we need to cover the entire array
blocks_per_grid = ( a_device . size + ( threads_per_block - 1) ) // threads_per_block
#blocks_per_grid = 10000 # For triggering mem error
print( f" Launching { blocks_per_grid } blocks with { threads_per_block } threads each .")
# 3. Launch the kernel on the GPU
add_kernel[ blocks_per_grid , threads_per_block ]( a_device , b_device , c_device )
cuda.synchronize()
# 4. Copy the result back to the host
c_device.copy_to_host ( c_host )
# Verify the result
if np.allclose ( c_host , a_host + b_host ) :
    print (" Success ! The results match .")
else :
    print (" Error : Results do not match .")
