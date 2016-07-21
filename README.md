# The-basic-SDH-algorithm-implementation-for-3D-data
Introduction :
This project calculates the spatial distance histogram of a set of 3D points using efficient CUDA
code. To optimize the code, I used various techniques like converting Array of Structure(AoS) to
Structure of Array(SoA), manipulating data layout and privatization. This report also explains the
performance limitation of the previous version of this project.

Project 1 :
In this project, I implemented the spatial distance histogram for a multi-threaded CUDA kernel
function. The CUDA kernel computes the distance counts in all buckets of the SDH, by making each
thread work on input data point. After all the points are processed, the resulting SDH is copied back to
a host data structure.

Project 1 Performance Limitation :
The execution time for 256 threads in each block is 96.53613 msec which is 22x faster than the
single threaded CPU execution time that is 2144 msec. On further analysis, I found that memory
latency and divergence slows down the execution time of the each thread.

Project 2 :
The main objective of this project is to improve the performance of the project 1. To enhance
performance, few techniques are used such as changing the data layout, privatization and conversion of
AoS to SoA. The execution time I got for 256 threads in a block is 14.80976 msec which is 6x faster
than the previous GPU version.
• Data layout to reduce thread divergence :
In the project 1, work is not equally divided among the threads. Few threads are overloaded
and few are not. To minimize, I manipulated the data layout from triangle to rectangle
access.
• Privatization for fast access:
To avoid multiple threads from accessing the same global memory location, privatization is
used. Private variable size should be small enough to fit into shared memory. Atomic
operation is expensive on global memory compared to incrementing per-block shared
variable. Once the privatized histogram have been accumulated in shared memory, I reduced
each histogram element by adding it to the global memory output.
• AoS to SoA to minimize memory latency :
Array of Structures is affected by memory access pattern whereas SoA gives a better
performance by accessing in contiguous memory.
Result:
Therefore through this project I implemented a efficient version of CUDA code for SDH
algorithm. I got a 6x boost with project 1 and 144x boost compared to single thread CPU code.
