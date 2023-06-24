```
Sequential, matrix mul (dot prod), order 1024 on host CPU,       1.9470 seconds at 1103.0 MFLOPS 

Better sequential, matrix mul (dot prod), order 1024 on host CPU,        0.3260 seconds at 6587.4 MFLOPS 

===== Device 'Intel(R) Core(TM) i9-9980HK CPU @ 2.40GHz' start =====
OpenCL, matrix mul 'C(i,j) per work item', order 1024,   0.1480 seconds at 14510.0 MFLOPS 
OpenCL, matrix mul 'C row per work item, 16 units', order 1024,  0.1450 seconds at 14810.2 MFLOPS 
OpenCL, matrix mul 'C row per work item, any units', order 1024,         0.1380 seconds at 15561.5 MFLOPS 
OpenCL, matrix mul 'C row per work item private memory, any units', order 1024,  0.4520 seconds at 4751.1 MFLOPS 
OpenCL, matrix mul 'C row per work item with local column, any units', order 1024,       0.5200 seconds at 4129.8 MFLOPS 
===== Device 'Intel(R) Core(TM) i9-9980HK CPU @ 2.40GHz' done =====

===== Device 'Intel(R) UHD Graphics 630' start =====
OpenCL, matrix mul 'C(i,j) per work item', order 1024,   0.4170 seconds at 5149.8 MFLOPS 
OpenCL, matrix mul 'C row per work item, 16 units', order 1024,  0.3640 seconds at 5899.7 MFLOPS 
OpenCL, matrix mul 'C row per work item, any units', order 1024,         0.3280 seconds at 6547.2 MFLOPS 
OpenCL, matrix mul 'C row per work item private memory, 16 units', order 1024,   0.3210 seconds at 6690.0 MFLOPS 
OpenCL, matrix mul 'C row per work item private memory, any units', order 1024,  0.3240 seconds at 6628.0 MFLOPS 
OpenCL, matrix mul 'C row per work item with local column, 16 units', order 1024,        0.2330 seconds at 9216.7 MFLOPS 
OpenCL, matrix mul 'C row per work item with local column, any units', order 1024,       0.3710 seconds at 5788.4 MFLOPS 
OpenCL, matrix mul 'Block fast, block size 16', order 1024,      0.0560 seconds at 38347.9 MFLOPS 
===== Device 'Intel(R) UHD Graphics 630' done =====

===== Device 'AMD Radeon Pro 5300M Compute Engine' start =====
OpenCL, matrix mul 'C(i,j) per work item', order 1024,   0.2510 seconds at 8555.7 MFLOPS 
OpenCL, matrix mul 'C row per work item, 16 units', order 1024,  1.5310 seconds at 1402.7 MFLOPS 
OpenCL, matrix mul 'C row per work item, any units', order 1024,         2.0800 seconds at 1032.4 MFLOPS 
OpenCL, matrix mul 'C row per work item private memory, 16 units', order 1024,   0.1980 seconds at 10845.9 MFLOPS 
OpenCL, matrix mul 'C row per work item private memory, any units', order 1024,  0.1880 seconds at 11422.8 MFLOPS 
OpenCL, matrix mul 'C row per work item with local column, 16 units', order 1024,        0.1760 seconds at 12201.6 MFLOPS 
OpenCL, matrix mul 'C row per work item with local column, any units', order 1024,       0.1980 seconds at 10845.9 MFLOPS 
OpenCL, matrix mul 'Block fast, block size 16', order 1024,      0.0110 seconds at 195225.8 MFLOPS 
===== Device 'AMD Radeon Pro 5300M Compute Engine' done =====
```
