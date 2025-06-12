//Copyright (C)2014-2025 GOWIN Semiconductor Corporation.
//All rights reserved.
//File Title: Timing Constraints file
//Tool Version: V1.9.10.01 
//Created Time: 2025-06-12 12:10:09
create_clock -name sys_clk -period 40 -waveform {0 20} [get_ports {sys_clk}]
