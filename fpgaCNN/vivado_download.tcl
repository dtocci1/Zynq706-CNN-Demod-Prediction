set CHAINPOSITION 1
open_hw
connect_hw_server
current_hw_target [get_hw_targets *]
open_hw_target
current_hw_device [lindex [get_hw_devices] $CHAINPOSITION]
refresh_hw_device -update_hw_probes false [lindex [get_hw_devices] $CHAINPOSITION]
set_property PROGRAM.FILE {C:\ProgramData\MATLAB\SupportPackages\R2021a\toolbox\dnnfpga\supportpackages\xilinx\bitstreams\zc706_single.bit} [lindex [get_hw_devices] $CHAINPOSITION]
program_hw_devices [lindex [get_hw_devices] $CHAINPOSITION]
refresh_hw_device [lindex [get_hw_devices] $CHAINPOSITION]
close_hw_target [get_hw_targets *]
close_hw
exit
