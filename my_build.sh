file=./GEMM_MMA/my_cutlass_gemm.cu
#file=./GEMM_MMA/gemm_ldgsts.cu
#file=./GEMM_MMA/gemm_bfco.cu
#file=./GEMM_MMA/gemm_reg.cu
file=./GEMM_MMA/gemm_low_level.cu
#file=./examples/cute/tutorial/sgemm_sm80.cu
#file=./GEMM_MMA/cute_layout.cu
#file=./GEMM_MMA/tiled_copy.cu
# file=./GEMM_MMA/print_test_a100.cu
#file=./GEMM_MMA/sgemm2.cu

executable=./gemm_example

compile(){
#DEBUG_MODE='-g -G'
target=gemm_example
if [ $1 ];then
  file=$1
  if [ $2 ];then
    target=$2
  fi
fi
rm $target
echo "Current compile file=$file and target=$target"
nvcc $DEBUG_MODE -I/home/test/My_Projects/cutlass/include \
    -I/home/test/My_Projects/cutlass/tools/util/include \
    --std=c++17 \
    -arch=sm_80 $file -o $target --expt-relaxed-constexpr |& tee build.log
./$target |& tee $target.log
}

run(){
target=gemm_example
./$target |& tee $target.log
}


profile_system(){
if [ $1 ];then
  executable=$1
fi
echo "Current executable=$executable"

nsys profile \
  --output=cutlass_kernel_report \
  $executable 

mv cutlass_kernel_report.nsys-rep ${executable}.nsys-rep

#  --trace=cuda,nvtx,osrt \
#  --capture-range=cudaProfilerApi \
#  --capture-range-end=stop \
}

profile_report(){
if [ $1 ];then
  executable=$1
fi
echo "Current executable=$executable"

if [ $2 ];then
  report_name=$2.ncu-rep
else
  report_name=my_report.ncu-rep
fi
echo "Renerate report ${report_name}!"
#nv-nsight-cu-cli \
ncu \
  -f \
  --set full \
  --export $report_name \
  $executable
#nv-nsight-cu-cli --target-processes all \
#  --metrics achieved_occupancy,sm__warps_active.avg,sm__inst_executed.avg,dram__bytes.avg \
#  --kernel-regex "GEMM_MMA.*" \
#  --export report_cutlass_mma.ncu-rep \
#  $executable
}

profile_ncu_bank_conflict(){
if [ $1 ];then
  executable=$1
fi
echo "Current executable=$executable"
  ncu --metrics \
       l1tex__data_bank_conflicts_pipe_lsu_mem_shared,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum \
       $executable
}


#target_source=./my_transpose_opt.cu
#target=trans_opt

#nvcc -lineinfo ${target_source} -o ${target}
compile $file my_gemm_12812816
#run
#compile GEMM_MMA/cute_layout.cu cute_layout
#profile_system $executable
#profile_report ./${target} ${target}
#profile_ncu_bank_conflict 
#profile_ncu_bank_conflict ./trans_opt
